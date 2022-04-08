#include "CodeGenStateMachine.h"
#include "MemrefTypes.h"
#include "Dialect.h"
#include "TreeTilingUtils.h"
#include "TiledTree.h"
#include "schedule.h"

namespace mlir
{
namespace decisionforest
{
    Value CreateZeroVectorIndexConst(ConversionPatternRewriter &rewriter, Location location, int32_t tileSize) {
        Value zeroConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
        auto vectorType = VectorType::get(tileSize, rewriter.getIndexType());
        auto vectorValue = rewriter.create<vector::BroadcastOp>(location, vectorType, zeroConst);
        return vectorValue;
    }

    Value ReduceComparisonResultVectorToInt(Value comparisonResult, int32_t tileSize, ConversionPatternRewriter &rewriter, Location location) {
        auto i32VectorType = VectorType::get(tileSize, rewriter.getI32Type());
        auto comparisonExtended = rewriter.create<arith::ExtUIOp>(location, i32VectorType, comparisonResult);

        auto zeroI32Const = rewriter.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI32Type());
        auto shiftVector = static_cast<Value>(rewriter.create<vector::BroadcastOp>(location, i32VectorType, zeroI32Const));
        for (int32_t shift=0, pos=tileSize-1 ; shift<tileSize; ++shift, --pos) {
            auto shiftValConst = rewriter.create<arith::ConstantIntOp>(location, int64_t(shift), rewriter.getI32Type());
            shiftVector = rewriter.create<vector::InsertOp>(location, static_cast<Value>(shiftValConst), 
                                                            static_cast<Value>(shiftVector), ArrayRef<int64_t>({ pos }));
        }

        auto leftShift = rewriter.create<arith::ShLIOp>(location, i32VectorType, comparisonExtended, shiftVector);
        auto kind = rewriter.getStringAttr("add");
        auto sum = rewriter.create<vector::ReductionOp>(location, rewriter.getI32Type(), kind, static_cast<Value>(leftShift), ValueRange{ });
        auto index = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(sum));
        return index;
    }

    Value ReduceComparisonResultVectorToInt_Bitcast(Value comparisonResult, int32_t tileSize, ConversionPatternRewriter &rewriter, Location location) {
    auto bitcastVectorType = VectorType::get(1, rewriter.getIntegerType(tileSize));
    auto bitcastOp = rewriter.create<vector::BitCastOp>(location, bitcastVectorType, comparisonResult);
    auto zeroConst = rewriter.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI32Type());
    auto integerResult = rewriter.create<vector::ExtractElementOp>(location, static_cast<Value>(bitcastOp), static_cast<Value>(zeroConst));
    auto zeroExtend = rewriter.create<arith::ExtUIOp>(location, integerResult, rewriter.getI64Type()); 
    auto index = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(zeroExtend));
    return index;
    }

    void InsertPrintVectorOp(ConversionPatternRewriter &rewriter, Location location, int32_t kind, int32_t bitWidth, int32_t tileSize, Value vectorValue) {
        auto tileSizeConst = rewriter.create<arith::ConstantIntOp>(location, tileSize, rewriter.getI32Type());
        auto kindConst = rewriter.create<arith::ConstantIntOp>(location, kind, rewriter.getI32Type());
        auto bitWidthConst = rewriter.create<arith::ConstantIntOp>(location, bitWidth, rewriter.getI32Type());
        std::vector<Value> vectorValues;
        for (int32_t i=0; i<tileSize ; ++i) {
            auto iConst = rewriter.create<arith::ConstantIntOp>(location, int64_t(i), rewriter.getI32Type());
            auto ithValue = rewriter.create<vector::ExtractElementOp>(location, vectorValue, iConst);
            vectorValues.push_back(ithValue);
        }
        rewriter.create<decisionforest::PrintVectorOp>(location, kindConst, bitWidthConst, tileSizeConst, ValueRange(vectorValues));
    }

    ScalarTraverseTileCodeGenerator:: ScalarTraverseTileCodeGenerator(Value treeMemref, Value rowMemref, Value node, Type resultType, Representation representation) {
      m_treeMemref = treeMemref;
      m_rowMemref = rowMemref;
      m_treeMemrefType = treeMemref.getType().cast<MemRefType>();
      m_nodeToTraverse = node;
      m_resultType = resultType;
      m_state = kLoadThreshold;
      m_representation = representation;
    }

    bool ScalarTraverseTileCodeGenerator::EmitNext(ConversionPatternRewriter& rewriter, Location& location) {
      auto treeTileType = m_treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
      auto featureIndexType = treeTileType.getIndexElementType();
      auto thresholdType = treeTileType.getThresholdElementType();

      // Assert tile size is 1
      assert (treeTileType.getTileSize() == 1);
      
      switch (m_state) {
        case kLoadThreshold:
          {
            m_nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, rewriter.getIndexType(), m_treeMemref, m_nodeToTraverse);
            if (decisionforest::InsertDebugHelpers) {
              rewriter.create<decisionforest::PrintTreeNodeOp>(location, m_nodeIndex);
            }
            m_loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location, thresholdType, m_treeMemref, static_cast<Value>(m_nodeIndex));
            m_state = kLoadFeatureIndex;
          }
          break;
        case kLoadFeatureIndex:
          {
            m_loadFeatureIndexOp = rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(location, featureIndexType, m_treeMemref, static_cast<Value>(m_nodeIndex));

            if (m_representation == kSparse) {
              // Load the child index
              m_loadChildIndex = rewriter.create<decisionforest::LoadChildIndexOp>(location, treeTileType.getChildIndexType(), m_treeMemref, static_cast<Value>(m_nodeIndex));
            }
            m_state = kLoadFeature;
          }
          break;
        case kLoadFeature:
          {
            auto rowMemrefType = m_rowMemref.getType().cast<MemRefType>();
            auto rowIndex = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(m_loadFeatureIndexOp));
            auto zeroIndex = rewriter.create<arith::ConstantIndexOp>(location, 0);
            m_loadFeatureOp = rewriter.create<memref::LoadOp>(
                                                              location,
                                                              rowMemrefType.getElementType(),
                                                              m_rowMemref,
                                                              ValueRange({static_cast<Value>(zeroIndex), static_cast<Value>(rowIndex)}));
            if(decisionforest::InsertDebugHelpers) {
              rewriter.create<decisionforest::PrintComparisonOp>(location, m_loadFeatureOp, m_loadThresholdOp, m_loadFeatureIndexOp);
            }
            m_state = kCompare;
          }
          break;
        case kCompare:
          {
            // TODO we need a cast here to make sure the threshold and the row element are the same type. The op expects both operands to be the same type.
            auto comparison = rewriter.create<arith::CmpFOp>(
                location,
                mlir::arith::CmpFPredicate::UGE,
                static_cast<Value>(m_loadFeatureOp),
                static_cast<Value>(m_loadThresholdOp));
            m_comparisonUnsigned = rewriter.create<arith::ExtUIOp>(location, rewriter.getI32Type(), static_cast<Value>(comparison));
            m_state = kNextNode;
          }
          break;
        case kNextNode:
          {
            Value newIndex;
            if (m_representation == kArray){
              // index = 2*index + 1 + result
              auto oneConstant = rewriter.create<arith::ConstantIndexOp>(location, 1);
              auto twoConstant = rewriter.create<arith::ConstantIndexOp>(location, 2);
              auto twoTimesIndex = rewriter.create<arith::MulIOp>(location, rewriter.getIndexType(), static_cast<Value>(m_nodeIndex), static_cast<Value>(twoConstant));
              auto twoTimesIndexPlus1 = rewriter.create<arith::AddIOp>(location, rewriter.getIndexType(), static_cast<Value>(twoTimesIndex), static_cast<Value>(oneConstant));
              auto comparisonResultIndex = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(m_comparisonUnsigned));
              newIndex = rewriter.create<arith::AddIOp>(location, rewriter.getIndexType(), static_cast<Value>(twoTimesIndexPlus1), static_cast<Value>(comparisonResultIndex));
            }
            else if (m_representation == kSparse) {
              assert(m_loadChildIndex && "ChildIndex should not be empty");

              auto childIndex = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(m_loadChildIndex));
              // index = childIndex + result
              auto comparisonResultIndex = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(m_comparisonUnsigned));
              newIndex = rewriter.create<arith::AddIOp>(location, rewriter.getIndexType(), static_cast<Value>(childIndex), static_cast<Value>(comparisonResultIndex));
            }
            else {
              assert(false && "Invalid representation!");
            }
            
            // node = indexToNode(index)
            m_result = rewriter.create<decisionforest::IndexToNodeOp>(location, m_resultType, m_treeMemref, static_cast<Value>(newIndex));
            
            m_state = kDone;
            return false;
          }
        case kDone:
          return false;
        default:
          assert(false && "Invalid state!");
          return false;
      }

      return true;
    }

    std::vector<Value> ScalarTraverseTileCodeGenerator::GetResult() {
      assert (m_state == kDone);
      std::vector<Value> results;
      results.push_back(m_result);
      return results;
    }

    VectorTraverseTileCodeGenerator::VectorTraverseTileCodeGenerator(Value tree, Value treeMemref, Value rowMemref, Value node, Type resultType, Representation representation, std::function<Value(Value)> getLutFunc) {
      m_tree = tree;
      m_treeMemref = treeMemref;
      m_rowMemref = rowMemref;
      m_treeMemrefType = treeMemref.getType().cast<MemRefType>();
      m_nodeToTraverse = node;
      m_resultType = resultType;
      m_state = kLoadThreshold;
      m_representation = representation;
      m_getLutFunc = getLutFunc;

      m_treeTileType = m_treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
      auto featureIndexType = m_treeTileType.getIndexFieldType();
      m_featureIndexVectorType = featureIndexType.cast<VectorType>();
      assert(m_featureIndexVectorType);
      
      auto thresholdType = m_treeTileType.getThresholdFieldType();
      m_thresholdVectorType = thresholdType.cast<VectorType>();
      assert(m_thresholdVectorType);

      m_tileShapeType = m_treeTileType.getTileShapeType();
      m_tileSize = m_treeTileType.getTileSize();

      assert (m_tileSize > 1);
    }

    bool VectorTraverseTileCodeGenerator::EmitNext(ConversionPatternRewriter& rewriter, Location& location) {
      switch (m_state) {
        case kLoadThreshold:
          {
            m_nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, rewriter.getIndexType(), m_treeMemref, m_nodeToTraverse);
            if (decisionforest::InsertDebugHelpers) {
              rewriter.create<decisionforest::PrintTreeNodeOp>(location, m_nodeIndex);
            }
            // Load threshold
            m_loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location, m_thresholdVectorType, m_treeMemref, static_cast<Value>(m_nodeIndex));
            if (decisionforest::InsertDebugHelpers) {
              Value vectorVal = m_loadThresholdOp;
              if (!m_thresholdVectorType.getElementType().isF64()) {
                auto doubleVectorType = mlir::VectorType::get({ m_tileSize }, rewriter.getF64Type());
                vectorVal = rewriter.create<arith::ExtFOp>(location, doubleVectorType, m_loadThresholdOp);
              }
              InsertPrintVectorOp(rewriter, location, 0 /*fp kind*/, 64, m_tileSize, vectorVal);
            }
            m_state = kLoadFeatureIndex;
          }
          break;
        case kLoadFeatureIndex:
          {
            m_loadFeatureIndexOp = rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(location, m_featureIndexVectorType, m_treeMemref, static_cast<Value>(m_nodeIndex));
            if (decisionforest::InsertDebugHelpers) {
              InsertPrintVectorOp(
                  rewriter,
                  location,
                  1 /*int kind*/,
                  m_featureIndexVectorType.getElementType().getIntOrFloatBitWidth(),
                  m_tileSize, static_cast<Value>(m_loadFeatureIndexOp));
            }

            m_state = kLoadTileShape;
          }
          break;
        case kLoadTileShape:
          {
            auto loadTileShapeOp = rewriter.create<decisionforest::LoadTileShapeOp>(location, m_tileShapeType, m_treeMemref, static_cast<Value>(m_nodeIndex));
            m_loadTileShapeIndexOp = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(loadTileShapeOp));

            m_state = m_representation == kSparse ? kLoadChildIndex : kLoadFeature;
          }
          break;
        case kLoadChildIndex:
          {
            assert(m_representation == kSparse && "Representation should be sparse");
            if (decisionforest::RemoveExtraHopInSparseRepresentation) {
              // Load both the child and 
              auto childIndicesVectorType = VectorType::get({ 2 }, m_treeTileType.getChildIndexType());
              auto loadChildandLeafIndexOp = rewriter.create<decisionforest::LoadChildAndLeafIndexOp>(location, childIndicesVectorType, m_treeMemref, static_cast<Value>(m_nodeIndex));
              auto indexVectorType = VectorType::get({ 2 }, rewriter.getIndexType());
              m_childIndex = rewriter.create<arith::IndexCastOp>(location, indexVectorType, static_cast<Value>(loadChildandLeafIndexOp));
              
              auto loadLeafBitMask = rewriter.create<decisionforest::LoadLeafBitMaskOp>(location, m_treeTileType.getLeafBitMaskType(), m_treeMemref, static_cast<Value>(m_nodeIndex));
              m_leafBitMask = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(loadLeafBitMask));
            }
            else {
              // Load the child index
              auto loadChildIndexOp = rewriter.create<decisionforest::LoadChildIndexOp>(location, m_treeTileType.getChildIndexType(), m_treeMemref, static_cast<Value>(m_nodeIndex));
              m_childIndex = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(loadChildIndexOp));
            }
            m_state = kLoadFeature;
          }
          break;
        case kLoadFeature:
          {
            auto rowMemrefType = m_rowMemref.getType().cast<MemRefType>();
            auto vectorIndexType = VectorType::get({ m_tileSize }, rewriter.getIndexType());
            auto rowIndex = rewriter.create<arith::IndexCastOp>(location, vectorIndexType, static_cast<Value>(m_loadFeatureIndexOp));
            auto zeroIndex = rewriter.create<arith::ConstantIndexOp>(location, 0);
            // auto zeroIndexVector = rewriter.create<vector::BroadcastOp>(location, vectorIndexType, zeroIndex);

            auto featuresVectorType = VectorType::get({ m_tileSize }, rowMemrefType.getElementType());
            auto oneI1Const = rewriter.create<arith::ConstantIntOp>(location, 1, rewriter.getI1Type());
            auto i1VectorType = VectorType::get(m_tileSize, rewriter.getI1Type());
            auto mask = rewriter.create<vector::BroadcastOp>(location, i1VectorType, oneI1Const);

            Value zeroPassThruConst;
            if (rowMemrefType.getElementType().isa<mlir::Float64Type>())
              zeroPassThruConst = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat(0.0), rowMemrefType.getElementType().cast<FloatType>());
            else if(rowMemrefType.getElementType().isa<mlir::Float32Type>())
              zeroPassThruConst = rewriter.create<arith::ConstantFloatOp>(location, llvm::APFloat((float)0.0), rowMemrefType.getElementType().cast<FloatType>());
            else
              assert(false && "Unsupported floating point type");
            auto zeroPassThruVector = rewriter.create<vector::BroadcastOp>(location, featuresVectorType, zeroPassThruConst);
            
            m_features = rewriter.create<vector::GatherOp>(
              location,
              featuresVectorType,
              m_rowMemref,
              ValueRange({static_cast<Value>(zeroIndex), static_cast<Value>(zeroIndex)}),
              rowIndex,
              mask,
              zeroPassThruVector);

              if (decisionforest::InsertDebugHelpers) {
                Value vectorVal = m_features;
                if (!featuresVectorType.getElementType().isF64()) {
                  auto doubleVectorType = mlir::VectorType::get({ m_tileSize }, rewriter.getF64Type());
                  vectorVal = rewriter.create<arith::ExtFOp>(location, doubleVectorType, m_features);
                }

                InsertPrintVectorOp(
                  rewriter,
                  location,
                  0 /*fp kind*/,
                  64, // double
                  m_tileSize,
                  vectorVal);
              }
            m_state = kCompare;
          }
          break;  
        case kCompare:
          {
            auto comparison = rewriter.create<arith::CmpFOp>(location,  mlir::arith::CmpFPredicate::ULT, static_cast<Value>(m_features), static_cast<Value>(m_loadThresholdOp));
            if (decisionforest::UseBitcastForComparisonOutcome)
              m_comparisonIndex = ReduceComparisonResultVectorToInt_Bitcast(comparison, m_tileSize, rewriter, location);
            else
              m_comparisonIndex = ReduceComparisonResultVectorToInt(comparison, m_tileSize, rewriter, location);
            
            // TODO This needs a different print routine!
            // if(decisionforest::InsertDebugHelpers) {
            //   rewriter.create<decisionforest::PrintComparisonOp>(location, feature, loadThresholdOp, loadFeatureIndexOp);
            // }

            m_state = kNextNode;
          }
          break;
        case kNextNode:
          {
            // Load the child index from the LUT
            auto lutValue = m_getLutFunc(m_tree);
            auto childIndexInt = rewriter.create<memref::LoadOp>(location, lutValue, ValueRange{m_loadTileShapeIndexOp, m_comparisonIndex});
            auto childIndex = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(childIndexInt));

            Value newIndex;
            if (m_representation == kArray) {
              auto oneConstant = rewriter.create<arith::ConstantIndexOp>(location, 1);
              auto tileSizeConstant = rewriter.create<arith::ConstantIndexOp>(location, m_tileSize+1);
              auto tileSizeTimesIndex = rewriter.create<arith::MulIOp>(location, rewriter.getIndexType(), static_cast<Value>(m_nodeIndex), static_cast<Value>(tileSizeConstant));
              auto tileSizeTimesIndexPlus1 = rewriter.create<arith::AddIOp>(location, rewriter.getIndexType(), static_cast<Value>(tileSizeTimesIndex), static_cast<Value>(oneConstant));
              
              newIndex = rewriter.create<arith::AddIOp>(location, rewriter.getIndexType(), static_cast<Value>(tileSizeTimesIndexPlus1), static_cast<Value>(childIndex));
            }
            else if (m_representation == kSparse) {
              if (decisionforest::RemoveExtraHopInSparseRepresentation) {
                // Shift so that the bit of leafBitMask corresponding to the 
                auto shift = rewriter.create<arith::ShRUIOp>(location, m_leafBitMask.getType(), m_leafBitMask, childIndex);

                // Add child index to the child and leaf index vector
                auto childIndexVectorValue = rewriter.create<vector::BroadcastOp>(location, m_childIndex.getType(), childIndex);
                auto potentialNewIndices = rewriter.create<arith::AddIOp>(location, m_childIndex, childIndexVectorValue);

                // Find the value of the bit
                auto oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
                auto bitValue = rewriter.create<arith::AndIOp>(location, rewriter.getIndexType(), shift, oneIndexConst);

                // Extract the right index
                newIndex = rewriter.create<vector::ExtractElementOp>(location, potentialNewIndices, bitValue);

                if (decisionforest::InsertDebugHelpers) {
                  // (is leaf bit, lutLookup result, new index)
                  auto zeroVector = CreateZeroVectorIndexConst(rewriter, location, 3);
                  auto zeroConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
                  auto elem0Set = rewriter.create<vector::InsertElementOp>(location, bitValue, zeroVector, zeroConst);
                  auto oneConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
                  auto elem1Set = rewriter.create<vector::InsertElementOp>(location, childIndex, elem0Set, oneConst);
                  auto twoConst = rewriter.create<arith::ConstantIndexOp>(location, 2);
                  auto elem2Set = rewriter.create<vector::InsertElementOp>(location, newIndex, elem1Set, twoConst);
                  InsertPrintVectorOp(rewriter, location, 1, 64, 3, elem2Set);
                }    
              }
              else {
                newIndex = rewriter.create<arith::AddIOp>(location, rewriter.getIndexType(), static_cast<Value>(m_childIndex), static_cast<Value> (childIndex));

                if (decisionforest::InsertDebugHelpers) {
                  // (child base index, lutLookup result, new index)
                  auto zeroVector = CreateZeroVectorIndexConst(rewriter, location, 3);
                  auto zeroConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
                  auto elem0Set = rewriter.create<vector::InsertElementOp>(location, childIndex, zeroVector, zeroConst);
                  auto oneConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
                  auto elem1Set = rewriter.create<vector::InsertElementOp>(location, childIndex, elem0Set, oneConst);
                  auto twoConst = rewriter.create<arith::ConstantIndexOp>(location, 2);
                  auto elem2Set = rewriter.create<vector::InsertElementOp>(location, newIndex, elem1Set, twoConst);
                  InsertPrintVectorOp(rewriter, location, 1, 64, 3, elem2Set);
                }
              }
            }
            
            // node = indexToNode(index)
            m_result = rewriter.create<decisionforest::IndexToNodeOp>(location, m_resultType, m_treeMemref, static_cast<Value>(newIndex));

            m_state = kDone;
            return false;
          }
        case kDone:
          return false;
        default:
          assert(false && "Invalid state!");
          return false;
      }

      return true;
    }

    std::vector<Value> VectorTraverseTileCodeGenerator::GetResult() {
      assert (m_state == kDone);
      std::vector<Value> results;
      results.push_back(m_result);
      return results;
    }
}
}