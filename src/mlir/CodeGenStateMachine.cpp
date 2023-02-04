#include "CodeGenStateMachine.h"
#include "MemrefTypes.h"
#include "Dialect.h"
#include "TreeTilingUtils.h"
#include "TiledTree.h"
#include "schedule.h"
#include "LIRLoweringHelpers.h"

using namespace mlir::decisionforest::helpers;

namespace mlir
{
namespace decisionforest
{
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
        auto kind = vector::CombiningKind::ADD;
        auto sum = rewriter.create<vector::ReductionOp>(location, kind, static_cast<Value>(leftShift));
        auto index = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(sum));
        return index;
    }

    Value ReduceComparisonResultVectorToInt_Bitcast(Value comparisonResult, int32_t tileSize, ConversionPatternRewriter &rewriter, Location location) {
      auto bitcastVectorType = VectorType::get(1, rewriter.getIntegerType(tileSize));
      auto bitcastOp = rewriter.create<vector::BitCastOp>(location, bitcastVectorType, comparisonResult);
      auto zeroConst = rewriter.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI32Type());
      auto integerResult = rewriter.create<vector::ExtractElementOp>(location, static_cast<Value>(bitcastOp), static_cast<Value>(zeroConst));
      auto zeroExtend = rewriter.create<arith::ExtUIOp>(location, rewriter.getI64Type(), integerResult); 
      auto index = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(zeroExtend));
      return index;
    }

    ScalarTraverseTileCodeGenerator:: ScalarTraverseTileCodeGenerator(Value rowMemref, Value node, 
                                                                      Type resultType, 
                                                                      std::shared_ptr<IRepresentation> representation,
                                                                      Value tree) {
      m_rowMemref = rowMemref;
      m_nodeToTraverse = node;
      m_resultType = resultType;
      m_state = kLoadThreshold;
      m_representation = representation;
      m_tree = tree;
    }

    bool ScalarTraverseTileCodeGenerator::EmitNext(ConversionPatternRewriter& rewriter, Location& location) {
      auto featureIndexType = m_representation->GetIndexElementType();
      auto thresholdType = m_representation->GetThresholdElementType();

      // Assert tile size is 1
      assert (m_representation->GetTileSize() == 1);
      
      switch (m_state) {
        case kLoadThreshold:
          {
            // TODO_Ashwin Remove the memref argument here
            m_nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, 
                                                                         rewriter.getIndexType(),
                                                                         m_representation->GetThresholdsMemref(m_tree),
                                                                         m_nodeToTraverse);
            if (decisionforest::InsertDebugHelpers) {
              rewriter.create<decisionforest::PrintTreeNodeOp>(location, m_nodeIndex);
            }
            m_loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location,
                                                                                      thresholdType, 
                                                                                      m_representation->GetThresholdsMemref(m_tree),
                                                                                      static_cast<Value>(m_nodeIndex));
            m_state = kLoadFeatureIndex;
          }
          break;
        case kLoadFeatureIndex:
          {
            m_loadFeatureIndexOp = rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(location,
                                                                                             featureIndexType,
                                                                                             m_representation->GetFeatureIndexMemref(m_tree),
                                                                                             static_cast<Value>(m_nodeIndex));

            m_extraLoads = m_representation->GenerateExtraLoads(location, rewriter, m_tree, m_nodeIndex);
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
            auto comparisonResultIndex = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(m_comparisonUnsigned));
            Value newIndex = m_representation->GenerateMoveToChild(location, rewriter, m_nodeIndex, comparisonResultIndex, 1, m_extraLoads);

            // TODO_Ashwin Remove the reference to the memref below
            // node = indexToNode(index)
            m_result = rewriter.create<decisionforest::IndexToNodeOp>(location, 
                                                                      m_resultType,
                                                                      m_representation->GetThresholdsMemref(m_tree),
                                                                      static_cast<Value>(newIndex));
            
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

    VectorTraverseTileCodeGenerator::VectorTraverseTileCodeGenerator(Value tree, Value rowMemref, Value node, 
                                                                     Type resultType, std::shared_ptr<IRepresentation> representation, 
                                                                     std::function<Value(Value)> getLutFunc) {
      m_tree = tree;
      m_rowMemref = rowMemref;
      m_nodeToTraverse = node;
      m_resultType = resultType;
      m_state = kLoadThreshold;
      m_representation = representation;
      m_getLutFunc = getLutFunc;

      auto featureIndexType = m_representation->GetIndexFieldType();
      m_featureIndexVectorType = featureIndexType.cast<VectorType>();
      assert(m_featureIndexVectorType);
      
      auto thresholdType = m_representation->GetThresholdFieldType();
      m_thresholdVectorType = thresholdType.cast<VectorType>();
      assert(m_thresholdVectorType);

      m_tileShapeType = m_representation->GetTileShapeType();
      m_tileSize = m_representation->GetTileSize();

      assert (m_tileSize > 1);
    }

    bool VectorTraverseTileCodeGenerator::EmitNext(ConversionPatternRewriter& rewriter, Location& location) {
      switch (m_state) {
        case kLoadThreshold:
          {
            // TODO_Ashwin remove reference to memref
            m_nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, 
                                                                         rewriter.getIndexType(), 
                                                                         m_representation->GetThresholdsMemref(m_tree),
                                                                         m_nodeToTraverse);
            if (decisionforest::InsertDebugHelpers) {
              rewriter.create<decisionforest::PrintTreeNodeOp>(location, m_nodeIndex);
            }
            // Load threshold
            m_loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location, 
                                                                                      m_thresholdVectorType,
                                                                                      m_representation->GetThresholdsMemref(m_tree),
                                                                                      static_cast<Value>(m_nodeIndex));
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
            m_loadFeatureIndexOp = rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(location, 
                                                                                             m_featureIndexVectorType,
                                                                                             m_representation->GetFeatureIndexMemref(m_tree),
                                                                                             static_cast<Value>(m_nodeIndex));
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
            auto loadTileShapeOp = rewriter.create<decisionforest::LoadTileShapeOp>(location, 
                                                                                    m_tileShapeType,
                                                                                    m_representation->GetTileShapeMemref(m_tree),
                                                                                    static_cast<Value>(m_nodeIndex));
            m_loadTileShapeIndexOp = rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(), static_cast<Value>(loadTileShapeOp));

            m_state = kLoadChildIndex;
          }
          break;
        case kLoadChildIndex:
          {
            m_extraLoads = m_representation->GenerateExtraLoads(location, rewriter, m_tree, m_nodeIndex);
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

            Value newIndex = m_representation->GenerateMoveToChild(location, rewriter, m_nodeIndex, childIndex, m_tileSize, m_extraLoads);
            
            // node = indexToNode(index)
            // TODO_Ashwin Remove memref reference
            m_result = rewriter.create<decisionforest::IndexToNodeOp>(location, 
                                                                      m_resultType,
                                                                      m_representation->GetThresholdsMemref(m_tree),
                                                                      static_cast<Value>(newIndex));

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