#include "Dialect.h"
#include "LIRLoweringHelpers.h"
#include "MemrefTypes.h"
#include "OpLoweringUtils.h"
#include "TiledTree.h"
#include "TreeTilingUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

#include "schedule.h"

#include "CodeGenStateMachine.h"

using namespace mlir::decisionforest::helpers;

namespace mlir {
namespace decisionforest {
// ===---------------------------------------------------=== //
// Helper functions
// ===---------------------------------------------------=== //

mlir::gpu::KernelDim3 GetThreadID(mlir::Operation *op);
mlir::gpu::KernelDim3 GetBlockID(mlir::Operation *op);

Value ReduceComparisonResultVectorToInt(Value comparisonResult,
                                        int32_t tileSize,
                                        ConversionPatternRewriter &rewriter,
                                        Location location) {
  auto i32VectorType = VectorType::get(tileSize, rewriter.getI32Type());
  auto comparisonExtended = rewriter.create<arith::ExtUIOp>(
      location, i32VectorType, comparisonResult);

  auto zeroI32Const = rewriter.create<arith::ConstantIntOp>(
      location, int64_t(0), rewriter.getI32Type());
  auto shiftVector = static_cast<Value>(rewriter.create<vector::BroadcastOp>(
      location, i32VectorType, zeroI32Const));
  for (int32_t shift = 0, pos = tileSize - 1; shift < tileSize;
       ++shift, --pos) {
    auto shiftValConst = rewriter.create<arith::ConstantIntOp>(
        location, int64_t(shift), rewriter.getI32Type());
    shiftVector = rewriter.create<vector::InsertOp>(
        location, static_cast<Value>(shiftValConst),
        static_cast<Value>(shiftVector), ArrayRef<int64_t>({pos}));
  }

  auto leftShift = rewriter.create<arith::ShLIOp>(
      location, i32VectorType, comparisonExtended, shiftVector);
  auto kind = vector::CombiningKind::ADD;
  auto sum = rewriter.create<vector::ReductionOp>(
      location, kind, static_cast<Value>(leftShift));
  auto index = rewriter.create<arith::IndexCastOp>(
      location, rewriter.getIndexType(), static_cast<Value>(sum));
  return index;
}

Value ReduceComparisonResultVectorToInt_Bitcast(
    Value comparisonResult, int32_t tileSize,
    ConversionPatternRewriter &rewriter, Location location) {
  auto bitcastVectorType =
      VectorType::get(1, rewriter.getIntegerType(tileSize));
  auto bitcastOp = rewriter.create<vector::BitCastOp>(
      location, bitcastVectorType, comparisonResult);
  auto zeroConst = rewriter.create<arith::ConstantIntOp>(location, int64_t(0),
                                                         rewriter.getI32Type());
  auto integerResult = rewriter.create<vector::ExtractElementOp>(
      location, static_cast<Value>(bitcastOp), static_cast<Value>(zeroConst));
  auto zeroExtend = rewriter.create<arith::ExtUIOp>(
      location, rewriter.getI64Type(), integerResult);
  auto index = rewriter.create<arith::IndexCastOp>(
      location, rewriter.getIndexType(), static_cast<Value>(zeroExtend));
  return index;
}

mlir::arith::CmpFPredicate
negateComparisonPredicate(mlir::arith::CmpFPredicateAttr cmpPredAttr) {
  auto cmpPred = cmpPredAttr.getValue();
  switch (cmpPred) {
  case arith::CmpFPredicate::ULT:
    return arith::CmpFPredicate::UGE;
  case arith::CmpFPredicate::UGE:
    return arith::CmpFPredicate::ULT;
  case arith::CmpFPredicate::UGT:
    return arith::CmpFPredicate::ULE;
  case arith::CmpFPredicate::ULE:
    return arith::CmpFPredicate::UGT;
  default:
    assert(false && "Unknown comparison predicate");
    return arith::CmpFPredicate::ULT;
  }
}

Operation *
FindGetRootOp(decisionforest::CooperativeTraverseTreeTileOp traverseTile) {
  // TODO_Ashwin This needs to become more general. Just hacking it
  // to work for now. Only handling cases where the traverse is in a while
  // loop and when its just a chain of calls
  const size_t npos = std::string::npos;
  auto owningWhileLoop = traverseTile->getParentOfType<scf::WhileOp>();
  Operation *definingOp = nullptr;
  if (owningWhileLoop) {
    // TODO_Ashwin a more elaborate matching of node arguments is
    // needed here. For now, just assuming that the argument number
    // of node value is the same in before and after (which is the case
    // for while loops that we generate)
    auto afterArguments = owningWhileLoop.getAfterArguments();
    size_t argNum = npos;
    for (size_t i = 0; i < afterArguments.size(); ++i) {
      if (afterArguments[i] == traverseTile.getNode()) {
        argNum = i;
        break;
      }
    }
    assert(argNum != npos);
    // auto beforeArguments = owningWhileLoop.getBeforeArguments();
    auto sourceNodeValue = owningWhileLoop.getOperand(argNum);
    definingOp = sourceNodeValue.getDefiningOp();
  } else {
    definingOp = traverseTile.getNode().getDefiningOp();
  }
  auto getRootOp = llvm::dyn_cast<decisionforest::GetRootOp>(definingOp);
  auto definingTraverseTile =
      llvm::dyn_cast<decisionforest::CooperativeTraverseTreeTileOp>(definingOp);
  if (getRootOp)
    return getRootOp;
  else if (definingTraverseTile)
    return FindGetRootOp(definingTraverseTile);

  assert(false && "Node values can only come from getRoot or traverseTile ops");
  return nullptr;
}

// ===---------------------------------------------------=== //
// ScalarTraverseTileCodeGenerator Methods
// ===---------------------------------------------------=== //

ScalarTraverseTileCodeGenerator::ScalarTraverseTileCodeGenerator(
    Value rowMemref, Value node, Type resultType,
    std::shared_ptr<IRepresentation> representation, Value tree,
    mlir::arith::CmpFPredicateAttr cmpPredicateAttr) {
  m_rowMemref = rowMemref;
  m_nodeToTraverse = node;
  m_resultType = resultType;
  m_state = kLoadThreshold;
  m_representation = representation;
  m_tree = tree;
  m_cmpPredicateAttr = cmpPredicateAttr;
}

bool ScalarTraverseTileCodeGenerator::EmitNext(
    ConversionPatternRewriter &rewriter, Location &location) {
  auto featureIndexType = m_representation->GetIndexElementType();
  auto thresholdType = m_representation->GetThresholdElementType();

  // Assert tile size is 1
  assert(m_representation->GetTileSize() == 1);

  switch (m_state) {
  case kLoadThreshold: {
    // TODO_Ashwin Remove the memref argument here
    m_nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(
        location, rewriter.getIndexType(),
        m_representation->GetThresholdsMemref(m_tree), m_nodeToTraverse);
    if (decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintTreeNodeOp>(location, m_nodeIndex);
    }
    Value treeIndex = m_representation->GetTreeIndex(m_tree);
    m_loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(
        location, thresholdType, m_representation->GetThresholdsMemref(m_tree),
        static_cast<Value>(m_nodeIndex), treeIndex);
    m_state = kLoadFeatureIndex;
  } break;
  case kLoadFeatureIndex: {
    Value treeIndex = m_representation->GetTreeIndex(m_tree);
    m_loadFeatureIndexOp =
        rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(
            location, featureIndexType,
            m_representation->GetFeatureIndexMemref(m_tree),
            static_cast<Value>(m_nodeIndex), treeIndex);

    m_extraLoads = m_representation->GenerateExtraLoads(location, rewriter,
                                                        m_tree, m_nodeIndex);
    m_state = kLoadFeature;
  } break;
  case kLoadFeature: {
    auto rowMemrefType = m_rowMemref.getType().cast<MemRefType>();
    auto rowIndex = rewriter.create<arith::IndexCastOp>(
        location, rewriter.getIndexType(),
        static_cast<Value>(m_loadFeatureIndexOp));
    auto zeroIndex = rewriter.create<arith::ConstantIndexOp>(location, 0);
    m_loadFeatureOp = rewriter.create<memref::LoadOp>(
        location, rowMemrefType.getElementType(), m_rowMemref,
        ValueRange(
            {static_cast<Value>(zeroIndex), static_cast<Value>(rowIndex)}));
    if (decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintComparisonOp>(
          location, m_loadFeatureOp, m_loadThresholdOp, m_loadFeatureIndexOp);
    }
    m_state = kCompare;
  } break;
  case kCompare: {
    // TODO we need a cast here to make sure the threshold and the row element
    // are the same type. The op expects both operands to be the same type.
    auto comparison = rewriter.create<arith::CmpFOp>(
        location, negateComparisonPredicate(m_cmpPredicateAttr),
        static_cast<Value>(m_loadFeatureOp),
        static_cast<Value>(m_loadThresholdOp));

    // auto threadIdx = GetThreadID(traverseTileOpPtr);
    // rewriter.create<gpu::PrintfOp>(location,
    //     "Thread %ld, %ld: Comparison %lf < %lf\n",
    //      ValueRange{threadIdx.x, threadIdx.y, m_loadFeatureOp.getResult(),
    //      m_loadThresholdOp.getResult()});

    m_comparisonUnsigned = rewriter.create<arith::ExtUIOp>(
        location, rewriter.getI32Type(), static_cast<Value>(comparison));
    m_state = kNextNode;
  } break;
  case kNextNode: {
    auto comparisonResultIndex = rewriter.create<arith::IndexCastOp>(
        location, rewriter.getIndexType(),
        static_cast<Value>(m_comparisonUnsigned));
    Value newIndex = m_representation->GenerateMoveToChild(
        location, rewriter, m_nodeIndex, comparisonResultIndex, 1,
        m_extraLoads);

    // TODO_Ashwin Remove the reference to the memref below
    // node = indexToNode(index)
    m_result = rewriter.create<decisionforest::IndexToNodeOp>(
        location, m_resultType, m_representation->GetThresholdsMemref(m_tree),
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
  assert(m_state == kDone);
  std::vector<Value> results;
  results.push_back(m_result);
  return results;
}

// ===---------------------------------------------------=== //
// VectorTraverseTileCodeGenerator Methods
// ===---------------------------------------------------=== //

VectorTraverseTileCodeGenerator::VectorTraverseTileCodeGenerator(
    Value tree, Value rowMemref, Value node, Type resultType,
    std::shared_ptr<IRepresentation> representation,
    std::function<Value(Value)> getLutFunc,
    mlir::arith::CmpFPredicateAttr cmpPredicateAttr) {
  m_tree = tree;
  m_rowMemref = rowMemref;
  m_nodeToTraverse = node;
  m_resultType = resultType;
  m_state = kLoadThreshold;
  m_representation = representation;
  m_getLutFunc = getLutFunc;
  m_cmpPredicateAttr = cmpPredicateAttr;

  auto featureIndexType = m_representation->GetIndexFieldType();
  m_featureIndexVectorType = featureIndexType.cast<VectorType>();
  assert(m_featureIndexVectorType);

  auto thresholdType = m_representation->GetThresholdFieldType();
  m_thresholdVectorType = thresholdType.cast<VectorType>();
  assert(m_thresholdVectorType);

  m_tileShapeType = m_representation->GetTileShapeType();
  m_tileSize = m_representation->GetTileSize();

  assert(m_tileSize > 1);
}

bool VectorTraverseTileCodeGenerator::EmitNext(
    ConversionPatternRewriter &rewriter, Location &location) {
  switch (m_state) {
  case kLoadThreshold: {
    // TODO_Ashwin remove reference to memref
    m_nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(
        location, rewriter.getIndexType(),
        m_representation->GetThresholdsMemref(m_tree), m_nodeToTraverse);
    if (decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintTreeNodeOp>(location, m_nodeIndex);
    }

    // Load threshold
    Value treeIndex = m_representation->GetTreeIndex(m_tree);
    m_loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(
        location, m_thresholdVectorType,
        m_representation->GetThresholdsMemref(m_tree),
        static_cast<Value>(m_nodeIndex), treeIndex);

    if (decisionforest::InsertDebugHelpers) {
      Value vectorVal = m_loadThresholdOp;
      if (!m_thresholdVectorType.getElementType().isF64()) {
        auto doubleVectorType =
            mlir::VectorType::get({m_tileSize}, rewriter.getF64Type());
        vectorVal = rewriter.create<arith::ExtFOp>(location, doubleVectorType,
                                                   static_cast<Value>(m_loadThresholdOp));
      }
      InsertPrintVectorOp(rewriter, location, 0 /*fp kind*/, 64, m_tileSize,
                          vectorVal);
    }
    m_state = kLoadFeatureIndex;
  } break;
  case kLoadFeatureIndex: {
    Value treeIndex = m_representation->GetTreeIndex(m_tree);
    m_loadFeatureIndexOp =
        rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(
            location, m_featureIndexVectorType,
            m_representation->GetFeatureIndexMemref(m_tree),
            static_cast<Value>(m_nodeIndex), treeIndex);

    if (decisionforest::InsertDebugHelpers) {
      InsertPrintVectorOp(
          rewriter, location, 1 /*int kind*/,
          m_featureIndexVectorType.getElementType().getIntOrFloatBitWidth(),
          m_tileSize, static_cast<Value>(m_loadFeatureIndexOp));
    }

    m_state = kLoadTileShape;
  } break;
  case kLoadTileShape: {
    Value treeIndex = m_representation->GetTreeIndex(m_tree);
    auto loadTileShapeOp = rewriter.create<decisionforest::LoadTileShapeOp>(
        location, m_tileShapeType, m_representation->GetTileShapeMemref(m_tree),
        static_cast<Value>(m_nodeIndex), treeIndex);
    m_loadTileShapeIndexOp = rewriter.create<arith::IndexCastOp>(
        location, rewriter.getIndexType(), static_cast<Value>(loadTileShapeOp));

    // gpu::Printf Op to print the node index, the thread block ID and the
    // thread ID auto blockIdx = GetBlockID(m_tree.getDefiningOp()); auto
    // threadIdx = GetThreadID(m_tree.getDefiningOp());
    // rewriter.create<gpu::PrintfOp>(location,
    //     "Block %ld, %ld Thread %ld, %ld: Node index %ld, Tile Shape %ld\n",
    //      ValueRange{blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
    //      m_nodeIndex, m_loadTileShapeIndexOp.getResult()});

    // Print the two elements of the feature index vector
    // auto featureIndexType = m_featureIndexVectorType.getElementType();
    // auto zeroIndexConstant =
    // rewriter.create<arith::ConstantIndexOp>(location, 0); auto
    // oneIndexConstant = rewriter.create<arith::ConstantIndexOp>(location, 1);
    // auto featureIndex0 = rewriter.create<vector::ExtractElementOp>(location,
    // featureIndexType, static_cast<Value>(m_loadFeatureIndexOp),
    // zeroIndexConstant.getResult()); auto featureIndex1 =
    // rewriter.create<vector::ExtractElementOp>(location, featureIndexType,
    // static_cast<Value>(m_loadFeatureIndexOp), oneIndexConstant.getResult());
    // rewriter.create<gpu::PrintfOp>(location,
    //     "Block %ld, %ld Thread %ld, %ld: Feature index %d, %d\n",
    //      ValueRange{blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
    //      featureIndex0, featureIndex1});

    m_state = kLoadChildIndex;
  } break;
  case kLoadChildIndex: {
    m_extraLoads = m_representation->GenerateExtraLoads(location, rewriter,
                                                        m_tree, m_nodeIndex);
    m_state = kLoadFeature;
  } break;
  case kLoadFeature: {
    auto rowMemrefType = m_rowMemref.getType().cast<MemRefType>();
    auto vectorIndexType =
        VectorType::get({m_tileSize}, rewriter.getIndexType());
    auto rowIndex = rewriter.create<arith::IndexCastOp>(
        location, vectorIndexType, static_cast<Value>(m_loadFeatureIndexOp));
    auto zeroIndex = rewriter.create<arith::ConstantIndexOp>(location, 0);
    // auto zeroIndexVector = rewriter.create<vector::BroadcastOp>(location,
    // vectorIndexType, zeroIndex);

    auto featuresVectorType =
        VectorType::get({m_tileSize}, rowMemrefType.getElementType());
    auto oneI1Const = rewriter.create<arith::ConstantIntOp>(
        location, 1, rewriter.getI1Type());
    auto i1VectorType = VectorType::get(m_tileSize, rewriter.getI1Type());
    auto mask = rewriter.create<vector::BroadcastOp>(location, i1VectorType,
                                                     oneI1Const);

    Value zeroPassThruConst;
    if (rowMemrefType.getElementType().isa<mlir::Float64Type>())
      zeroPassThruConst = rewriter.create<arith::ConstantFloatOp>(
          location, llvm::APFloat(0.0),
          rowMemrefType.getElementType().cast<FloatType>());
    else if (rowMemrefType.getElementType().isa<mlir::Float32Type>())
      zeroPassThruConst = rewriter.create<arith::ConstantFloatOp>(
          location, llvm::APFloat((float)0.0),
          rowMemrefType.getElementType().cast<FloatType>());
    else
      assert(false && "Unsupported floating point type");
    auto zeroPassThruVector = rewriter.create<vector::BroadcastOp>(
        location, featuresVectorType, zeroPassThruConst);

    m_features = rewriter.create<vector::GatherOp>(
        location, featuresVectorType, m_rowMemref,
        ValueRange(
            {static_cast<Value>(zeroIndex), static_cast<Value>(zeroIndex)}),
        rowIndex, mask, zeroPassThruVector);

    // auto threadIdx = GetThreadID(m_tree.getDefiningOp());
    // auto blockIdx = GetBlockID(m_tree.getDefiningOp());
    // auto featureElementType = featuresVectorType.getElementType();
    // auto zeroIndexConstant =
    // rewriter.create<arith::ConstantIndexOp>(location, 0); auto
    // oneIndexConstant = rewriter.create<arith::ConstantIndexOp>(location, 1);
    // auto feature0 = rewriter.create<vector::ExtractElementOp>(location,
    // featureElementType, static_cast<Value>(m_features),
    // zeroIndexConstant.getResult()); auto feature1 =
    // rewriter.create<vector::ExtractElementOp>(location, featureElementType,
    // static_cast<Value>(m_features), oneIndexConstant.getResult());
    // rewriter.create<gpu::PrintfOp>(location,
    //     "Block %ld, %ld Thread %ld, %ld: Features %lf, %lf\n",
    //      ValueRange{blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
    //      feature0, feature1});

    if (decisionforest::InsertDebugHelpers) {
      Value vectorVal = m_features;
      if (!featuresVectorType.getElementType().isF64()) {
        auto doubleVectorType =
            mlir::VectorType::get({m_tileSize}, rewriter.getF64Type());
        vectorVal = rewriter.create<arith::ExtFOp>(location, rewriter.getF64Type(),
                                                   static_cast<Value>(m_features));
      }

      InsertPrintVectorOp(rewriter, location, 0 /*fp kind*/,
                          64, // double
                          m_tileSize, vectorVal);
    }
    m_state = kCompare;
  } break;
  case kCompare: {
    auto comparison = rewriter.create<arith::CmpFOp>(
        location, m_cmpPredicateAttr.getValue(), static_cast<Value>(m_features),
        static_cast<Value>(m_loadThresholdOp));
    if (decisionforest::UseBitcastForComparisonOutcome)
      m_comparisonIndex = ReduceComparisonResultVectorToInt_Bitcast(
          comparison, m_tileSize, rewriter, location);
    else
      m_comparisonIndex = ReduceComparisonResultVectorToInt(
          comparison, m_tileSize, rewriter, location);

    // TODO This needs a different print routine!
    // if(decisionforest::InsertDebugHelpers) {
    //   rewriter.create<decisionforest::PrintComparisonOp>(location, feature,
    //   loadThresholdOp, loadFeatureIndexOp);
    // }

    m_state = kNextNode;
  } break;
  case kNextNode: {
    // Load the child index from the LUT
    auto lutValue = m_getLutFunc(m_tree);

    // Print the blockID, thread ID, m_loadTileShapeIndexOp, m_comparisonIndex
    // auto threadIdx = GetThreadID(m_tree.getDefiningOp());
    // auto blockIdx = GetBlockID(m_tree.getDefiningOp());
    // rewriter.create<gpu::PrintfOp>(location,
    //     "Block %ld, %ld Thread %ld, %ld: TileShapeIndex %ld, ComparisonIndex
    //     %ld\n",
    //      ValueRange{blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
    //      m_loadTileShapeIndexOp, m_comparisonIndex});

    auto childIndexInt = rewriter.create<memref::LoadOp>(
        location, lutValue,
        ValueRange{m_loadTileShapeIndexOp, m_comparisonIndex});
    auto childIndex = rewriter.create<arith::IndexCastOp>(
        location, rewriter.getIndexType(), static_cast<Value>(childIndexInt));

    Value newIndex = m_representation->GenerateMoveToChild(
        location, rewriter, m_nodeIndex, childIndex, m_tileSize, m_extraLoads);

    // node = indexToNode(index)
    // TODO_Ashwin Remove memref reference
    m_result = rewriter.create<decisionforest::IndexToNodeOp>(
        location, m_resultType, m_representation->GetThresholdsMemref(m_tree),
        static_cast<Value>(newIndex));
    // Print the blockID, thread ID and the next node index

    // auto threadIdx = GetThreadID(m_tree.getDefiningOp());
    // auto blockIdx = GetBlockID(m_tree.getDefiningOp());
    // rewriter.create<gpu::PrintfOp>(location,
    //   "Block %ld, %ld Thread %ld, %ld: Next node %ld\n",
    //     ValueRange{blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
    //     newIndex});

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
  assert(m_state == kDone);
  std::vector<Value> results;
  results.push_back(m_result);
  return results;
}

#ifdef TREEBEARD_GPU_SUPPORT

Value GenerateLocalThreadId(ConversionPatternRewriter &rewriter,
                            Location location, gpu::LaunchOp launchOp);

int64_t GetNumberOfThreadsInThreadBlock(gpu::LaunchOp gpuLaunchOp);

Value GPUTraverseLoweringState::GetSharedMemoryResultsMatrix(
    gpu::LaunchOp launchOp, ConversionPatternRewriter &rewriter,
    Location location, int32_t tileSize) {
  if (m_resultsSharedMemBuffer)
    return m_resultsSharedMemBuffer;
  auto numThreads = GetNumberOfThreadsInThreadBlock(launchOp);
  auto sharedMemoryType =
      MemRefType::get({numThreads, tileSize}, rewriter.getI8Type(), {}, 3);
  {
    decisionforest::helpers::SaveAndRestoreInsertionPoint saveIp(rewriter);
    // set insertion point of rewriter to start of module
    auto module = launchOp->getParentOfType<ModuleOp>();
    rewriter.setInsertionPoint(&module.front());
    // create a global memref of sharedMemoryType
    /*auto global =*/rewriter.create<memref::GlobalOp>(
        location, "tileTraversalSharedMemoryBuffer",
        /*sym_visibility=*/rewriter.getStringAttr("private"),
        /*type=*/sharedMemoryType,
        /*initial_value=*/rewriter.getUnitAttr(),
        /*constant=*/false, IntegerAttr());
  }
  {
    // Save and reset insertion point of rewriter to start of function
    decisionforest::helpers::SaveAndRestoreInsertionPoint saveIp(rewriter);
    rewriter.setInsertionPointToStart(&launchOp.getBody().front());

    // get the global memref
    auto getGlobalMemref = rewriter.create<memref::GetGlobalOp>(
        location, sharedMemoryType, "tileTraversalSharedMemoryBuffer");
    m_resultsSharedMemBuffer = getGlobalMemref;
  }
  return m_resultsSharedMemBuffer;
}

// ===---------------------------------------------------=== //
// GPUVectorTraverseTileCodeGenerator Methods
// ===---------------------------------------------------=== //
GPUVectorTraverseTileCodeGenerator::GPUVectorTraverseTileCodeGenerator(
    decisionforest::CooperativeTraverseTreeTileOp traverseOp, Type resultType,
    std::shared_ptr<IRepresentation> representation,
    std::function<Value(Value)> getLutFunc,
    mlir::arith::CmpFPredicateAttr cmpPredicateAttr,
    std::shared_ptr<GPUTraverseLoweringState> traverseLoweringState) {
  m_traverseTileOp = traverseOp;
  m_tree = traverseOp.getTree();
  // m_rowMemref = rowMemref;
  m_nodeToTraverse = traverseOp.getNode();
  m_resultType = resultType;
  m_state = kInit;
  m_representation = representation;
  m_getLutFunc = getLutFunc;
  m_cmpPredicateAttr = cmpPredicateAttr;
  m_traverseLoweringState = traverseLoweringState;

  m_featureIndexType = m_representation->GetIndexElementType();
  m_thresholdType = m_representation->GetThresholdElementType();

  m_tileShapeType = m_representation->GetTileShapeType();
  m_tileSize = m_representation->GetTileSize();
  m_currentTileElem = 0;

  m_getRootOp = FindGetRootOp(m_traverseTileOp);
  m_nodeIndexShMemBuffer =
      m_traverseLoweringState->getRootOpToShMemNodeArrayMap.find(m_getRootOp)
          ->second;
  assert(m_tileSize > 1);
}

bool GPUVectorTraverseTileCodeGenerator::EmitNext(
    ConversionPatternRewriter &rewriter, Location &location) {

  auto gpuLaunchOp = m_traverseTileOp->getParentOfType<gpu::LaunchOp>();
  assert(gpuLaunchOp);

  switch (m_state) {
  case kInit: {
    m_threadBlockThreadId =
        GenerateLocalThreadId(rewriter, location, gpuLaunchOp);
    m_tileSizeConst =
        rewriter.create<arith::ConstantIndexOp>(location, m_tileSize);
    m_threadId = rewriter.create<arith::RemSIOp>(
        location, m_threadBlockThreadId, m_tileSizeConst);
    m_threadBaseId = rewriter.create<arith::SubIOp>(
        location, m_threadBlockThreadId, m_threadId);
    m_cmpOutcomesShMemBuffer =
        m_traverseLoweringState->GetSharedMemoryResultsMatrix(
            gpuLaunchOp, rewriter, location, m_tileSize);

    m_nodeToTraverseIndex = rewriter.create<decisionforest::NodeToIndexOp>(
        location, rewriter.getIndexType(),
        m_representation->GetThresholdsMemref(m_tree), m_nodeToTraverse);

    m_state = kLoadThreshold;
  } break;
  case kLoadThreshold: {
    auto tileElementConst =
        rewriter.create<arith::ConstantIndexOp>(location, m_currentTileElem);
    // compute the sum of tileElementConst and threadBaseId
    auto tileElementPlusThreadBaseId = rewriter.create<arith::AddIOp>(
        location, tileElementConst, m_threadBaseId);
#ifdef TREEBEARD_GPU_USE_SHMEM_NODE_INDEX
    m_nodeIndex =
        rewriter
            .create<memref::LoadOp>(location, m_nodeIndexShMemBuffer,
                                    tileElementPlusThreadBaseId.getResult())
            .getResult();
#else  // TREEBEARD_GPU_USE_SHMEM_NODE_INDEX
    // Cast m_nodeToTraverseIndex to i32
    auto nodeToTraverseIndexI32 = rewriter.create<arith::IndexCastOp>(
        location, rewriter.getI32Type(), m_nodeToTraverseIndex);
    // Cast tileElementPlusThreadBaseId.getResult() to i32
    auto tileElementPlusThreadBaseIdI32 = rewriter.create<arith::IndexCastOp>(
        location, rewriter.getI32Type(), tileElementPlusThreadBaseId);
    // Cast m_tileSizeConst to i32
    // auto tileSizeConstI32 = rewriter.create<arith::IndexCastOp>(
    //     location, rewriter.getI32Type(), m_tileSizeConst);
    auto numThreads = std::min(
        32, static_cast<int32_t>(
                decisionforest::GetNumberOfThreadsInThreadBlock(gpuLaunchOp)));
    auto width = rewriter.create<arith::ConstantIntOp>(location, numThreads,
                                                       rewriter.getI32Type());
    auto nodeIndexShuffleOp = rewriter.create<gpu::ShuffleOp>(
        location, nodeToTraverseIndexI32, tileElementPlusThreadBaseIdI32,
        width.getResult(), gpu::ShuffleMode::IDX);
    assert(nodeIndexShuffleOp.getShuffleResult() ==
           nodeIndexShuffleOp.getResult(0));
    // m_nodeIndex = cast nodeIndexShuffleOp.getShuffleResult() to Index
    m_nodeIndex = rewriter.create<arith::IndexCastOp>(
        location, rewriter.getIndexType(),
        nodeIndexShuffleOp.getShuffleResult());
    // Create a gpu.printf to print the block id, thread id, the arguments of
    // the shuffle and the result

    // auto blockIdx = GetBlockID(m_traverseTileOp);
    // auto threadIdx = GetThreadID(m_traverseTileOp);
    // rewriter.create<gpu::PrintfOp>(
    //     location, "Block %ld, %d Thread %ld, %ld: Shuffle %d, %d, %d = %d\n",
    //     ValueRange{blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
    //                nodeToTraverseIndexI32, tileElementPlusThreadBaseIdI32,
    //                width.getResult(),
    //                nodeIndexShuffleOp.getShuffleResult()});
#endif // TREEBEARD_GPU_USE_SHMEM_NODE_INDEX
    // Load threshold
    Value treeIndex = m_representation->GetTreeIndex(m_tree);
    m_loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(
        location, m_thresholdType,
        m_representation->GetThresholdsMemref(m_tree),
        static_cast<Value>(m_nodeIndex), treeIndex, m_threadId);
    m_state = kLoadFeatureIndex;
  } break;
  case kLoadFeatureIndex: {
    Value treeIndex = m_representation->GetTreeIndex(m_tree);
    m_loadFeatureIndexOp =
        rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(
            location, m_featureIndexType,
            m_representation->GetFeatureIndexMemref(m_tree),
            static_cast<Value>(m_nodeIndex), treeIndex, m_threadId);
    m_state = kLoadFeature;
  } break;
  case kLoadFeature: {
    m_rowMemref = m_traverseTileOp.getData()[m_currentTileElem];
    // Create an index const with value 0
    auto zeroIndex = rewriter.create<arith::ConstantIndexOp>(location, 0);
    // Cast m_loadFeatureIndexOp to index
    auto featureIndex = rewriter.create<arith::IndexCastOp>(
        location, rewriter.getIndexType(), m_loadFeatureIndexOp);
    m_features = rewriter.create<memref::LoadOp>(
        location, m_rowMemref, ValueRange{zeroIndex, featureIndex.getResult()});
    m_state = kCompare;
  } break;
  case kCompare: {
    auto tileElementConst =
        rewriter.create<arith::ConstantIndexOp>(location, m_currentTileElem);
    auto comparison = rewriter.create<arith::CmpFOp>(
        location, m_cmpPredicateAttr.getValue(), static_cast<Value>(m_features),
        static_cast<Value>(m_loadThresholdOp));
    auto comparisonUnsigned = rewriter.create<arith::ExtUIOp>(
        location, rewriter.getI8Type(), static_cast<Value>(comparison));
    // add threadBaseId to tileElementConst
    auto tileElementPlusThreadBaseId = rewriter.create<arith::AddIOp>(
        location, tileElementConst, m_threadBaseId);

    rewriter.create<memref::StoreOp>(
        location, comparisonUnsigned, m_cmpOutcomesShMemBuffer,
        ValueRange{tileElementPlusThreadBaseId.getResult(), m_threadId});

    // Generate a gpu::PrintfOp to print ThreadID, comparison result, feature
    // index, feature value, threshold value auto blockIdx =
    // GetBlockID(m_traverseTileOp); auto threadIdx =
    // GetThreadID(m_traverseTileOp);

    // // Cast comparisonUnsigned to an i32
    // auto comparisonUnsignedIndex =
    // rewriter.create<arith::IndexCastOp>(location, rewriter.getIndexType(),
    // comparisonUnsigned.getResult());

    // // get the defining op of m_rowMemref and cast it subview
    // auto subviewOp =
    // dyn_cast<memref::SubViewOp>(m_rowMemref.getDefiningOp()); auto rowIdx =
    // subviewOp.getOffsets()[0];

    // rewriter.create<gpu::PrintfOp>(location,
    //      "Block %ld, %d Thread %ld, %ld: Comparison %lf < %lf = %ld RowIdx =
    //      %ld\n", ValueRange{blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
    //      m_features, m_loadThresholdOp, comparisonUnsignedIndex.getResult(),
    //      rowIdx});

    ++m_currentTileElem;
    if (m_currentTileElem == m_tileSize)
      m_state = kLoadChildIndex;
    else
      m_state = kLoadThreshold;
  } break;
  case kLoadChildIndex: {
    // Load the element at m_threadBlockThreadId in m_nodeIndexShMemBuffer into
    // m_nodeIndex
#ifdef TREEBEARD_GPU_USE_SHMEM_NODE_INDEX
    m_nodeIndex = rewriter
                      .create<memref::LoadOp>(location, m_nodeIndexShMemBuffer,
                                              m_threadBlockThreadId)
                      .getResult();
#else  // TREEBEARD_GPU_USE_SHMEM_NODE_INDEX
    m_nodeIndex = m_nodeToTraverseIndex;
#endif // TREEBEARD_GPU_USE_SHMEM_NODE_INDEX
    m_extraLoads = m_representation->GenerateExtraLoads(location, rewriter,
                                                        m_tree, m_nodeIndex);
    m_state = kLoadTileShape;
  } break;
  case kLoadTileShape: {
    Value treeIndex = m_representation->GetTreeIndex(m_tree);
    auto loadTileShapeOp = rewriter.create<decisionforest::LoadTileShapeOp>(
        location, m_tileShapeType, m_representation->GetTileShapeMemref(m_tree),
        m_nodeIndex, treeIndex);
    m_loadTileShapeIndexOp = rewriter.create<arith::IndexCastOp>(
        location, rewriter.getIndexType(), static_cast<Value>(loadTileShapeOp));

    m_state = kNextNode;
  } break;
  case kNextNode: {
    // Generate for loop to go over the m_threadBlockThreadId-th row of
    // m_cmpOutcomesShMemBuffer and or all the elements together first create a
    // zero index const

    // Can't use a barrier here because different threads processing different
    // trees may diverge. We can now only support tile size upto 32!
    // rewriter.create<gpu::BarrierOp>(location);
    auto zeroIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
    // create a one index const
    auto oneIndexConst = rewriter.create<arith::ConstantIndexOp>(location, 1);
    auto loop = rewriter.create<scf::ForOp>(
        location, zeroIndexConst.getResult(), m_tileSizeConst,
        oneIndexConst.getResult(), ValueRange{zeroIndexConst.getResult()});
    // set insertion point of rewriter to start of loop
    auto &entryBlock = *loop.getBody();
    rewriter.setInsertionPointToStart(&entryBlock);

    // read the index-th element of the m_threadBlockThreadId row from
    // m_cmpOutcomesShMemBuffer
    auto cmpOutcome = rewriter.create<memref::LoadOp>(
        location, m_cmpOutcomesShMemBuffer,
        ValueRange{m_threadBlockThreadId, loop.getInductionVar()});

    // cast the cmpOutcome to an index
    auto cmpOutcomeIndex = rewriter.create<arith::IndexCastOp>(
        location, rewriter.getIndexType(), cmpOutcome);

    // Print the threadID, loop index and compOutcomeIndex
    // auto threadIdx = GetThreadID(m_traverseTileOp);
    // auto blockIdx = GetBlockID(m_traverseTileOp);
    // rewriter.create<gpu::PrintfOp>(location,
    //      "Block %ld, %ld Thread %ld, %ld: Loop index %ld, cmpOutcomeIndex
    //      %ld\n", ValueRange{blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
    //      loop.getInductionVar(), cmpOutcomeIndex});

    // auto tileSizeMinusI = rewriter.create<arith::SubIOp>(location,
    // m_tileSizeConst, loop.getInductionVar()); subtract one from
    // tileSizeMinusThreadId auto tileSizeMinusIMinusOne =
    // rewriter.create<arith::SubIOp>(location, tileSizeMinusI.getResult(),
    // oneIndexConst.getResult());

    // Left shift by m_threadId
    // auto shiftedComparison = rewriter.create<arith::ShLIOp>(location,
    // cmpOutcomeIndex.getResult(), tileSizeMinusIMinusOne.getResult());
    auto shiftedComparison = rewriter.create<arith::ShLIOp>(
        location, cmpOutcomeIndex.getResult(), loop.getInductionVar());

    // or it with the current value of the loop argument
    auto orOp = rewriter.create<arith::OrIOp>(
        location, shiftedComparison.getResult(), entryBlock.getArgument(1));

    // yield the result of the or
    rewriter.create<scf::YieldOp>(location, orOp.getResult());

    // set rewriter insertion point to after the loop
    rewriter.setInsertionPointAfter(loop);

    // comparison index is the result of the loop
    auto comparisonIndex = loop.getResult(0);

    // Load the child index from the LUT
    auto lutValue = m_getLutFunc(m_tree);
    auto childIndexInt = rewriter.create<memref::LoadOp>(
        location, lutValue,
        ValueRange{m_loadTileShapeIndexOp, comparisonIndex});
    auto childIndex = rewriter.create<arith::IndexCastOp>(
        location, rewriter.getIndexType(), static_cast<Value>(childIndexInt));

    // Load lutValue[m_loadTileShapeIndexOp][0...3] and print the values
    // for (int i = 0; i < 4; ++i)
    // {
    //   // create an index constant with value i
    //   auto iIndexConst = rewriter.create<arith::ConstantIndexOp>(location,
    //   i); auto lutValueInt = rewriter.create<memref::LoadOp>(location,
    //   lutValue, ValueRange{m_loadTileShapeIndexOp, iIndexConst.getResult()});
    //   auto lutValueIndex = rewriter.create<arith::IndexCastOp>(location,
    //   rewriter.getIndexType(), static_cast<Value>(lutValueInt));
    //   rewriter.create<gpu::PrintfOp>(location,
    //      "\tBlock %ld, %ld Thread %ld, %ld: lutValue[%ld][%ld] = %ld\n",
    //      ValueRange{blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
    //      m_loadTileShapeIndexOp, iIndexConst.getResult(), lutValueIndex});
    // }

    // // Generate a gpu::PrintfOp to print m_loadTileShapeIndexOp,
    // comparisonIndex and childIndex
    // // auto threadIdx = GetThreadID(m_traverseTileOp);
    // rewriter.create<gpu::PrintfOp>(location,
    //      "Block %ld, %ld Thread %ld, %ld: m_loadTileShapeIndexOp %ld,
    //      comparisonIndex %ld, childIndex %ld\n", ValueRange{blockIdx.x,
    //      blockIdx.y, threadIdx.x, threadIdx.y, m_loadTileShapeIndexOp,
    //      comparisonIndex, childIndex});

#ifdef TREEBEARD_GPU_USE_SHMEM_NODE_INDEX
    // load the node index from m_nodeIndexShMemBuffer
    auto nodeIndexShMem = rewriter.create<memref::LoadOp>(
        location, m_nodeIndexShMemBuffer, m_threadBlockThreadId);
    Value newIndex = m_representation->GenerateMoveToChild(
        location, rewriter, nodeIndexShMem, childIndex, m_tileSize,
        m_extraLoads);

    // Store newIndex into m_nodeIndexShMemBuffer at m_threadBlockThreadId
    rewriter.create<memref::StoreOp>(location, newIndex, m_nodeIndexShMemBuffer,
                                     m_threadBlockThreadId);
#else  // TREEBEARD_GPU_USE_SHMEM_NODE_INDEX
    Value newIndex = m_representation->GenerateMoveToChild(
        location, rewriter, m_nodeToTraverseIndex, childIndex, m_tileSize,
        m_extraLoads);
#endif // TREEBEARD_GPU_USE_SHMEM_NODE_INDEX

    // node = indexToNode(index)
    // TODO_Ashwin Remove memref reference
    m_result = rewriter.create<decisionforest::IndexToNodeOp>(
        location, m_resultType, m_representation->GetThresholdsMemref(m_tree),
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

std::vector<Value> GPUVectorTraverseTileCodeGenerator::GetResult() {
  assert(m_state == kDone);
  std::vector<Value> results;
  results.push_back(m_result);
  return results;
}
#endif // TREEBEARD_GPU_SUPPORT
} // namespace decisionforest
} // namespace mlir