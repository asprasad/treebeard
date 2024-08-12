#include "Dialect.h"
#include <iostream>
#include <memory>
#include <vector>
// #include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

#include "CodeGenStateMachine.h"
#include "Dialect.h"
#include "LIRLoweringHelpers.h"
#include "MemrefTypes.h"
#include "ModelSerializers.h"
#include "OpLoweringUtils.h"
#include "Representations.h"
#include "TiledTree.h"
#include "TraverseTreeTileOpLowering.h"
#include "TreeTilingUtils.h"
#include "schedule.h"

using namespace mlir::decisionforest::helpers;

/*
Plan and issues
* We can add a memref argument to the inference function, but modifying
  the function type while lowering doesn't seem easy. We don't know the
  type of the model memref until all optimizations have run -- so we
  can't just add an argument while we create the function in HIR.
* We could just clone our function into a function with a different
  signature though
* [Problem] If we add a single memref to represent the model, we're
  we're assuming that all trees are tiled identically! Is this an
  assumption we should bake into the code gen?
* [Selected] [Option] What if we add a global memref to the module and a
function that just returns this memref? We can the populate it in C code and the
inference function can use the global. We could just have multiple global
memrefs if we want different tiling for different trees.
  - [Problem] How would the IR pick between these memrefs though? We
    would need something like an array of memrefs so we can pick the
    right one based on the tree index. Also, each of these memrefs
    would have a different type.
    [A] This may not be a problem. We need to statically know the
    type of the tree (which includes tree tiling) to be able to generate
    code. So we should know which memref to access if there is one
    memref per unique tree type.
*/

/*
  trees = memref<Tiles, ?>
  offsets = memref<int32>

  all rows in batch
    all trees in forest
      tree = trees + offset[treeIndex] // memref.subview
      n = 0
      while (!IsLeaf(n))
        thresholds = LoadTileThresholds(tree, n)
        indices  = LoadTileFeatureIndices(tree, n)
        features = gather(data[i], indices)
        outcome = features < thresholds // type bool if tileSize = 1
        // Assuming TileSize == 1
        n = 2*n + 1 + outcome

*/
using namespace mlir;

namespace mlir {
namespace decisionforest {
// Definitions in GPURepresentations.cpp

// Add a simple initialization function that initializes a GPU buffer by
// transferring values in a CPU buffer to the GPU buffer (single data transfer)
void GenerateSimpleInitializer(const std::string &funcName,
                               ConversionPatternRewriter &rewriter,
                               Location location, ModuleOp module,
                               MemRefType memrefType);

int64_t GetNumberOfThreadsInThreadBlock(gpu::LaunchOp gpuLaunchOp);
Value GenerateLocalThreadId(ConversionPatternRewriter &rewriter,
                            Location location, gpu::LaunchOp launchOp);

mlir::gpu::KernelDim3 GetThreadID(mlir::Operation *op);
mlir::gpu::KernelDim3 GetBlockID(mlir::Operation *op);

} // namespace decisionforest
} // namespace mlir

Value getLUT;

namespace mlir {
namespace decisionforest {

struct EnsembleConstantOpLowering : public ConversionPattern {
  std::shared_ptr<decisionforest::IModelSerializer> m_serializer;
  std::shared_ptr<decisionforest::IRepresentation> m_representation;

  EnsembleConstantOpLowering(
      MLIRContext *ctx,
      std::shared_ptr<decisionforest::IModelSerializer> serializer,
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : ConversionPattern(
            mlir::decisionforest::EnsembleConstantOp::getOperationName(),
            1 /*benefit*/, ctx),
        m_serializer(serializer), m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    mlir::decisionforest::EnsembleConstantOp ensembleConstOp =
        llvm::dyn_cast<mlir::decisionforest::EnsembleConstantOp>(op);
    assert(ensembleConstOp);
    assert(operands.size() == 0);
    if (!ensembleConstOp)
      return mlir::failure();

    auto location = op->getLoc();
    auto owningModule = op->getParentOfType<mlir::ModuleOp>();
    assert(owningModule);

    auto ret = m_representation->GenerateModelGlobals(op, operands, rewriter,
                                                      m_serializer);
    if (ret.failed()) {
      return ret;
    }

    auto forestType = ensembleConstOp.getResult()
                          .getType()
                          .cast<decisionforest::TreeEnsembleType>();
    auto firstTreeType =
        forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
    auto firstTreeTileSize = firstTreeType.getTileSize();

    // Type lookUpTableMemrefType;
    // Value getLUT;
    if (firstTreeTileSize > 1) {
      std::string lookupTableMemrefName = "lookupTable";
      auto lookUpTableMemrefType =
          AddChildIndexLookUpTable(owningModule, ensembleConstOp, rewriter,
                                   location, lookupTableMemrefName);
      getLUT = rewriter.create<memref::GetGlobalOp>(
          location, lookUpTableMemrefType, lookupTableMemrefName);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }

  Type AddChildIndexLookUpTable(
      mlir::ModuleOp module,
      mlir::decisionforest::EnsembleConstantOp &ensembleConstOp,
      ConversionPatternRewriter &rewriter, Location location,
      std::string &lookupTableMemrefName) const {
    SaveAndRestoreInsertionPoint saveAndRestoreInsertPoint(rewriter);
    rewriter.setInsertionPoint(&module.front());

    auto forestType = ensembleConstOp.getResult()
                          .getType()
                          .cast<decisionforest::TreeEnsembleType>();
    // We will assume that all trees have the same tile size
    auto numTrees = static_cast<int32_t>(forestType.getNumberOfTrees());
    assert(numTrees > 0);
    auto firstTreeType =
        forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
    auto firstTreeTileSize = firstTreeType.getTileSize();
    for (int32_t i = 1; i < numTrees; ++i) {
      auto treeType =
          forestType.getTreeType(i).cast<mlir::decisionforest::TreeType>();
      auto tileSize = treeType.getTileSize();
      assert(firstTreeTileSize == tileSize &&
             "All tree's should have the same tile size");
    }
    auto tileSize = firstTreeTileSize;
    if (tileSize == 1)
      return Type(); // We don't need a lookup table if the tile size is 1

    auto numberOfTileOutcomes = static_cast<int>(std::pow(2, tileSize));
    auto numberOfTileShapes =
        mlir::decisionforest::TileShapeToTileIDMap::NumberOfTileShapes(
            tileSize);
    // TODO We may need to implement something smarter here. We don't really
    // need I8's for each outcome. We could store all outcomes in a single int64
    // for tile size 4 for example (each entry needs 3 bits and there are 16
    // entries -- one for each outcome).
    auto lutMemrefType = MemRefType::get(
        {numberOfTileShapes, numberOfTileOutcomes}, rewriter.getI8Type());
    std::vector<int8_t> lutData(numberOfTileShapes * numberOfTileOutcomes);
    mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLookUpTable(
        lutData.data(), tileSize, /*entryBitWidth*/ 8);
    mlir::decisionforest::createConstantGlobalOp(
        rewriter, location, lookupTableMemrefName, lutMemrefType, lutData);

    return lutMemrefType;
  }
};

struct EnsembleConstantOpGPULowering : public ConversionPattern {
  std::shared_ptr<decisionforest::IModelSerializer> m_serializer;
  std::shared_ptr<decisionforest::IRepresentation> m_representation;

  EnsembleConstantOpGPULowering(
      MLIRContext *ctx,
      std::shared_ptr<decisionforest::IModelSerializer> serializer,
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : ConversionPattern(
            mlir::decisionforest::EnsembleConstantOp::getOperationName(),
            1 /*benefit*/, ctx),
        m_serializer(serializer), m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    mlir::decisionforest::EnsembleConstantOp ensembleConstOp =
        llvm::dyn_cast<mlir::decisionforest::EnsembleConstantOp>(op);
    assert(ensembleConstOp);
    assert(operands.size() == 0);
    if (!ensembleConstOp)
      return mlir::failure();

    auto location = op->getLoc();
    auto owningModule = op->getParentOfType<mlir::ModuleOp>();
    assert(owningModule);

    auto ret = m_representation->GenerateModelGlobals(op, operands, rewriter,
                                                      m_serializer);
    if (ret.failed()) {
      return ret;
    }

    auto forestType = ensembleConstOp.getResult()
                          .getType()
                          .cast<decisionforest::TreeEnsembleType>();
    auto firstTreeType =
        forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
    auto firstTreeTileSize = firstTreeType.getTileSize();

    // Type lookUpTableMemrefType;
    // Value getLUT;
    if (firstTreeTileSize > 1) {
      AddChildIndexLookUpTable(owningModule, ensembleConstOp, rewriter,
                               location);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }

  Type AddChildIndexLookUpTable(
      mlir::ModuleOp module,
      mlir::decisionforest::EnsembleConstantOp &ensembleConstOp,
      ConversionPatternRewriter &rewriter, Location location) const {

    auto func = ensembleConstOp->getParentOfType<func::FuncOp>();
    assert(func);

    SaveAndRestoreInsertionPoint saveAndRestoreInsertPoint(rewriter);
    rewriter.setInsertionPoint(&module.front());

    auto forestType = ensembleConstOp.getResult()
                          .getType()
                          .cast<decisionforest::TreeEnsembleType>();
    // We will assume that all trees have the same tile size
    auto numTrees = static_cast<int32_t>(forestType.getNumberOfTrees());
    assert(numTrees > 0);
    auto firstTreeType =
        forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
    auto firstTreeTileSize = firstTreeType.getTileSize();
    for (int32_t i = 1; i < numTrees; ++i) {
      auto treeType =
          forestType.getTreeType(i).cast<mlir::decisionforest::TreeType>();
      auto tileSize = treeType.getTileSize();
      assert(firstTreeTileSize == tileSize &&
             "All tree's should have the same tile size");
    }
    auto tileSize = firstTreeTileSize;
    if (tileSize == 1)
      return Type(); // We don't need a lookup table if the tile size is 1

    auto numberOfTileOutcomes = static_cast<int>(std::pow(2, tileSize));
    auto numberOfTileShapes =
        mlir::decisionforest::TileShapeToTileIDMap::NumberOfTileShapes(
            tileSize);
    // TODO We may need to implement something smarter here. We don't really
    // need I8's for each outcome. We could store all outcomes in a single int64
    // for tile size 4 for example (each entry needs 3 bits and there are 16
    // entries -- one for each outcome).
    auto lutMemrefType = MemRefType::get(
        {numberOfTileShapes, numberOfTileOutcomes}, rewriter.getI8Type());

    func.insertArgument(func.getNumArguments(), lutMemrefType,
                        mlir::DictionaryAttr(), location);

    decisionforest::GenerateSimpleInitializer("Init_LUT", rewriter, location,
                                              module, lutMemrefType);
    getLUT = func.getArgument(func.getNumArguments() - 1);
    return lutMemrefType;
  }
};

struct GetTreeOpLowering : public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;
  GetTreeOpLowering(
      MLIRContext *ctx,
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : ConversionPattern(
            mlir::decisionforest::GetTreeFromEnsembleOp::getOperationName(),
            1 /*benefit*/, ctx),
        m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    mlir::decisionforest::GetTreeFromEnsembleOp getTreeOp =
        llvm::dyn_cast<mlir::decisionforest::GetTreeFromEnsembleOp>(op);
    assert(getTreeOp);
    assert(operands.size() == 2);
    if (!getTreeOp)
      return mlir::failure();
    m_representation->GenerateTreeMemref(rewriter, op, operands[0],
                                         operands[1]);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

Value GetLUTFromTreeOperand(Value treeValue) { return getLUT; }

struct GetTreeClassIdOpLowering : public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;

  GetTreeClassIdOpLowering(
      MLIRContext *ctx,
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : ConversionPattern(
            mlir::decisionforest::GetTreeClassIdOp::getOperationName(),
            1 /*benefit*/, ctx),
        m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 2);
    auto classId = m_representation->GenerateGetTreeClassId(
        rewriter, op, operands[0], operands[1]);
    rewriter.replaceOp(op, static_cast<Value>(classId));
    return mlir::success();
  }
};

struct GetRootOpLowering : public ConversionPattern {
  std::shared_ptr<mlir::decisionforest::IRepresentation> m_representation;
  GetRootOpLowering(
      MLIRContext *ctx,
      std::shared_ptr<mlir::decisionforest::IRepresentation> representation)
      : ConversionPattern(mlir::decisionforest::GetRootOp::getOperationName(),
                          1 /*benefit*/, ctx),
        m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto getRootOp = AssertOpIsOfType<mlir::decisionforest::GetRootOp>(op);
    auto nodeIndexConst =
        rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);

    // [TODO_Ashwin] We're just getting some memref here to pass as an argument
    // to the IndexToNodeOp. Maybe we should just get rid of that argument?
    auto treeMemref = m_representation->GetThresholdsMemref(operands[0]);
    auto nodeType = getRootOp.getResult().getType();
    auto node = rewriter.create<decisionforest::IndexToNodeOp>(
        op->getLoc(), nodeType, treeMemref, static_cast<Value>(nodeIndexConst));
    rewriter.replaceOp(op, static_cast<Value>(node));
    return mlir::success();
  }
};

#ifdef TREEBEARD_GPU_SUPPORT
struct GPUGetRootOpLowering : public ConversionPattern {
  std::shared_ptr<mlir::decisionforest::IRepresentation> m_representation;
  std::shared_ptr<GPUTraverseLoweringState> m_traverseLoweringState;
  GPUGetRootOpLowering(
      MLIRContext *ctx,
      std::shared_ptr<mlir::decisionforest::IRepresentation> representation,
      std::shared_ptr<GPUTraverseLoweringState> traverseLoweringState)
      : ConversionPattern(mlir::decisionforest::GetRootOp::getOperationName(),
                          1 /*benefit*/, ctx),
        m_representation(representation),
        m_traverseLoweringState(traverseLoweringState) {}

  void GenerateTreeIndexBuffers(ConversionPatternRewriter &rewriter,
                                mlir::Operation *op,
                                mlir::Value treeValue) const {
    auto getRootOp = AssertOpIsOfType<decisionforest::GetRootOp>(op);
    auto treeType =
        getRootOp.getTree().getType().cast<decisionforest::TreeType>();
    auto tileSize = treeType.getTileSize();
    if (tileSize == 1) {
      return;
    }

    // Add shared memory buffer that is the size of numThreads
    // Initialize every element to zero
    // __syncthreads()
    auto owningModule = op->getParentOfType<mlir::ModuleOp>();
    auto gpuLaunchOp = op->getParentOfType<gpu::LaunchOp>();
    auto location = op->getLoc();

    auto numThreads =
        decisionforest::GetNumberOfThreadsInThreadBlock(gpuLaunchOp);

    std::string globalBufferName =
        "__nodeIndexBuffer_" + std::to_string(reinterpret_cast<int64_t>(op));
    auto globalBufferType =
        MemRefType::get({numThreads}, rewriter.getIndexType(), {}, 3);
    // create a global for shared memory
    {
      SaveAndRestoreInsertionPoint saveAndRestoreInsertPoint(rewriter);
      rewriter.setInsertionPoint(&owningModule.front());
      rewriter.create<memref::GlobalOp>(
          location, globalBufferName,
          /*sym_visibility=*/rewriter.getStringAttr("private"),
          /*type=*/globalBufferType,
          /*initial_value=*/rewriter.getUnitAttr(),
          /*constant=*/false,
          /*alignment*/ IntegerAttr());
    }
    auto globalRef = rewriter.create<memref::GetGlobalOp>(
        location, globalBufferType, globalBufferName);
    auto threadIndex =
        decisionforest::GenerateLocalThreadId(rewriter, location, gpuLaunchOp);
    auto zeroConst = rewriter.create<arith::ConstantIndexOp>(location, 0);
    rewriter.create<memref::StoreOp>(location, zeroConst.getResult(),
                                     globalRef.getResult(),
                                     ValueRange{threadIndex});
    // Don't need this barrier since we are only supporting tile sizes < 32
    // Anyway, we can guarantee that threads in a warp will not diverge
    // rewriter.create<gpu::BarrierOp>(location);

    m_traverseLoweringState->getRootOpToShMemNodeArrayMap[op] = globalRef;
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto getRootOp = AssertOpIsOfType<mlir::decisionforest::GetRootOp>(op);
    Value treeValue = operands[0];

#ifdef TREEBEARD_GPU_USE_SHMEM_NODE_INDEX
    GenerateTreeIndexBuffers(rewriter, op, treeValue);
#endif // #ifdef TREEBEARD_GPU_USE_SHMEM_NODE_INDEX

    auto nodeIndexConst =
        rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);

    // [TODO_Ashwin] We're just getting some memref here to pass as an argument
    // to the IndexToNodeOp. Maybe we should just get rid of that argument?
    auto treeMemref = m_representation->GetThresholdsMemref(treeValue);
    auto nodeType = getRootOp.getResult().getType();
    auto node = rewriter.create<decisionforest::IndexToNodeOp>(
        op->getLoc(), nodeType, treeMemref, static_cast<Value>(nodeIndexConst));
    rewriter.replaceOp(op, static_cast<Value>(node));
    return mlir::success();
  }
};
#endif // TREEBEARD_GPU_SUPPORT

struct IsLeafOpLowering : public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;
  IsLeafOpLowering(
      MLIRContext *ctx,
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : ConversionPattern(mlir::decisionforest::IsLeafOp::getOperationName(),
                          1 /*benefit*/, ctx),
        m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    mlir::decisionforest::IsLeafOp isLeafOp =
        AssertOpIsOfType<mlir::decisionforest::IsLeafOp>(op);
    assert(operands.size() == 2);
    if (!isLeafOp)
      return mlir::failure();

    auto location = op->getLoc();

    auto treeMemref = m_representation->GetThresholdsMemref(operands[0]);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert(treeMemrefType);
    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(
        location, rewriter.getIndexType(), treeMemref,
        operands[1]); // Convert the node to an index

    auto isLeafValue = m_representation->GenerateIsLeafOp(
        rewriter, op, operands[0], nodeIndex);
    rewriter.replaceOp(op, static_cast<Value>(isLeafValue));

    return mlir::success();
  }
};

struct InterleavedTraverseTreeTileOpLowering : public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;

  InterleavedTraverseTreeTileOpLowering(
      MLIRContext *ctx,
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : ConversionPattern(mlir::decisionforest::InterleavedTraverseTreeTileOp::
                              getOperationName(),
                          1 /*benefit*/, ctx),
        m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    decisionforest::InterleavedTraverseTreeTileOpLoweringHelper
        traverseLoweringHelper(GetLUTFromTreeOperand, m_representation);
    return traverseLoweringHelper.matchAndRewrite(
        AssertOpIsOfType<mlir::decisionforest::InterleavedTraverseTreeTileOp>(
            op),
        operands, rewriter);
  }
};

struct TraverseTreeTileOpLowering : public ConversionPattern {
  std::shared_ptr<mlir::decisionforest::IRepresentation> m_representation;
  TraverseTreeTileOpLowering(
      MLIRContext *ctx,
      std::shared_ptr<mlir::decisionforest::IRepresentation> representation)
      : ConversionPattern(
            mlir::decisionforest::TraverseTreeTileOp::getOperationName(),
            1 /*benefit*/, ctx),
        m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto traverseTileOp =
        AssertOpIsOfType<mlir::decisionforest::TraverseTreeTileOp>(op);
    auto traverseTileAdaptor =
        decisionforest::TraverseTreeTileOpAdaptor(traverseTileOp);
    assert(operands.size() == 3);
    if (!traverseTileOp)
      return mlir::failure();

    decisionforest::InterleavedCodeGenStateMachine codeGenStateMachine;
    if (m_representation->GetTileSize() == 1)
      codeGenStateMachine.AddStateMachine(
          std::make_unique<decisionforest::ScalarTraverseTileCodeGenerator>(
              traverseTileAdaptor.getData(), traverseTileAdaptor.getNode(),
              traverseTileOp.getResult().getType(), m_representation,
              traverseTileAdaptor.getTree(),
              traverseTileOp.getPredicateAttr()));
    else
      codeGenStateMachine.AddStateMachine(
          std::make_unique<decisionforest::VectorTraverseTileCodeGenerator>(
              traverseTileAdaptor.getTree(), traverseTileAdaptor.getData(),
              traverseTileAdaptor.getNode(),
              traverseTileOp.getResult().getType(), m_representation,
              GetLUTFromTreeOperand, traverseTileOp.getPredicateAttr()));

    // Emit code.
    auto location = op->getLoc();
    while (codeGenStateMachine.EmitNext(rewriter, location))
      ;

    rewriter.replaceOp(op,
                       static_cast<Value>(codeGenStateMachine.GetResult()[0]));
    return mlir::success();
  }
};

#ifdef TREEBEARD_GPU_SUPPORT
struct CooperativeTraverseTreeTileOpLowering : public ConversionPattern {
  std::shared_ptr<mlir::decisionforest::IRepresentation> m_representation;
  std::shared_ptr<GPUTraverseLoweringState> m_traverseTileLoweringState;
  CooperativeTraverseTreeTileOpLowering(
      MLIRContext *ctx,
      std::shared_ptr<mlir::decisionforest::IRepresentation> representation,
      std::shared_ptr<GPUTraverseLoweringState> traverseTileLoweringState)
      : ConversionPattern(mlir::decisionforest::CooperativeTraverseTreeTileOp::
                              getOperationName(),
                          1 /*benefit*/, ctx),
        m_representation(representation),
        m_traverseTileLoweringState(traverseTileLoweringState) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto traverseTileOp =
        AssertOpIsOfType<mlir::decisionforest::CooperativeTraverseTreeTileOp>(
            op);
    if (!traverseTileOp)
      return mlir::failure();

    decisionforest::InterleavedCodeGenStateMachine codeGenStateMachine;
    codeGenStateMachine.AddStateMachine(
        std::make_unique<decisionforest::GPUVectorTraverseTileCodeGenerator>(
            traverseTileOp, traverseTileOp.getResult().getType(),
            m_representation, GetLUTFromTreeOperand,
            traverseTileOp.getPredicateAttr(), m_traverseTileLoweringState));

    // Emit code.
    auto location = op->getLoc();
    while (codeGenStateMachine.EmitNext(rewriter, location))
      ;

    rewriter.replaceOp(op,
                       static_cast<Value>(codeGenStateMachine.GetResult()[0]));
    return mlir::success();
  }
};
#endif // TREEBEARD_GPU_SUPPORT

struct GetLeafValueOpLowering : public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;

  GetLeafValueOpLowering(
      MLIRContext *ctx,
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : ConversionPattern(
            mlir::decisionforest::GetLeafValueOp::getOperationName(),
            1 /*benefit*/, ctx),
        m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto getLeafVal =
        AssertOpIsOfType<mlir::decisionforest::GetLeafValueOp>(op);
    assert(operands.size() == 2);
    if (!getLeafVal)
      return mlir::failure();
    auto location = op->getLoc();

    auto treeMemref = m_representation->GetThresholdsMemref(operands[0]);
    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(
        location, rewriter.getIndexType(), treeMemref, operands[1]);
    if (decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintTreeNodeOp>(location, nodeIndex);
    }

    // auto blockIdx = GetBlockID(getLeafVal);
    // auto threadIdx = GetThreadID(getLeafVal);
    // rewriter.create<gpu::PrintfOp>(
    //     location, "Block %ld, %d Thread %ld, %ld: getLeafValue\n",
    //     ValueRange{blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y});

    auto leafValue = m_representation->GenerateGetLeafValueOp(
        rewriter, op, operands[0], nodeIndex);
    // TODO cast the loaded value to the correct result type of the tree.
    rewriter.replaceOp(op, static_cast<Value>(leafValue));
    return mlir::success();
  }
};

struct GetLeafTileValueOpLowering : public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;

  GetLeafTileValueOpLowering(
      MLIRContext *ctx,
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : ConversionPattern(
            mlir::decisionforest::GetLeafTileValueOp::getOperationName(),
            1 /*benefit*/, ctx),
        m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto getLeafVal =
        AssertOpIsOfType<mlir::decisionforest::GetLeafTileValueOp>(op);
    assert(operands.size() == 2);
    if (!getLeafVal)
      return mlir::failure();
    decisionforest::GetLeafTileValueOpAdaptor getLeafValAdaptor(getLeafVal);
    auto location = op->getLoc();

    auto thresholdType = m_representation->GetThresholdFieldType();
    auto node = getLeafValAdaptor.getNode();
    auto tree = getLeafValAdaptor.getTree();
    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(
        location, rewriter.getIndexType(),
        m_representation->GetThresholdsMemref(tree), node);
    if (decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintTreeNodeOp>(location, nodeIndex);
    }

    // Load threshold
    // TODO Ideally, this should be a different op for when we deal with tile
    // sizes != 1. We will then need to load a single threshold value and cast
    // it the trees return type
    Value treeIndex = m_representation->GetTreeIndex(tree);
    auto loadThresholdOp =
        rewriter.create<decisionforest::LoadTileThresholdsOp>(
            location, thresholdType,
            m_representation->GetThresholdsMemref(tree),
            static_cast<Value>(nodeIndex), treeIndex);
    Value leafValue = loadThresholdOp;
    auto tileSize = m_representation->GetTileSize();
    if (tileSize != 1) {
      auto thresholdVectorType = thresholdType.cast<VectorType>();
      if (decisionforest::InsertDebugHelpers) {
        Value vectorVal = loadThresholdOp;
        if (!thresholdVectorType.getElementType().isF64()) {
          auto doubleVectorType =
              mlir::VectorType::get({tileSize}, rewriter.getF64Type());
          vectorVal = rewriter.create<arith::ExtFOp>(location, doubleVectorType,
                                                     loadThresholdOp);
        }
        InsertPrintVectorOp(rewriter, location, 0, 64, tileSize, vectorVal);
      }
      auto zeroConst = rewriter.create<arith::ConstantIntOp>(
          location, int64_t(0), rewriter.getI32Type());
      auto extractElement = rewriter.create<vector::ExtractElementOp>(
          location, static_cast<Value>(loadThresholdOp), zeroConst);
      leafValue = extractElement;
    }

    // TODO cast the loaded value to the correct result type of the tree.
    rewriter.replaceOp(op, static_cast<Value>(leafValue));
    return mlir::success();
  }
};

struct IsLeafTileOpLowering : public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;
  IsLeafTileOpLowering(
      MLIRContext *ctx,
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : ConversionPattern(
            mlir::decisionforest::IsLeafTileOp::getOperationName(),
            1 /*benefit*/, ctx),
        m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    mlir::decisionforest::IsLeafTileOp isLeafOp =
        AssertOpIsOfType<mlir::decisionforest::IsLeafTileOp>(op);
    assert(operands.size() == 2);
    if (!isLeafOp)
      return mlir::failure();

    auto location = op->getLoc();

    auto treeMemref = m_representation->GetThresholdsMemref(operands[0]);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert(treeMemrefType);

    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(
        location, rewriter.getIndexType(), treeMemref,
        operands[1]); // Convert the node to an index
    auto result = m_representation->GenerateIsLeafTileOp(
        rewriter, op, operands[0], nodeIndex);

    // auto nodeIndexType = nodeIndex.getType().cast<IndexType>();
    // assert(nodeIndexType);

    rewriter.replaceOp(op, static_cast<Value>(result));

    return mlir::success();
  }
};

struct CacheTreesFromEnsembleOpLowering : public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;
  std::shared_ptr<decisionforest::IModelSerializer> m_serializer;
  CacheTreesFromEnsembleOpLowering(
      MLIRContext *ctx,
      std::shared_ptr<decisionforest::IRepresentation> representation,
      std::shared_ptr<decisionforest::IModelSerializer> serializer)
      : ConversionPattern(
            mlir::decisionforest::CacheTreesFromEnsembleOp::getOperationName(),
            1 /*benefit*/, ctx),
        m_representation(representation), m_serializer(serializer) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    m_representation->LowerCacheTreeOp(rewriter, op, operands, m_serializer);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct CacheRowsOpLowering : public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;
  CacheRowsOpLowering(
      MLIRContext *ctx,
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : ConversionPattern(
            mlir::decisionforest::CacheInputRowsOp::getOperationName(),
            1 /*benefit*/, ctx),
        m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    m_representation->LowerCacheRowsOp(rewriter, op, operands);
    return mlir::success();
  }
};

struct MidLevelIRToMemrefLoweringPass
    : public PassWrapper<MidLevelIRToMemrefLoweringPass,
                         OperationPass<mlir::ModuleOp>> {
  std::shared_ptr<decisionforest::IModelSerializer> m_serializer;
  std::shared_ptr<decisionforest::IRepresentation> m_representation;
  MidLevelIRToMemrefLoweringPass(
      std::shared_ptr<decisionforest::IModelSerializer> serializer,
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : m_serializer(serializer), m_representation(representation) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() final {
    // [BUG!!] TODO Since MLIR runs this pass multi-threaded, if multiple passes
    // access the representation object need to be protected!
    m_representation->InitRepresentation();

    ConversionTarget target(getContext());

    target.addLegalDialect<
        AffineDialect, memref::MemRefDialect, scf::SCFDialect,
        decisionforest::DecisionForestDialect, vector::VectorDialect,
        math::MathDialect, arith::ArithDialect, func::FuncDialect,
        gpu::GPUDialect>();

    target.addIllegalOp<
        decisionforest::EnsembleConstantOp,
        decisionforest::GetTreeFromEnsembleOp, decisionforest::GetRootOp,
        decisionforest::IsLeafOp, decisionforest::IsLeafTileOp,
        decisionforest::TraverseTreeTileOp,
        decisionforest::InterleavedTraverseTreeTileOp,
        decisionforest::GetLeafValueOp, decisionforest::GetLeafTileValueOp,
        decisionforest::GetTreeClassIdOp,
        decisionforest::CacheTreesFromEnsembleOp,
        decisionforest::CacheInputRowsOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<EnsembleConstantOpLowering>(patterns.getContext(),
                                             m_serializer, m_representation);
    patterns.add<TraverseTreeTileOpLowering>(patterns.getContext(),
                                             m_representation);
    patterns.add<InterleavedTraverseTreeTileOpLowering>(patterns.getContext(),
                                                        m_representation);
    patterns.add<GetRootOpLowering>(patterns.getContext(), m_representation);
    patterns.add<GetTreeOpLowering>(patterns.getContext(), m_representation);
    patterns.add<GetTreeClassIdOpLowering>(patterns.getContext(),
                                           m_representation);
    patterns.add<GetLeafValueOpLowering>(patterns.getContext(),
                                         m_representation);
    patterns.add<GetLeafTileValueOpLowering>(patterns.getContext(),
                                             m_representation);
    patterns.add<IsLeafOpLowering>(patterns.getContext(), m_representation);
    patterns.add<IsLeafTileOpLowering>(patterns.getContext(), m_representation);
    patterns.add<CacheTreesFromEnsembleOpLowering>(
        patterns.getContext(), m_representation, m_serializer);
    patterns.add<CacheRowsOpLowering>(patterns.getContext(), m_representation);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

#ifdef TREEBEARD_GPU_SUPPORT
struct MidLevelIRToGPUMemrefLoweringPass
    : public PassWrapper<MidLevelIRToGPUMemrefLoweringPass,
                         OperationPass<mlir::ModuleOp>> {
  std::shared_ptr<decisionforest::IModelSerializer> m_serializer;
  std::shared_ptr<decisionforest::IRepresentation> m_representation;
  MidLevelIRToGPUMemrefLoweringPass(
      std::shared_ptr<decisionforest::IModelSerializer> serializer,
      std::shared_ptr<decisionforest::IRepresentation> representation)
      : m_serializer(serializer), m_representation(representation) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() final {
    // [BUG!!] TODO Since MLIR runs this pass multi-threaded, if multiple passes
    // access the representation object need to be protected!
    ConversionTarget target(getContext());

    target.addLegalDialect<
        AffineDialect, memref::MemRefDialect, scf::SCFDialect,
        decisionforest::DecisionForestDialect, vector::VectorDialect,
        math::MathDialect, arith::ArithDialect, func::FuncDialect,
        gpu::GPUDialect>();

    target.addIllegalOp<
        decisionforest::EnsembleConstantOp,
        decisionforest::GetTreeFromEnsembleOp, decisionforest::GetRootOp,
        decisionforest::IsLeafOp, decisionforest::IsLeafTileOp,
        decisionforest::TraverseTreeTileOp,
        decisionforest::InterleavedTraverseTreeTileOp,
        decisionforest::GetLeafValueOp, decisionforest::GetLeafTileValueOp,
        decisionforest::GetTreeClassIdOp,
        decisionforest::CacheTreesFromEnsembleOp,
        decisionforest::CacheInputRowsOp,
        decisionforest::CooperativeTraverseTreeTileOp>();

    auto traverseLoweringState = std::make_shared<GPUTraverseLoweringState>();

    RewritePatternSet patterns(&getContext());
    patterns.add<EnsembleConstantOpGPULowering>(patterns.getContext(),
                                                m_serializer, m_representation);
    patterns.add<CooperativeTraverseTreeTileOpLowering>(
        patterns.getContext(), m_representation, traverseLoweringState);
    patterns.add<TraverseTreeTileOpLowering>(patterns.getContext(),
                                             m_representation);
    patterns.add<InterleavedTraverseTreeTileOpLowering>(patterns.getContext(),
                                                        m_representation);
    patterns.add<GPUGetRootOpLowering>(patterns.getContext(), m_representation,
                                       traverseLoweringState);
    patterns.add<GetTreeOpLowering>(patterns.getContext(), m_representation);
    patterns.add<GetTreeClassIdOpLowering>(patterns.getContext(),
                                           m_representation);
    patterns.add<GetLeafValueOpLowering>(patterns.getContext(),
                                         m_representation);
    patterns.add<GetLeafTileValueOpLowering>(patterns.getContext(),
                                             m_representation);
    patterns.add<IsLeafOpLowering>(patterns.getContext(), m_representation);
    patterns.add<IsLeafTileOpLowering>(patterns.getContext(), m_representation);
    patterns.add<CacheTreesFromEnsembleOpLowering>(
        patterns.getContext(), m_representation, m_serializer);
    patterns.add<CacheRowsOpLowering>(patterns.getContext(), m_representation);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
#endif // TREEBEARD_GPU_SUPPORT

} // namespace decisionforest
} // namespace mlir

namespace mlir {
namespace decisionforest {

void LowerEnsembleToMemrefs(mlir::MLIRContext &context, mlir::ModuleOp module,
                            std::shared_ptr<IModelSerializer> serializer,
                            std::shared_ptr<IRepresentation> representation) {
  // llvm::DebugFlag = true;
  // Lower from mid-level IR to low-level IR
  representation->InitRepresentation();

  mlir::PassManager pm(&context);
  pm.addPass(std::make_unique<MidLevelIRToMemrefLoweringPass>(serializer,
                                                              representation));

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to memrefs failed.\n";
  }
}

#ifdef TREEBEARD_GPU_SUPPORT

void LowerGPUEnsembleToMemrefs(
    mlir::MLIRContext &context, mlir::ModuleOp module,
    std::shared_ptr<IModelSerializer> serializer,
    std::shared_ptr<IRepresentation> representation) {
  // llvm::DebugFlag = true;
  representation->InitRepresentation();
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  pm.addPass(std::make_unique<MidLevelIRToGPUMemrefLoweringPass>(
      serializer, representation));

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to memrefs failed.\n";
  }
}

#endif // TREEBEARD_GPU_SUPPORT

} // namespace decisionforest
} // namespace mlir