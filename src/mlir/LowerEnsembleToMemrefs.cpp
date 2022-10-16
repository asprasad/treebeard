#include <iostream>
#include <memory>
#include "Dialect.h"
// #include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "MemrefTypes.h"
#include "Dialect.h"
#include "TreeTilingUtils.h"
#include "TiledTree.h"
#include "schedule.h"
#include "CodeGenStateMachine.h"
#include "TraverseTreeTileOpLowering.h"
#include "OpLoweringUtils.h"
#include "ModelSerializers.h"
#include "Representations.h"
#include "LIRLoweringHelpers.h"

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
* [Selected] [Option] What if we add a global memref to the module and a function 
  that just returns this memref? We can the populate it in C code 
  and the inference function can use the global. We could just have
  multiple global memrefs if we want different tiling for different
  trees. 
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


Value getLUT;

namespace {

void ClearGlobalMaps() {
  ensembleConstantToMemrefsMap.clear();
  getTreeOperationMap.clear();
}

void ClearSparseGlobalMaps() {
  sparseEnsembleConstantToMemrefsMap.clear();
  sparseGetTreeOperationMap.clear();
}

struct EnsembleConstantOpLowering: public ConversionPattern {
  std::shared_ptr<decisionforest::IModelSerializer> m_serializer;
  std::shared_ptr<decisionforest::IRepresentation> m_representation;

  EnsembleConstantOpLowering(MLIRContext *ctx, std::shared_ptr<decisionforest::IModelSerializer> serializer,
                             std::shared_ptr<decisionforest::IRepresentation> representation)
   : ConversionPattern(mlir::decisionforest::EnsembleConstantOp::getOperationName(), 1 /*benefit*/, ctx), 
      m_serializer(serializer), m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    mlir::decisionforest::EnsembleConstantOp ensembleConstOp = llvm::dyn_cast<mlir::decisionforest::EnsembleConstantOp>(op);
    assert(ensembleConstOp);
    assert(operands.size() == 0);
    if (!ensembleConstOp)
        return mlir::failure();
    
    auto location = op->getLoc();
    auto owningModule = op->getParentOfType<mlir::ModuleOp>();
    assert (owningModule);
    
    auto ret = m_representation->GenerateModelGlobals(op, operands, rewriter, m_serializer);
    if (ret.failed()) {
      return ret;
    }

    auto forestType = ensembleConstOp.getResult().getType().cast<decisionforest::TreeEnsembleType>();
    auto firstTreeType = forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
    auto firstTreeTileSize = firstTreeType.getTileSize();

    // Type lookUpTableMemrefType;
    // Value getLUT;
    if (firstTreeTileSize > 1) {
      std::string lookupTableMemrefName = "lookupTable";
      auto lookUpTableMemrefType = AddChildIndexLookUpTable(owningModule, ensembleConstOp, rewriter, location, lookupTableMemrefName);
      getLUT = rewriter.create<memref::GetGlobalOp>(location, lookUpTableMemrefType, lookupTableMemrefName);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }

  Type AddChildIndexLookUpTable(mlir::ModuleOp module, mlir::decisionforest::EnsembleConstantOp& ensembleConstOp,
                                 ConversionPatternRewriter &rewriter, Location location, std::string& lookupTableMemrefName) const {
    SaveAndRestoreInsertionPoint saveAndRestoreInsertPoint(rewriter);
    rewriter.setInsertionPoint(&module.front());

    auto forestType = ensembleConstOp.getResult().getType().cast<decisionforest::TreeEnsembleType>();
    // We will assume that all trees have the same tile size
    auto numTrees = static_cast<int32_t>(forestType.getNumberOfTrees());
    assert(numTrees > 0);
    auto firstTreeType = forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
    auto firstTreeTileSize = firstTreeType.getTileSize();
    for (int32_t i=1 ; i<numTrees ; ++i) {
      auto treeType = forestType.getTreeType(i).cast<mlir::decisionforest::TreeType>();
      auto tileSize = treeType.getTileSize();
      assert (firstTreeTileSize == tileSize && "All tree's should have the same tile size");
    }
    auto tileSize = firstTreeTileSize;
    if (tileSize == 1)
      return Type(); // We don't need a lookup table if the tile size is 1
    
    auto numberOfTileOutcomes = static_cast<int>(std::pow(2, tileSize));
    auto numberOfTileShapes = mlir::decisionforest::TileShapeToTileIDMap::NumberOfTileShapes(tileSize);
    // TODO We may need to implement something smarter here. We don't really need I8's for each outcome. We could store all outcomes
    // in a single int64 for tile size 4 for example (each entry needs 3 bits and there are 16 entries -- one for each outcome). 
    auto lutMemrefType = MemRefType::get({numberOfTileShapes, numberOfTileOutcomes}, rewriter.getI8Type());

    rewriter.create<memref::GlobalOp>(location, lookupTableMemrefName,
                                      /*sym_visibility=*/rewriter.getStringAttr("private"),
                                      /*type=*/lutMemrefType,
                                      /*initial_value=*/rewriter.getUnitAttr(),
                                      /*constant=*/false, IntegerAttr());
    AddGlobalMemrefGetter(module, lookupTableMemrefName, lutMemrefType, rewriter, location);

    return lutMemrefType;
  }
  
};

struct GetTreeOpLowering: public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;
  GetTreeOpLowering(MLIRContext *ctx, std::shared_ptr<decisionforest::IRepresentation> representation) 
  : ConversionPattern(mlir::decisionforest::GetTreeFromEnsembleOp::getOperationName(), 1 /*benefit*/, ctx), m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    mlir::decisionforest::GetTreeFromEnsembleOp getTreeOp = llvm::dyn_cast<mlir::decisionforest::GetTreeFromEnsembleOp>(op);
    assert(getTreeOp);
    assert(operands.size() == 2);
    if (!getTreeOp)
        return mlir::failure();
    m_representation->GenerateTreeMemref(rewriter, op, operands[0], operands[1]);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

Value GetLUTFromTreeOperand(Value treeValue) {
  return getLUT;
}

struct GetTreeClassIdOpLowering: public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;

  GetTreeClassIdOpLowering(MLIRContext *ctx, std::shared_ptr<decisionforest::IRepresentation> representation) 
  : ConversionPattern(mlir::decisionforest::GetTreeClassIdOp::getOperationName(), 1 /*benefit*/, ctx), m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 2);
    auto classId = m_representation->GenerateGetTreeClassId(rewriter, op, operands[0], operands[1]);
    rewriter.replaceOp(op, static_cast<Value>(classId));
    return mlir::success();
  }
};

struct GetRootOpLowering: public ConversionPattern {
  std::shared_ptr<mlir::decisionforest::IRepresentation> m_representation;
  GetRootOpLowering(MLIRContext *ctx, std::shared_ptr<mlir::decisionforest::IRepresentation> representation)
   : ConversionPattern(mlir::decisionforest::GetRootOp::getOperationName(), 1 /*benefit*/, ctx), m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    auto getRootOp = AssertOpIsOfType<mlir::decisionforest::GetRootOp>(op);
    auto nodeIndexConst = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);

    auto treeMemref = m_representation->GetTreeMemref(operands[0]);
    auto nodeType = getRootOp.getResult().getType();
    auto node = rewriter.create<decisionforest::IndexToNodeOp>(op->getLoc(), nodeType, treeMemref, static_cast<Value>(nodeIndexConst));
    rewriter.replaceOp(op, static_cast<Value>(node));
    return mlir::success();
  }
};

struct IsLeafOpLowering: public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;
  IsLeafOpLowering(MLIRContext *ctx, std::shared_ptr<decisionforest::IRepresentation> representation) 
  : ConversionPattern(mlir::decisionforest::IsLeafOp::getOperationName(), 1 /*benefit*/, ctx), m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    
    // Create a subview of the model memref corresponding to this ensemble with the index equal to offsetMemref[treeIndex]
    mlir::decisionforest::IsLeafOp isLeafOp = AssertOpIsOfType<mlir::decisionforest::IsLeafOp>(op);
    assert(operands.size() == 2);
    if (!isLeafOp)
        return mlir::failure();
    
    auto location = op->getLoc();
    
    auto treeMemref = m_representation->GetTreeMemref(operands[0]);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert (treeMemrefType);
    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, rewriter.getIndexType(), treeMemref, operands[1]); // Convert the node to an index

    auto isLeafValue = m_representation->GenerateIsLeafOp(rewriter, op, operands[0], nodeIndex);
    rewriter.replaceOp(op, static_cast<Value>(isLeafValue));

    return mlir::success();
  }
};

struct InterleavedTraverseTreeTileOpLowering : public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;

  InterleavedTraverseTreeTileOpLowering(MLIRContext *ctx, std::shared_ptr<decisionforest::IRepresentation> representation) 
  : ConversionPattern(mlir::decisionforest::InterleavedTraverseTreeTileOp::getOperationName(), 1 /*benefit*/, ctx), m_representation(representation) {}
  
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,ConversionPatternRewriter &rewriter) const final {
    decisionforest::InterleavedTraverseTreeTileOpLoweringHelper traverseLoweringHelper(GetLUTFromTreeOperand, m_representation);
    return traverseLoweringHelper.matchAndRewrite(AssertOpIsOfType<mlir::decisionforest::InterleavedTraverseTreeTileOp>(op), operands, rewriter);
  }
};

struct TraverseTreeTileOpLowering : public ConversionPattern {
  std::shared_ptr<mlir::decisionforest::IRepresentation> m_representation;
  TraverseTreeTileOpLowering(MLIRContext *ctx, std::shared_ptr<mlir::decisionforest::IRepresentation> representation) 
  : ConversionPattern(mlir::decisionforest::TraverseTreeTileOp::getOperationName(), 1 /*benefit*/, ctx), m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    auto traverseTileOp = AssertOpIsOfType<mlir::decisionforest::TraverseTreeTileOp>(op);
    assert(operands.size() == 3);
    if (!traverseTileOp)
        return mlir::failure();
    
    auto treeMemref = m_representation->GetTreeMemref(operands[0]);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert (treeMemrefType);

    auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    decisionforest::InterleavedCodeGenStateMachine codeGenStateMachine;
    if (treeTileType.getTileSize() == 1)
      codeGenStateMachine.AddStateMachine(
        std::make_unique<decisionforest::ScalarTraverseTileCodeGenerator>(
          treeMemref,
          operands[2],
          operands[1],
          traverseTileOp.getResult().getType(),
          m_representation));
    else
      codeGenStateMachine.AddStateMachine(
        std::make_unique<decisionforest::VectorTraverseTileCodeGenerator>(
          operands[0],
          treeMemref,
          operands[2],
          operands[1],
          traverseTileOp.getResult().getType(),
          m_representation,
          GetLUTFromTreeOperand));
    
    // Emit code.
    auto location = op->getLoc();
    while (codeGenStateMachine.EmitNext(rewriter, location));
    
    rewriter.replaceOp(op, static_cast<Value>(codeGenStateMachine.GetResult()[0]));
    return mlir::success();
  }
};

struct GetLeafValueOpLowering : public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;

  GetLeafValueOpLowering(MLIRContext *ctx, std::shared_ptr<decisionforest::IRepresentation> representation) 
  : ConversionPattern(mlir::decisionforest::GetLeafValueOp::getOperationName(), 1 /*benefit*/, ctx), m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    auto getLeafVal = AssertOpIsOfType<mlir::decisionforest::GetLeafValueOp>(op);
    assert(operands.size() == 2);
    if (!getLeafVal)
        return mlir::failure();
    auto location = op->getLoc();

    auto treeMemref = m_representation->GetTreeMemref(operands[0]);
    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, rewriter.getIndexType(), treeMemref, operands[1]);
    if (decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintTreeNodeOp>(location, nodeIndex);
    }

    auto leafValue = m_representation->GenerateGetLeafValueOp(rewriter, op, operands[0], nodeIndex);
    // TODO cast the loaded value to the correct result type of the tree. 
    rewriter.replaceOp(op, static_cast<Value>(leafValue));
    return mlir::success();
  }
};

struct GetLeafTileValueOpLowering : public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;

  GetLeafTileValueOpLowering(MLIRContext *ctx, std::shared_ptr<decisionforest::IRepresentation> representation)
   : ConversionPattern(mlir::decisionforest::GetLeafTileValueOp::getOperationName(), 1 /*benefit*/, ctx), m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    auto getLeafVal = AssertOpIsOfType<mlir::decisionforest::GetLeafTileValueOp>(op);
    assert(operands.size() == 2);
    if (!getLeafVal)
        return mlir::failure();
    auto location = op->getLoc();

    auto treeMemref = m_representation->GetTreeMemref(operands[0]);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert (treeMemrefType);

    auto treeTileType = treeMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
    auto thresholdType = treeTileType.getThresholdFieldType();
    auto node = operands[1];
    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, rewriter.getIndexType(), treeMemref, node);
    if (decisionforest::InsertDebugHelpers) {
      rewriter.create<decisionforest::PrintTreeNodeOp>(location, nodeIndex);
    }

    // Load threshold
    // TODO Ideally, this should be a different op for when we deal with tile sizes != 1. We will then need to load 
    // a single threshold value and cast it the trees return type
    auto loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(location, thresholdType, treeMemref, static_cast<Value>(nodeIndex));
    Value leafValue = loadThresholdOp;
    
    if (treeTileType.getTileSize() != 1) {
      auto thresholdVectorType = thresholdType.cast<VectorType>();
      if (decisionforest::InsertDebugHelpers) {
        Value vectorVal = loadThresholdOp;
        if (!thresholdVectorType.getElementType().isF64()) {
          auto doubleVectorType = mlir::VectorType::get({ treeTileType.getTileSize() }, rewriter.getF64Type());
          vectorVal = rewriter.create<arith::ExtFOp>(location, doubleVectorType, loadThresholdOp);
        }
        InsertPrintVectorOp(rewriter, location, 0, 64, treeTileType.getTileSize(), vectorVal);
      }
      auto zeroConst = rewriter.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI32Type());
      auto extractElement = rewriter.create<vector::ExtractElementOp>(location, static_cast<Value>(loadThresholdOp), zeroConst);
      leafValue = extractElement;
    }
    
    // TODO cast the loaded value to the correct result type of the tree. 
    rewriter.replaceOp(op, static_cast<Value>(leafValue));
    return mlir::success();
  }
};

struct IsLeafTileOpLowering: public ConversionPattern {
  std::shared_ptr<decisionforest::IRepresentation> m_representation;
  IsLeafTileOpLowering(MLIRContext *ctx, std::shared_ptr<decisionforest::IRepresentation> representation) 
  : ConversionPattern(mlir::decisionforest::IsLeafTileOp::getOperationName(), 1 /*benefit*/, ctx), m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    
    // Create a subview of the model memref corresponding to this ensemble with the index equal to offsetMemref[treeIndex]
    mlir::decisionforest::IsLeafTileOp isLeafOp = AssertOpIsOfType<mlir::decisionforest::IsLeafTileOp>(op);
    assert(operands.size() == 2);
    if (!isLeafOp)
        return mlir::failure();
    
    auto location = op->getLoc();

    auto treeMemref = m_representation->GetTreeMemref(operands[0]);
    auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
    assert (treeMemrefType);

    auto nodeIndex = rewriter.create<decisionforest::NodeToIndexOp>(location, rewriter.getIndexType(), treeMemref, operands[1]); // Convert the node to an index
    auto result = m_representation->GenerateIsLeafTileOp(rewriter, op, operands[0], nodeIndex);

    // auto nodeIndexType = nodeIndex.getType().cast<IndexType>();
    // assert(nodeIndexType);

    rewriter.replaceOp(op, static_cast<Value>(result));

    return mlir::success();
  }
};

struct MidLevelIRToMemrefLoweringPass: public PassWrapper<MidLevelIRToMemrefLoweringPass, OperationPass<mlir::func::FuncOp>> {
  std::shared_ptr<decisionforest::IModelSerializer> m_serializer;
  std::shared_ptr<decisionforest::IRepresentation> m_representation;
  MidLevelIRToMemrefLoweringPass(std::shared_ptr<decisionforest::IModelSerializer> serializer, std::shared_ptr<decisionforest::IRepresentation> representation)
    :m_serializer(serializer), m_representation(representation) { }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() final {
    // [BUG!!] TODO Since MLIR runs this pass multi-threaded, if multiple passes access these globals, they need to be protected!
    
    // Clear the global maps that store the mappings for the ensemble constants
    ClearGlobalMaps();
    ClearSparseGlobalMaps();

    ConversionTarget target(getContext());

    target.addLegalDialect<AffineDialect, memref::MemRefDialect, 
                           scf::SCFDialect, decisionforest::DecisionForestDialect, vector::VectorDialect,
                           math::MathDialect, arith::ArithmeticDialect, func::FuncDialect>();

    target.addIllegalOp<decisionforest::EnsembleConstantOp,
                        decisionforest::GetTreeFromEnsembleOp,
                        decisionforest::GetRootOp,
                        decisionforest::IsLeafOp,
                        decisionforest::IsLeafTileOp,
                        decisionforest::TraverseTreeTileOp,
                        decisionforest::InterleavedTraverseTreeTileOp,
                        decisionforest::GetLeafValueOp,
                        decisionforest::GetLeafTileValueOp,
                        decisionforest::GetTreeClassIdOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<EnsembleConstantOpLowering>(patterns.getContext(), m_serializer, m_representation);
    patterns.add<TraverseTreeTileOpLowering>(patterns.getContext(), m_representation);
    patterns.add<InterleavedTraverseTreeTileOpLowering>(patterns.getContext(), m_representation);
    patterns.add<GetRootOpLowering>(patterns.getContext(), m_representation);
    patterns.add<GetTreeOpLowering>(patterns.getContext(), m_representation);
    patterns.add<GetTreeClassIdOpLowering>(patterns.getContext(), m_representation);
    patterns.add<GetLeafValueOpLowering>(patterns.getContext(), m_representation);
    patterns.add<GetLeafTileValueOpLowering>(patterns.getContext(), m_representation);
    patterns.add<IsLeafOpLowering>(patterns.getContext(), m_representation);
    patterns.add<IsLeafTileOpLowering>(patterns.getContext(), m_representation);
      
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
  }
};
}

namespace mlir
{
namespace decisionforest
{

void LowerEnsembleToMemrefs(mlir::MLIRContext& context, mlir::ModuleOp module, 
                            std::shared_ptr<IModelSerializer> serializer,
                            std::shared_ptr<IRepresentation> representation) {
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(std::make_unique<MidLevelIRToMemrefLoweringPass>(serializer, representation));

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to memrefs failed.\n";
  }
}

} // decisionforest
} // mlir