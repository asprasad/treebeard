#include "Dialect.h"
// #include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace {

static RankedTensorType getRowTypeFromArgumentType(RankedTensorType type) {
  assert (type.hasRank() && "expected only rank shapes");
  return RankedTensorType::get({type.getShape()[1]}, type.getElementType());
}

struct PredictForestOpLowering: public ConversionPattern {
  PredictForestOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::PredictForestOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    mlir::decisionforest::PredictForestOp forestOp = llvm::dyn_cast<mlir::decisionforest::PredictForestOp>(op);
    assert(forestOp);
    assert(operands.size() == 1);
    if (!forestOp)
        return mlir::failure();

    auto inputArgument = operands[0];
    int64_t batchSize = 1;
    if (inputArgument.getType().isa<mlir::RankedTensorType>())
    {
        auto tensorType = inputArgument.getType().cast<mlir::RankedTensorType>();
        if (tensorType.getShape().size() != 2) // We can currently only deal with 2D tensors as inputs
            return mlir::failure();
        batchSize = tensorType.getShape()[0]; // The number of rows in our input tensor is the batchsize
        return LowerPredictForestOp_Batch(op, forestOp, operands, rewriter, tensorType, batchSize);  
    }
    else
    {
        assert(false && "Lowering for non-tensor argument not implemented");
        return mlir::failure();
    }
  }

  LogicalResult
  LowerPredictForestOp_Batch(Operation *op, mlir::decisionforest::PredictForestOp forestOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter, 
                            mlir::RankedTensorType tensorType, int64_t batchSize) const
  {
        auto location = op->getLoc();
        auto context = tensorType.getContext();

        // Create the return array (TODO This needs to become a function argument)
        auto resultType = op->getResults()[0].getType();
        auto resultTensorType = resultType.cast<mlir::RankedTensorType>();
        // auto resultMemrefType = convertTensorToMemRef(resultTensorType);
        // auto allocOp = rewriter.create<memref::AllocOp>(location, resultMemrefType);

        assert (tensorType.getElementType().isa<mlir::FloatType>());
        auto floatConstZero = rewriter.create<ConstantFloatOp>(location, llvm::APFloat(0.0), resultTensorType.getElementType().cast<mlir::FloatType>());
        std::vector<Value> values(batchSize, static_cast<Value>(floatConstZero));
        auto tensorResult = rewriter.create<tensor::FromElementsOp>(location, resultTensorType.getElementType(), values);
        // The input data on which we need to run inference
        auto data = operands[0];

        // Create the decision tree constant
        auto forestAttribute = forestOp.ensemble(); // Get the ensemble attribute
        auto forestType = forestAttribute.getType().cast<mlir::decisionforest::TreeEnsembleType>();
        auto forestConst = rewriter.create<mlir::decisionforest::EnsembleConstantOp>(location, forestType, forestAttribute);

        // Create a for loop over the inputs
        auto batchSizeConst = rewriter.create<ConstantIndexOp>(location, batchSize); 
        auto zeroConst = rewriter.create<ConstantIndexOp>(location, 0);
        auto oneIndexConst = rewriter.create<ConstantIndexOp>(location, 1);
        auto batchLoop = rewriter.create<scf::ForOp>(location, zeroConst, batchSizeConst, oneIndexConst, static_cast<Value>(tensorResult));
        
        rewriter.setInsertionPointToStart(batchLoop.getBody());
        auto i = batchLoop.getInductionVar();
        
        // Create a for loop over the trees
        int64_t numTrees = static_cast<int64_t>(forestType.getNumberOfTrees());
        auto ensembleSizeConst = rewriter.create<ConstantIndexOp>(location, numTrees); 
        // auto zeroConst2 = rewriter.create<ConstantIndexOp>(location, 0);
        // auto oneIndexConst2 = rewriter.create<ConstantIndexOp>(location, 1);
        auto treeLoop = rewriter.create<scf::ForOp>(location, zeroConst, ensembleSizeConst, oneIndexConst, static_cast<Value>(batchLoop.getBody()->getArguments()[1]));
        
        rewriter.setInsertionPointToStart(treeLoop.getBody());
        auto j = treeLoop.getInductionVar();

        // TODO The forest shouldn't contain the tree type because each tree may have a different type, 
        // but the tree type should have more information -- tiling for example. We need to construct 
        // the default tiling here.
        assert (forestType.doAllTreesHaveSameType());
        auto treeType = forestType.getTreeType(0).cast<mlir::decisionforest::TreeType>();
        auto tree = rewriter.create<decisionforest::GetTreeFromEnsembleOp>(location, treeType, forestConst, j);

        auto nodeType = mlir::decisionforest::NodeType::get(context);
        auto node = rewriter.create<decisionforest::GetRootOp>(location, nodeType, tree);
        
        // Create the while loop to walk the tree
        scf::WhileOp whileLoop = rewriter.create<scf::WhileOp>(location, nodeType, static_cast<Value>(node));
        Block *before = rewriter.createBlock(&whileLoop.before(), {}, nodeType, location);
        Block *after = rewriter.createBlock(&whileLoop.after(), {}, nodeType, location);

        // Create the 'do' part for the condition.
        {
            rewriter.setInsertionPointToStart(&whileLoop.before().front());
            auto node = before->getArguments()[0];
            auto isLeaf = rewriter.create<decisionforest::IsLeafOp>(location, rewriter.getI1Type(), tree, node);
            auto falseConstant = rewriter.create<ConstantIntOp>(location, int64_t(0), rewriter.getI1Type());
            auto equalTo = rewriter.create<CmpIOp>(location, mlir::CmpIPredicate::eq, static_cast<Value>(isLeaf), static_cast<Value>(falseConstant));
            rewriter.create<scf::ConditionOp>(location, equalTo, ValueRange({node})); // this is the terminator
        }
        // Create the loop body
        {
            rewriter.setInsertionPointToStart(&whileLoop.after().front());

            // Extract the slice of the input tensor for the current iteration -- row i
            auto rowType = getRowTypeFromArgumentType(tensorType);
            // auto rowSizeConst = rewriter.create<ConstantIndexOp>(location, rowType.getShape()[1]); 

            auto zeroIndexAttr = rewriter.getIndexAttr(0);
            auto oneIndexAttr = rewriter.getIndexAttr(1);
            auto rowSizeAttr = rewriter.getIndexAttr(rowType.getShape()[0]);
            auto row = rewriter.create<tensor::ExtractSliceOp>(location, rowType, static_cast<Value>(data), ArrayRef<OpFoldResult>({i, zeroIndexAttr}),
                                                              ArrayRef<OpFoldResult>({oneIndexAttr, rowSizeAttr}), ArrayRef<OpFoldResult>({oneIndexAttr, oneIndexAttr}));
            
            auto node = after->getArguments()[0];
            
            auto traverseTile = rewriter.create<decisionforest::TraverseTreeTileOp>(location, nodeType, tree, node, row);
            rewriter.create<scf::YieldOp>(location, static_cast<Value>(traverseTile));
        }
        rewriter.setInsertionPointAfter(whileLoop);
        auto treePrediction = rewriter.create<decisionforest::GetLeafValueOp>(location, treeType.getResultType(), tree, whileLoop.results()[0]);
        
        // result[i]
        auto resultElementType = resultTensorType.getElementType();
        auto readResultOfi = rewriter.create<tensor::ExtractOp>(location, resultElementType, treeLoop.getBody()->getArguments()[1], i);
        // Accumulate the tree prediction
        assert(forestType.getReductionType() == decisionforest::ReductionType::kAdd);
        auto accumulatedValue = rewriter.create<AddFOp>(location, resultElementType, readResultOfi, treePrediction);
        // result[i] = Accumulated value
        auto updatedResultTensor = rewriter.create<tensor::InsertOp>(location, resultTensorType, accumulatedValue, treeLoop.getBody()->getArguments()[1], i);
        rewriter.create<scf::YieldOp>(location, static_cast<Value>(updatedResultTensor));

        rewriter.setInsertionPointAfter(treeLoop);
        rewriter.create<scf::YieldOp>(location, static_cast<Value>(treeLoop.results()[0]));

        rewriter.replaceOp(op, static_cast<Value>(batchLoop.results()[0]));
        return mlir::success();
  }

};

struct HighLevelIRToMidLevelIRLoweringPass: public PassWrapper<HighLevelIRToMidLevelIRLoweringPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, tensor::TensorDialect, StandardOpsDialect, scf::SCFDialect>();
  }
  void runOnFunction() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<AffineDialect, memref::MemRefDialect, tensor::TensorDialect, StandardOpsDialect, scf::SCFDialect, decisionforest::DecisionForestDialect>();

    target.addIllegalOp<decisionforest::PredictForestOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<PredictForestOpLowering>(&getContext());

    // loweredSparseConstants.clear();

    if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
        signalPassFailure();
  }
};
}

namespace mlir
{
namespace decisionforest
{
void LowerFromHighLevelToMidLevelIR(mlir::MLIRContext& context, mlir::ModuleOp module) {
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  optPM.addPass(std::make_unique<HighLevelIRToMidLevelIRLoweringPass>());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to mid level IR failed.\n";
  }
}

} // decisionforest
} // mlir