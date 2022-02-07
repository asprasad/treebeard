#include "Dialect.h"
// #include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace
{

struct WalkDecisionTreeOpLowering: public ConversionPattern {
  WalkDecisionTreeOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::WalkDecisionTreeOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    mlir::decisionforest::WalkDecisionTreeOp walkTreeOp = llvm::dyn_cast<mlir::decisionforest::WalkDecisionTreeOp>(op);
    assert(walkTreeOp);
    assert(operands.size() == 2);
    if (!walkTreeOp)
        return mlir::failure();

    auto tree = operands[0];
    auto inputRow = operands[1];
    
    auto location = op->getLoc();
    auto context = inputRow.getContext();
    auto treeType = tree.getType().cast<mlir::decisionforest::TreeType>();

    auto nodeType = mlir::decisionforest::NodeType::get(context);
    auto node = rewriter.create<decisionforest::GetRootOp>(location, nodeType, tree);

    scf::WhileOp whileLoop = rewriter.create<scf::WhileOp>(location, nodeType, static_cast<Value>(node));
    Block *before = rewriter.createBlock(&whileLoop.before(), {}, nodeType, location);
    Block *after = rewriter.createBlock(&whileLoop.after(), {}, nodeType, location);

    // Create the 'do' part for the condition.
    {
        rewriter.setInsertionPointToStart(&whileLoop.before().front());
        auto node = before->getArguments()[0];
        auto isLeaf = rewriter.create<decisionforest::IsLeafOp>(location, rewriter.getI1Type(), tree, node);
        auto falseConstant = rewriter.create<arith::ConstantIntOp>(location, int64_t(0), rewriter.getI1Type());
        auto equalTo = rewriter.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, static_cast<Value>(isLeaf), static_cast<Value>(falseConstant));
        rewriter.create<scf::ConditionOp>(location, equalTo, ValueRange({node})); // this is the terminator
    }
    // Create the loop body
    {
        rewriter.setInsertionPointToStart(&whileLoop.after().front());
        auto node = after->getArguments()[0];
        
        auto traverseTile = rewriter.create<decisionforest::TraverseTreeTileOp>(
          location,
          nodeType,
          tree,
          node,
          inputRow);

        rewriter.create<scf::YieldOp>(location, static_cast<Value>(traverseTile));
    }
    rewriter.setInsertionPointAfter(whileLoop);
    auto treePrediction = rewriter.create<decisionforest::GetLeafValueOp>(location, treeType.getThresholdType(), tree, whileLoop.results()[0]);
    rewriter.replaceOp(op, static_cast<Value>(treePrediction));

    return mlir::success();
  }
};

struct WalkDecisionTreeOpLoweringPass: public PassWrapper<WalkDecisionTreeOpLoweringPass, FunctionPass> {
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<StandardOpsDialect, scf::SCFDialect>();
  }

  void runOnFunction() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<memref::MemRefDialect, StandardOpsDialect, scf::SCFDialect, 
                           decisionforest::DecisionForestDialect, math::MathDialect, arith::ArithmeticDialect>();
    target.addIllegalOp<decisionforest::WalkDecisionTreeOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<WalkDecisionTreeOpLowering>(&getContext());

    if (failed(applyPartialConversion(getFunction(), target, std::move(patterns))))
        signalPassFailure();
  }
};

}

namespace mlir
{
namespace decisionforest
{

void AddWalkDecisionTreeOpLoweringPass(mlir::OpPassManager &optPM) {
  optPM.addPass(std::make_unique<WalkDecisionTreeOpLoweringPass>());
}

}
}