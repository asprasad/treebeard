#include "Dialect.h"
#include "OpLoweringUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"

#include "llvm/Target/TargetMachine.h"


using namespace mlir;

namespace {

struct ReduceOpLowering: public ConversionPattern {
  ReduceOpLowering(MLIRContext *ctx) : ConversionPattern(mlir::decisionforest::ReduceOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    auto reduceOp = AssertOpIsOfType<decisionforest::ReduceOp>(op);
    auto location = reduceOp.getLoc();
    decisionforest::ReduceOpAdaptor adaptor(operands);

    auto loadVal = rewriter.create<memref::LoadOp>(location, adaptor.getTargetMemref(), adaptor.getIndices());
    auto addVal = rewriter.create<arith::AddFOp>(location, loadVal.getResult(), adaptor.getValue());
    rewriter.create<memref::StoreOp>(location, addVal.getResult(), adaptor.getTargetMemref(), adaptor.getIndices());

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct LowerReductionOps: public PassWrapper<LowerReductionOps, OperationPass<mlir::func::FuncOp>> {
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, arith::ArithDialect>();
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<memref::MemRefDialect, scf::SCFDialect, 
                           decisionforest::DecisionForestDialect, math::MathDialect,
                           arith::ArithDialect, func::FuncDialect, gpu::GPUDialect>();

    target.addIllegalOp<decisionforest::ReduceOp,
                        decisionforest::ReduceDimensionInplaceOp,
                        decisionforest::ReduceDimensionOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ReduceOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
  }
};

}

namespace mlir
{
namespace decisionforest
{

void LowerReduceOps(mlir::MLIRContext& context, mlir::ModuleOp module) {
  // llvm::DebugFlag = true;
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(std::make_unique<LowerReductionOps>());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering reduction ops failed.\n";
  }
  // llvm::DebugFlag = false;
}

}
}