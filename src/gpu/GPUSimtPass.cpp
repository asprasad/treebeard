#ifdef TREEBEARD_GPU_SUPPORT
#include <iostream>
#include <memory>
#include <vector>
#include "Dialect.h"
// #include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

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


using namespace mlir;

namespace mlir {
namespace decisionforest {

struct TraverseToCooperativeTraverseTreeTileOp : public ConversionPattern {
  TraverseToCooperativeTraverseTreeTileOp(MLIRContext *ctx) 
  : ConversionPattern(mlir::decisionforest::TraverseTreeTileOp::getOperationName(), 1 /*benefit*/, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
    auto traverseTileOp = AssertOpIsOfType<mlir::decisionforest::TraverseTreeTileOp>(op);
    auto traverseTileAdaptor = decisionforest::TraverseTreeTileOpAdaptor(traverseTileOp);
    assert(operands.size() == 3);
    if (!traverseTileOp)
        return mlir::failure();
    
    auto treeType = traverseTileAdaptor.getTree().getType().cast<decisionforest::TreeType>();
    if (treeType.getTileSize() == 1)
      return mlir::failure();
    
    // rewriter.replaceOp(op, static_cast<Value>(codeGenStateMachine.GetResult()[0]));
    return mlir::failure();
  }
};


struct ConvertTraverseToCooperativeTraverse: public PassWrapper<ConvertTraverseToCooperativeTraverse, OperationPass<mlir::func::FuncOp>> {
  ConvertTraverseToCooperativeTraverse() { }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, 
                    memref::MemRefDialect,
                    scf::SCFDialect,
                    decisionforest::DecisionForestDialect>();
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalDialect<AffineDialect, memref::MemRefDialect, 
                           scf::SCFDialect, decisionforest::DecisionForestDialect, vector::VectorDialect,
                           math::MathDialect, arith::ArithDialect, func::FuncDialect, gpu::GPUDialect>();

    target.addIllegalOp<decisionforest::TraverseTreeTileOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<TraverseToCooperativeTraverseTreeTileOp>(patterns.getContext());
      
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
  }
};

void ConvertTraverseToSimtTraverse(mlir::MLIRContext &context,
                                   mlir::ModuleOp module) {
  // llvm::DebugFlag = true;
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(std::make_unique<ConvertTraverseToCooperativeTraverse>());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "GPU SIMT pass failed.\n";
  }
}

} // namespace decisionforest
} // namespace mlir

#endif