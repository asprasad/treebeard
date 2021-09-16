#include "Dialect.h"
// #include "Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

// #include "mlir/Dialect/StandardOps/Transforms/Passes.h"
// #include "mlir/Dialect/SCF/Passes.h"
// #include "mlir/Dialect/Tensor/Transforms/Passes.h"
// #include "mlir/Transforms/Passes.h"
// #include "mlir/Dialect/Linalg/Passes.h"

using namespace mlir;

namespace {
struct DecisionForestToLLVMLoweringPass : public PassWrapper<DecisionForestToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect, AffineDialect, memref::MemRefDialect, tensor::TensorDialect, StandardOpsDialect>();
  }
  void runOnOperation() final;
};
} // end anonymous namespace

void DecisionForestToLLVMLoweringPass::runOnOperation() {
  // define the conversion target
  // LowerToLLVMOptions options(&getContext());
  // options.useBarePtrCallConv = true;
  // options.emitCWrappers = true;
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
  target.addIllegalDialect<memref::MemRefDialect>();
  // target.addLegalOp<memref::GlobalOp>(), memref::GetGlobalOp, memref::SubViewOp, memref::LoadOp, memref::StoreOp>();
  target.addLegalDialect<decisionforest::DecisionForestDialect>();
  // target.addLegalOp<FuncOp, ReturnOp>();

  // Convert our MemRef types to LLVM types
  LLVMTypeConverter typeConverter(&getContext()); //, options);

  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateLoopToStdConversionPatterns(patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();  
}

namespace mlir
{
namespace decisionforest
{
void LowerToLLVM(mlir::MLIRContext& context, mlir::ModuleOp module) {
  // Lower from high-level IR to mid-level IR
  mlir::PassManager pm(&context);
  // mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  pm.addPass(std::make_unique<DecisionForestToLLVMLoweringPass>());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to LLVM failed.\n";
  }
}

// The routine below lowers tensors to memrefs. We don't need it currently
// as we're not using tensors. Commenting it out so we can remove some MLIR 
// link time dependencies. 

// void LowerTensorTypes(mlir::MLIRContext& context, mlir::ModuleOp module) {
//   // Lower from high-level IR to mid-level IR
//   mlir::PassManager pm(&context);
//   // mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
//   // Partial bufferization passes.
//   pm.addPass(createTensorConstantBufferizePass());
//   pm.addNestedPass<FuncOp>(createSCFBufferizePass());
//   pm.addNestedPass<FuncOp>(createStdBufferizePass());
//   pm.addNestedPass<FuncOp>(createLinalgBufferizePass());
//   pm.addNestedPass<FuncOp>(createTensorBufferizePass());
//   pm.addPass(createFuncBufferizePass());

//   // Finalizing bufferization pass.
//   pm.addNestedPass<FuncOp>(createFinalizingBufferizePass());

//   if (mlir::failed(pm.run(module))) {
//     llvm::errs() << "Conversion from NodeType to Index failed.\n";
//   }
// }

} // decisionforest
} // mlir