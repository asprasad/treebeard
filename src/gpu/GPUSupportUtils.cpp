#include "Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "GPUSupportUtils.h"

using namespace mlir;

namespace 
{

// Replace all uses of the input argument memref with the gpu memref inside
// the gpu kernel
void ReplaceArgumentUsesWithGPUMemref(func::FuncOp func, Value inputGPUMemref, Value outputGPUMemref) {
  Value argMemref = func.getArgument(0);
  for (auto& use: argMemref.getUses()) {
    auto owningOp=use.getOwner();
    gpu::LaunchOp gpuLaunchOp = owningOp->getParentOfType<gpu::LaunchOp>();
    if (gpuLaunchOp)
      owningOp->replaceUsesOfWith(argMemref, inputGPUMemref);
  }
  Value outputMemref = func.getArgument(1);
  for (auto& use: outputMemref.getUses()) {
    auto owningOp=use.getOwner();
    gpu::LaunchOp gpuLaunchOp = owningOp->getParentOfType<gpu::LaunchOp>();
    if (gpuLaunchOp)
      owningOp->replaceUsesOfWith(outputMemref, outputGPUMemref);
  }

  // Workaround for a bug in the use chains of the second arg
  func.walk([&](memref::LoadOp loadOp) {
    auto owningOp=loadOp.getOperation();
    gpu::LaunchOp gpuLaunchOp = owningOp->getParentOfType<gpu::LaunchOp>();
    if (gpuLaunchOp)
      owningOp->replaceUsesOfWith(outputMemref, outputGPUMemref);
  });
}

void AddGPUAllocationsAndTransfers(mlir::ModuleOp module) {
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getName() == "Prediction_Function") {
      auto location = func.getLoc();
      // Add the required transfers here
      mlir::OpBuilder builder(func.getContext());
      func.walk([&](gpu::LaunchOp launchOp){
        builder.setInsertionPoint(launchOp.getOperation());
      });
      
      auto waitOp = builder.create<gpu::WaitOp>(location, gpu::AsyncTokenType::get(module.getContext()), ValueRange{});
      auto inputAlloc = builder.create<gpu::AllocOp>(location, func.getArgument(0).getType(), waitOp.getAsyncToken().getType(), 
                                                     ValueRange{waitOp.getAsyncToken()}, ValueRange{}, ValueRange{});
      auto inputTransfer = builder.create<gpu::MemcpyOp>(location, inputAlloc.getAsyncToken().getType(), ValueRange{inputAlloc.getAsyncToken()}, 
                                                         inputAlloc.getMemref(), static_cast<Value>(func.getArgument(0)));

      auto outputAlloc = builder.create<gpu::AllocOp>(location, func.getArgument(1).getType(), inputTransfer.getAsyncToken().getType(), 
                                                      ValueRange{inputTransfer.getAsyncToken()}, ValueRange{}, ValueRange{});

      // Find the return statement and add a tranfer before it
      auto& region = func.getRegion();
      for (auto& block: region.getBlocks()) {
        for (auto returnOp : block.getOps<func::ReturnOp>()) {
          auto op = returnOp.getOperation();
          auto insertPoint = builder.saveInsertionPoint();
          builder.setInsertionPoint(op);
          auto waitBeforeReturn = builder.create<gpu::WaitOp>(location, gpu::AsyncTokenType::get(module.getContext()), ValueRange{});
          /*auto outputTransfer = */ builder.create<gpu::MemcpyOp>(location, waitBeforeReturn.getAsyncToken().getType(), ValueRange{waitBeforeReturn.getAsyncToken()}, 
                                                                   static_cast<Value>(func.getArgument(1)), outputAlloc.getMemref());
          builder.setInsertionPoint(insertPoint.getBlock(), insertPoint.getPoint());                                                              
        }
      }
      
      ReplaceArgumentUsesWithGPUMemref(func, inputAlloc.getMemref(), outputAlloc.getMemref());
    }
    return mlir::WalkResult::advance();
  });
}

} // anonymous namespace

namespace mlir
{
namespace decisionforest
{

void GreedilyMapParallelLoopsToGPU(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(createGpuMapParallelLoopsPass());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to mid level IR failed.\n";
  }
}

void ConvertParallelLoopsToGPU(mlir::MLIRContext& context, mlir::ModuleOp module) {
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(createParallelLoopToGpuPass());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to mid level IR failed.\n";
  }
  // Insert required GPU allocations and transfers to and from the gpu memory.
  // Replace the uses of the function arguments with the new gpu allocations.
  AddGPUAllocationsAndTransfers(module);
}

void OutlineGPUKernels(mlir::MLIRContext& context, mlir::ModuleOp module) {
  mlir::PassManager pm(&context);
  pm.addPass(createGpuKernelOutliningPass());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to mid level IR failed.\n";
  }
}

} // decisionforest
} // mlir