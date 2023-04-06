#ifdef TREEBEARD_GPU_SUPPORT

#include "Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"

#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "GPUSupportUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"
#include <vector>

using namespace mlir;

namespace 
{

// Replace all uses of the input argument memref with the gpu memref inside
// the gpu kernel
void ReplaceCPUReferencesWithGPUMemref(const llvm::DenseMap<Value, Value>& cpuToGPUMemrefMap) {
  for (auto& cpuToGPUMemrefPair: cpuToGPUMemrefMap) {
    Value cpuMemref = cpuToGPUMemrefPair.first;
    Value gpuMemref = cpuToGPUMemrefPair.second;
    for (auto& use: cpuMemref.getUses()) {
      auto *owningOp=use.getOwner();
      gpu::LaunchOp gpuLaunchOp = owningOp->getParentOfType<gpu::LaunchOp>();
      if (gpuLaunchOp) {
        owningOp->replaceUsesOfWith(cpuMemref, gpuMemref);

        // Workaround for a bug in the use chains of the second arg
        gpuLaunchOp.walk([&](memref::LoadOp loadOp) {
          loadOp->replaceUsesOfWith(cpuMemref, gpuMemref);
        });
      }
    }
  }
}

void AddGPUAllocationsAndTransfers(mlir::ModuleOp module) {
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getName() == "Prediction_Function") {
      auto location = func.getLoc();

      mlir::OpBuilder builder(func.getContext());

      std::vector<Value> memrefsInFuncOp;
      std::vector<Value> memrefsUsedPostGpuLaunch;

      for (auto& arg : func.getArguments()) {
        if (arg.getType().isa<MemRefType>())
          memrefsInFuncOp.push_back(arg);
      }

      // Collect memrefs used in func op and after gpu.launch
      for (auto &funcRegion : func->getRegions()) {
        for (auto &block : funcRegion.getBlocks()) {
          bool opIsAfterGpuLaunch = false;
          for (auto &op : block.getOperations()) {
            auto gpuOp = dyn_cast<gpu::LaunchOp>(op);
            if (gpuOp) {
              opIsAfterGpuLaunch = true;
              continue;
            }

            if (opIsAfterGpuLaunch) {
              op.walk([&](Operation* operation) {
                for (const auto &operand : operation->getOperands()) {
                  if (operand.getType().isa<MemRefType>())
                    memrefsUsedPostGpuLaunch.push_back(operand);
                }
              });
            } else {
              op.walk([&](Operation *operation) {
                for (const auto &operand : operation->getOperands()) {
                  if (operand.getType().isa<MemRefType>())
                    memrefsInFuncOp.push_back(operand);
                }
              });
            }
          }
        }
      }

      // Determine memrefs to be transferred to and from the gpu.
      llvm::DenseSet<Value> requiresTransferToGpu, requiresTransferFromGpu;
      for (auto& result : memrefsInFuncOp) {
        for (auto& use : result.getUses()) {
          auto *owningOp=use.getOwner();
          auto launchOp = owningOp->getParentOfType<gpu::LaunchOp>();
          if (launchOp) {
            requiresTransferToGpu.insert(result); 

            // Relies on the fact that we currently don't allocate anything on the GPU for the output
            // without allocating on the input side. If we do allocate, that needs to be handled separately.
            for (auto &postGpuResult : memrefsUsedPostGpuLaunch) {
              if (postGpuResult == result) {
                requiresTransferFromGpu.insert(result);
              }
            }
          }
        }
      }

      gpu::LaunchOp gpuLaunchOp;
      func.walk([&](gpu::LaunchOp launchOp){
        builder.setInsertionPoint(launchOp.getOperation());
        gpuLaunchOp = launchOp;
      });

      auto waitOp = builder.create<gpu::WaitOp>(location, gpu::AsyncTokenType::get(module.getContext()), ValueRange{});
      Value waitToken = waitOp.getAsyncToken();

      llvm::DenseMap<Value, Value> cpuToGpuMemrefMap, gpuToCpuMemrefMap;

      // Generate input transfers.
      // #TODOSampath - Each transfer doesn't have to wait for subsequent transfers. 
      // We can collect the wait tokens in a vector and wait for all of them using the gpu::WaitOp
      for (auto& transferToGpu : requiresTransferToGpu) {
        auto inputAlloc = builder.create<gpu::AllocOp>(location, transferToGpu.getType(), waitToken.getType(), 
                                                       ValueRange{waitToken}, ValueRange{}, ValueRange{});
        auto inputTransfer = builder.create<gpu::MemcpyOp>(location, inputAlloc.getAsyncToken().getType(), ValueRange{inputAlloc.getAsyncToken()}, 
                                                           inputAlloc.getMemref(), static_cast<Value>(transferToGpu));
        waitToken = inputTransfer.getAsyncToken();
        cpuToGpuMemrefMap[transferToGpu] = inputAlloc.getMemref();

        if (requiresTransferFromGpu.contains(transferToGpu)) {
          gpuToCpuMemrefMap[transferToGpu] = inputAlloc.getMemref();
        }
      }

      builder.setInsertionPointAfter(gpuLaunchOp.getOperation());
      auto waitForGpuKernel = builder.create<gpu::WaitOp>(location, gpu::AsyncTokenType::get(module.getContext()), ValueRange{});
      waitToken = waitForGpuKernel.getAsyncToken();

      // Copy out any values that are needed by the CPU.
      for (auto&cpuGpuPair : gpuToCpuMemrefMap) {
        auto cpuMemRef = cpuGpuPair.first;
        auto gpuMemRef = cpuGpuPair.second;

        auto outputTransfer = builder.create<gpu::MemcpyOp>(location, waitToken.getType(), waitToken, cpuMemRef, gpuMemRef);
        waitToken = outputTransfer.getAsyncToken();
      }

      auto waitForTransfers = builder.create<gpu::WaitOp>(location, gpu::AsyncTokenType::get(module.getContext()), waitToken);
      waitToken = waitForTransfers.getAsyncToken();

      // Deallocate the GPU memory. 
      // #TODOSampath - Assumes that we haven't allocated anything that we haven't used to transfer memory from CPU.
      for (auto& cpuGpuMemrefPairs : cpuToGpuMemrefMap) {
        auto dealloc = builder.create<gpu::DeallocOp>(location, waitToken.getType(), waitToken, cpuGpuMemrefPairs.second);
        waitToken = dealloc.getAsyncToken();
      } 

      // wait for final dealloc.
      builder.create<gpu::WaitOp>(location, gpu::AsyncTokenType::get(module.getContext()), waitToken);

      ReplaceCPUReferencesWithGPUMemref(cpuToGpuMemrefMap);
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

#endif // TREEBEARD_GPU_SUPPORT