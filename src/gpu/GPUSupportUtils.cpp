#ifdef TREEBEARD_GPU_SUPPORT

#include "Dialect.h"
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"

#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"

#include "TreebeardContext.h"

#include "GPUSupportUtils.h"

using namespace mlir;

namespace {

StringRef getMappingAttrName() { return "mapping"; }

FlatSymbolRefAttr getOrInsertFunction(std::string &functionName,
                                      mlir::FunctionType functionType,
                                      OpBuilder &rewriter, ModuleOp module) {
  auto *context = module.getContext();
  if (module.lookupSymbol<func::FuncOp>(functionName))
    return SymbolRefAttr::get(context, functionName);

  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto func = rewriter.create<func::FuncOp>(module.getLoc(), functionName,
                                            functionType);
  func.setVisibility(SymbolTable::Visibility::Private);
  return SymbolRefAttr::get(context, functionName);
}

FlatSymbolRefAttr getOrInsertStartTimerFunc(OpBuilder &rewriter,
                                            ModuleOp module) {
  std::string functionName = "startKernelTimer";
  auto functionType = rewriter.getFunctionType({}, {});
  return getOrInsertFunction(functionName, functionType, rewriter, module);
}

FlatSymbolRefAttr getOrInsertEndTimerFunc(OpBuilder &rewriter,
                                          ModuleOp module) {
  std::string functionName = "endKernelTimer";
  auto functionType = rewriter.getFunctionType({}, {});
  return getOrInsertFunction(functionName, functionType, rewriter, module);
}

void addGPUKernelTimingCalls(gpu::LaunchOp launchOp, mlir::OpBuilder &builder) {
  if (!decisionforest::measureGpuKernelTime)
    return;

  auto location = launchOp.getLoc();
  auto module = launchOp->getParentOfType<ModuleOp>();
  auto startTimerFunc = getOrInsertStartTimerFunc(builder, module);
  auto endTimerFunc = getOrInsertEndTimerFunc(builder, module);

  {
    PatternRewriter::InsertionGuard insertionGuard(builder);

    builder.setInsertionPoint(launchOp.getOperation());
    builder.create<func::CallOp>(location, startTimerFunc, TypeRange{});

    builder.setInsertionPointAfter(launchOp.getOperation());
    builder.create<func::CallOp>(location, endTimerFunc, TypeRange{});
  }
}

// Replace all uses of the input argument memref with the gpu memref inside
// the gpu kernel
void ReplaceCPUReferencesWithGPUMemref(
    const llvm::DenseMap<Value, Value> &cpuToGPUMemrefMap) {
  for (auto &cpuToGPUMemrefPair : cpuToGPUMemrefMap) {
    Value cpuMemref = cpuToGPUMemrefPair.first;
    Value gpuMemref = cpuToGPUMemrefPair.second;
    for (auto &use : cpuMemref.getUses()) {
      auto *owningOp = use.getOwner();
      gpu::LaunchOp gpuLaunchOp = owningOp->getParentOfType<gpu::LaunchOp>();
      if (gpuLaunchOp) {
        owningOp->replaceUsesOfWith(cpuMemref, gpuMemref);

        // Workaround for a bug in the use chains of the second arg
        gpuLaunchOp.walk([&](memref::LoadOp loadOp) {
          loadOp->replaceUsesOfWith(cpuMemref, gpuMemref);
        });

        gpuLaunchOp.walk([&](memref::StoreOp storeOp) {
          storeOp->replaceUsesOfWith(cpuMemref, gpuMemref);
        });

        gpuLaunchOp.walk([&](decisionforest::ReduceOp reduceOp) {
          reduceOp->replaceUsesOfWith(cpuMemref, gpuMemref);
        });

        gpuLaunchOp.walk([&](memref::SubViewOp subviewOp) {
          subviewOp->replaceUsesOfWith(cpuMemref, gpuMemref);
        });

        gpuLaunchOp.walk(
            [&](decisionforest::CooperativeReduceInplaceOp reduceOp) {
              reduceOp->replaceUsesOfWith(cpuMemref, gpuMemref);
            });

        gpuLaunchOp.walk(
            [&](decisionforest::CooperativeReduceArgMaxOp reduceOp) {
              reduceOp->replaceUsesOfWith(cpuMemref, gpuMemref);
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

      for (auto &arg : func.getArguments()) {
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
              op.walk([&](Operation *operation) {
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
      for (auto &result : memrefsInFuncOp) {
        for (auto &use : result.getUses()) {
          auto *owningOp = use.getOwner();
          auto launchOp = owningOp->getParentOfType<gpu::LaunchOp>();
          if (launchOp) {
            requiresTransferToGpu.insert(result);

            // Relies on the fact that we currently don't allocate anything on
            // the GPU for the output without allocating on the input side. If
            // we do allocate, that needs to be handled separately.
            for (auto &postGpuResult : memrefsUsedPostGpuLaunch) {
              if (postGpuResult == result) {
                requiresTransferFromGpu.insert(result);
              }
            }
          }
        }
      }

      gpu::LaunchOp gpuLaunchOp;
      func.walk([&](gpu::LaunchOp launchOp) {
        builder.setInsertionPoint(launchOp.getOperation());
        gpuLaunchOp = launchOp;
      });

      auto waitOp = builder.create<gpu::WaitOp>(
          location, gpu::AsyncTokenType::get(module.getContext()),
          ValueRange{});
      Value waitToken = waitOp.getAsyncToken();

      llvm::DenseMap<Value, Value> cpuToGpuMemrefMap, gpuToCpuMemrefMap;

      // Generate input transfers.
      // #TODOSampath - Each transfer doesn't have to wait for subsequent
      // transfers. We can collect the wait tokens in a vector and wait for all
      // of them using the gpu::WaitOp
      for (auto &transferToGpu : requiresTransferToGpu) {
        auto inputAlloc = builder.create<gpu::AllocOp>(
            location, transferToGpu.getType(), waitToken.getType(),
            ValueRange{waitToken}, ValueRange{}, ValueRange{});
        auto inputTransfer = builder.create<gpu::MemcpyOp>(
            location, inputAlloc.getAsyncToken().getType(),
            ValueRange{inputAlloc.getAsyncToken()}, inputAlloc.getMemref(),
            static_cast<Value>(transferToGpu));
        waitToken = inputTransfer.getAsyncToken();
        cpuToGpuMemrefMap[transferToGpu] = inputAlloc.getMemref();

        if (requiresTransferFromGpu.contains(transferToGpu)) {
          gpuToCpuMemrefMap[transferToGpu] = inputAlloc.getMemref();
        }
      }

      // Wait for all the transfers and allocs before the gpu.launch to finish
      /*auto waitForTransfersAndAllocs =*/builder.create<gpu::WaitOp>(
          location, Type(), ValueRange{waitToken});

      // Add the transfers as an async dependency to the gpu.launch op.
      // gpuLaunchOp.addAsyncDependency(waitToken);

      builder.setInsertionPointAfter(gpuLaunchOp.getOperation());

      auto waitForGpuKernel = builder.create<gpu::WaitOp>(
          location, gpu::AsyncTokenType::get(module.getContext()),
          ValueRange{});
      waitToken = waitForGpuKernel.getAsyncToken();
      // waitToken = gpuLaunchOp.getAsyncToken();

      // Copy out any values that are needed by the CPU.
      for (auto &cpuGpuPair : gpuToCpuMemrefMap) {
        auto cpuMemRef = cpuGpuPair.first;
        auto gpuMemRef = cpuGpuPair.second;

        auto outputTransfer = builder.create<gpu::MemcpyOp>(
            location, waitToken.getType(), waitToken, cpuMemRef, gpuMemRef);
        waitToken = outputTransfer.getAsyncToken();
      }

      // auto waitForTransfers = builder.create<gpu::WaitOp>(location,
      // gpu::AsyncTokenType::get(module.getContext()), waitToken); waitToken =
      // waitForTransfers.getAsyncToken();

      // Deallocate the GPU memory.
      // #TODOSampath - Assumes that we haven't allocated anything that we
      // haven't used to transfer memory from CPU.
      for (auto &cpuGpuMemrefPairs : cpuToGpuMemrefMap) {
        auto dealloc = builder.create<gpu::DeallocOp>(
            location, waitToken.getType(), waitToken, cpuGpuMemrefPairs.second);
        waitToken = dealloc.getAsyncToken();
      }

      // wait for final dealloc.
      builder.create<gpu::WaitOp>(location, Type(), waitToken);

      ReplaceCPUReferencesWithGPUMemref(cpuToGpuMemrefMap);
      addGPUKernelTimingCalls(gpuLaunchOp, builder);
    }
    return mlir::WalkResult::advance();
  });
}

struct MakeGPULoopsPerfectlyNestedPass
    : public PassWrapper<MakeGPULoopsPerfectlyNestedPass,
                         OperationPass<mlir::func::FuncOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect>();
  }

  void makeGPULoopsPerfectlyNested() {
    auto func = this->getOperation();
    func.walk([&](scf::ParallelOp parallelOp) {
      // If this loop is a batch loop and has a parent loop that
      // is also a batch loop, then move all operations in between
      // the two loops into the child loop
      auto parentParallelOp = parallelOp->getParentOfType<scf::ParallelOp>();
      if (!parentParallelOp)
        return;
      auto parentLoopMappingAttr =
          decisionforest::getMappingAttr(parentParallelOp);
      if (!parentLoopMappingAttr)
        return;
      auto loopMappingAttr = decisionforest::getMappingAttr(parallelOp);
      if (!loopMappingAttr)
        return;
      if (!decisionforest::isThreadBlockLoop(parallelOp) ||
          !decisionforest::isThreadBlockLoop(parentParallelOp))
        return;
      // Move all the ops in the parent loop that are before the child
      // loop into the child loop
      auto parentLoopBody = parentParallelOp.getBody();
      auto &parentLoopBodyOps = parentLoopBody->getOperations();
      auto loopBody = parallelOp.getBody();
      auto &firstOp = loopBody->front();
      for (auto &op : parentLoopBodyOps) {
        if (&op == parallelOp.getOperation())
          break;
        op.moveBefore(&firstOp);
      }
    });
  }

  void runOnOperation() final { makeGPULoopsPerfectlyNested(); }
};

} // anonymous namespace

namespace mlir {
namespace decisionforest {

bool measureGpuKernelTime = false;

gpu::ParallelLoopDimMappingAttr getMappingAttr(scf::ParallelOp parallelOp) {
  auto mappingAttr = parallelOp->getAttr(getMappingAttrName());
  if (!mappingAttr)
    return nullptr;
  auto mappingArrayAttr = mappingAttr.cast<ArrayAttr>();
  return mappingArrayAttr[0].cast<gpu::ParallelLoopDimMappingAttr>();
}

bool isThreadBlockLoop(scf::ParallelOp parallelOp) {
  auto mappingAttr = getMappingAttr(parallelOp);
  if (!mappingAttr)
    return false;
  return mappingAttr.getProcessor() == gpu::Processor::BlockX ||
         mappingAttr.getProcessor() == gpu::Processor::BlockY ||
         mappingAttr.getProcessor() == gpu::Processor::BlockZ;
}

bool isThreadLoop(scf::ParallelOp parallelOp) {
  auto mappingAttr = getMappingAttr(parallelOp);
  if (!mappingAttr)
    return false;
  return mappingAttr.getProcessor() == gpu::Processor::ThreadX ||
         mappingAttr.getProcessor() == gpu::Processor::ThreadY ||
         mappingAttr.getProcessor() == gpu::Processor::ThreadZ;
}

void GreedilyMapParallelLoopsToGPU(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(createGpuMapParallelLoopsPass());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Map parallel loops to GPU failed.\n";
  }
}

void ConvertParallelLoopsToGPU(mlir::MLIRContext &context,
                               mlir::ModuleOp module) {
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  // optPM.addPass(std::make_unique<MakeGPULoopsPerfectlyNestedPass>());
  optPM.addPass(createParallelLoopToGpuPass());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Convert parallel loops to GPULaunchOp pass failed.\n";
  }
  // Insert required GPU allocations and transfers to and from the gpu memory.
  // Replace the uses of the function arguments with the new gpu allocations.
  AddGPUAllocationsAndTransfers(module);
}

void OutlineGPUKernels(mlir::MLIRContext &context, mlir::ModuleOp module) {
  mlir::PassManager pm(&context);
  pm.addPass(createGpuKernelOutliningPass());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Running GPU Outlining pass failed.\n";
  }
}

void RunCanonicalizerPass(mlir::MLIRContext &context, mlir::ModuleOp module) {
  mlir::PassManager pm(&context);
  mlir::GreedyRewriteConfig config;
  std::vector<std::string> disabledPatterns = {
      "(anonymous namespace)::MergeNestedParallelLoops"};
  pm.addPass(createCanonicalizerPass(config, disabledPatterns));

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Canonicalizer pass failed.\n";
  }
}

} // namespace decisionforest
} // namespace mlir

#endif // TREEBEARD_GPU_SUPPORT