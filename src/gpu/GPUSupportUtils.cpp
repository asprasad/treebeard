#include "Dialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "GPUSupportUtils.h"

namespace mlir
{
namespace decisionforest
{

void GreedilyMapParallelLoopsToGPU(mlir::ModuleOp module) {
  for (Region &region : module->getRegions())
    greedilyMapParallelSCFToGPU(region);
}

void ConvertParallelLoopsToGPU(mlir::MLIRContext& context, mlir::ModuleOp module) {
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(createParallelLoopToGpuPass());

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Lowering to mid level IR failed.\n";
  }
}

} // decisionforest
} // mlir