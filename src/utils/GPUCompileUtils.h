#ifndef _GPUCOMPILEUTILS_H_
#define _GPUCOMPILEUTILS_H_

#include "Dialect.h"
#include "TreebeardContext.h"
#include "forestcreator.h"

namespace TreeBeard {

// Lower a HIR module to a GPU module
mlir::ModuleOp LowerHIRModuleToGPU(mlir::ModuleOp module,
                                   TreebeardContext &tbContext);

// Construct a GPU module from a TreebeardContext
mlir::ModuleOp
ConstructGPUModuleFromTreebeardContext(TreebeardContext &tbContext);

void DoGPUAutoSchedule(mlir::MLIRContext &context, mlir::ModuleOp module,
                       const TreeBeard::GPUAutoScheduleOptions &options);

struct GPUAutoSchedulerResults {
  TreeBeard::GPUAutoScheduleOptions gpuAutoSchedulingOptions;
  bool unrollTreeWalks;
  std::string representation;
};

GPUAutoSchedulerResults findBestGPUSchedule(const std::string &benchmark,
                                            int32_t batchSize);

} // namespace TreeBeard

#endif // _GPUCOMPILEUTILS_H_