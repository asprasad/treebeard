#ifdef TREEBEARD_GPU_SUPPORT

#ifndef _GPUSUPPORTUTILS_H_
#define _GPUSUPPORTUTILS_H_

#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"

namespace mlir {
namespace decisionforest {

gpu::ParallelLoopDimMappingAttr getMappingAttr(scf::ParallelOp parallelOp);
bool isThreadBlockLoop(scf::ParallelOp parallelOp);
bool isThreadLoop(scf::ParallelOp parallelOp);

void RunCanonicalizerPass(mlir::MLIRContext &context, mlir::ModuleOp module);
void GreedilyMapParallelLoopsToGPU(mlir::ModuleOp module);
void ConvertParallelLoopsToGPU(mlir::MLIRContext &context,
                               mlir::ModuleOp module);
void OutlineGPUKernels(mlir::MLIRContext &context, mlir::ModuleOp module);
void LowerGPUToLLVM(
    mlir::MLIRContext &context, mlir::ModuleOp module,
    std::shared_ptr<decisionforest::IRepresentation> representation);
} // namespace decisionforest
} // namespace mlir

#endif // _GPUSUPPORTUTILS_H_

#endif // TREEBEARD_GPU_SUPPORT