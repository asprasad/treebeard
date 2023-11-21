#ifdef TREEBEARD_GPU_SUPPORT

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <thread>
#include <vector>

#include "ExecutionHelpers.h"
#include "TiledTree.h"
#include "TreeTilingUtils.h"
#include "forestcreator.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "xgboostparser.h"
#include "llvm/ADT/STLExtras.h"

#include "CompileUtils.h"
#include "ForestTestUtils.h"
#include "GPUCompileUtils.h"
#include "GPUExecutionHelper.h"
#include "GPUModelSerializers.h"
#include "GPUSupportUtils.h"
#include "LowerReduceOps.h"
#include "ModelSerializers.h"
#include "ReorgForestRepresentation.h"
#include "Representations.h"
#include "TestUtilsCommon.h"

namespace TreeBeard {

mlir::ModuleOp LowerHIRModuleToGPU(mlir::ModuleOp module,
                                   TreebeardContext &tbContext) {

  auto &context = tbContext.context;
  auto tileSize = tbContext.options.tileSize;
  auto representation = tbContext.representation;
  auto serializer = tbContext.serializer;

  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(context, module);
  // module->dump();

  mlir::decisionforest::legalizeReductionsAndCanonicalize(context, module);
  // module->dump();

  mlir::decisionforest::convertToCooperativeReduceAndCanonicalize(context,
                                                                  module);
  // module->dump();

  // mlir::decisionforest::GreedilyMapParallelLoopsToGPU(module);

  mlir::decisionforest::ConvertParallelLoopsToGPU(context, module);
  // module->dump();

  mlir::decisionforest::lowerReductionsAndCanonicalize(context, module);
  // module->dump();
  // return module;

  // Commenting since previous pass runs canonicalizer
  // decisionforest::RunCanonicalizerPass(context, module);
  // module->dump();

  if (tileSize > 1)
    decisionforest::ConvertTraverseToSimtTraverse(context, module);
  // module->dump();

  mlir::decisionforest::LowerGPUEnsembleToMemrefs(context, module, serializer,
                                                  representation);
  // module->dump();

  mlir::decisionforest::ConvertNodeTypeToIndexType(context, module);
  // module->dump();

  mlir::decisionforest::LowerGPUToLLVM(context, module, representation,
                                       tbContext.gpuCompileInfo);
  // module->dump();
  // std::cout << "Shared memory used: "
  //           << tbContext.gpuCompileInfo.sharedMemoryInBytes << "\n";
  return module;
}

// ===---------------------------------------------------=== //
// GPU Compilation Helpers
// ===---------------------------------------------------=== //
mlir::ModuleOp
ConstructGPUModuleFromTreebeardContext(TreebeardContext &tbContext) {
  const CompilerOptions &options = tbContext.options;
  auto &forestCreator = *tbContext.forestConstructor;

  // Build the HIR MLIR module from the input file
  auto module = BuildHIRModule(tbContext, forestCreator);
  // module->dump();

  // If tiling is enabled, then use uniform tiling and pad all trees
  assert(tbContext.options.tileSize == 1 ||
         (tbContext.options.tilingType == TreeBeard::TilingType::kUniform &&
          tbContext.options.makeAllLeavesSameDepth));

  DoTilingTransformation(module, tbContext);

  assert(
      !(options.scheduleManipulator && tbContext.shouldUseGPUAutoSchedule()) &&
      "Cannot have a custom schedule manipulator and the inbuilt one together");
  if (options.scheduleManipulator) {
    auto schedule = forestCreator.GetSchedule();
    options.scheduleManipulator->Run(schedule);
  }
  if (tbContext.shouldUseGPUAutoSchedule()) {
    DoGPUAutoSchedule(tbContext.context, module, tbContext.gpuScheduleOptions);
  }
  module = LowerHIRModuleToGPU(module, tbContext);
  return module;
}

} // namespace TreeBeard
#endif // TREEBEARD_GPU_SUPPORT