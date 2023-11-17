#ifndef _COMPILEUTILS_H_
#define _COMPILEUTILS_H_

#include "Dialect.h"
#include "GPUCompileUtils.h"
#include "LowerReduceOps.h"
#include "TreebeardContext.h"
#include "forestcreator.h"
#include "xgboostparser.h"

namespace TreeBeard {
inline mlir::ModuleOp BuildHIRModule(TreebeardContext &tbContext,
                                     ForestCreator &forestCreator) {
  const CompilerOptions &options = tbContext.options;

  forestCreator.ConstructForest();
  forestCreator.SetChildIndexBitWidth(options.childIndexBitWidth);
  auto module = forestCreator.GetEvaluationFunction();

  return module;
}

inline void DoTilingTransformation(mlir::ModuleOp module,
                                   TreebeardContext &tbContext) {
  const CompilerOptions &options = tbContext.options;
  auto &context = tbContext.context;
  if (options.tileSize == 1) {
    if (options.makeAllLeavesSameDepth)
      mlir::decisionforest::padTreesToMakeAllLeavesSameDepth(context, module);
    return;
  }

  // TODO maybe all the manipulation before the lowering to mid-level IR can be
  // a single custom function?
  if (options.tilingType == TilingType::kUniform)
    mlir::decisionforest::DoUniformTiling(context, module, options.tileSize,
                                          options.tileShapeBitWidth,
                                          options.makeAllLeavesSameDepth);
  else if (options.tilingType == TilingType::kHybrid)
    mlir::decisionforest::DoProbabilityBasedTiling(
        context, module, options.tileSize, options.tileShapeBitWidth);
  else if (options.tilingType == TilingType::kHybrid)
    mlir::decisionforest::DoHybridTiling(context, module, options.tileSize,
                                         options.tileShapeBitWidth);
  else
    assert(false && "Unknown tiling type");
}

inline void LowerHIRModuleToLLVM(mlir::ModuleOp module,
                                 TreebeardContext &tbContext) {
  const CompilerOptions &options = tbContext.options;
  auto &context = tbContext.context;

  // TODO this needs to change to something that knows how to do all schedule
  // manipulation
  if (options.reorderTreesByDepth) {
    assert(options.pipelineSize == -1 ||
           (options.pipelineSize <= options.batchSize));
    mlir::decisionforest::DoReorderTreesByDepth(
        context, module, options.pipelineSize, options.numberOfCores,
        options.numParallelTreeBatches);
    assert(!options.scheduleManipulator &&
           "Cannot have a custom schedule manipulator and the inbuilt one "
           "together");
  }
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(context, module);
  // module->dump();
  mlir::decisionforest::legalizeReductionsAndCanonicalize(context, module);
  // module->dump();

  mlir::decisionforest::lowerReductionsAndCanonicalize(context, module);
  // module->dump();
  // return;

  mlir::decisionforest::LowerEnsembleToMemrefs(
      context, module, tbContext.serializer, tbContext.representation);
  mlir::decisionforest::ConvertNodeTypeToIndexType(context, module);
  // module->dump();
  mlir::decisionforest::LowerToLLVM(context, module, tbContext.representation);
  // mlir::decisionforest::dumpLLVMIR(module, false);
}

inline mlir::ModuleOp
ConstructLLVMDialectModuleFromForestCreator(TreebeardContext &tbContext,
                                            ForestCreator &forestCreator) {

  const CompilerOptions &options = tbContext.options;

  auto module = BuildHIRModule(tbContext, forestCreator);
  DoTilingTransformation(module, tbContext);

  if (options.scheduleManipulator) {
    auto schedule = forestCreator.GetSchedule();
    options.scheduleManipulator->Run(schedule);
    assert(!options.reorderTreesByDepth &&
           "Cannot have a custom schedule manipulator and the inbuilt one "
           "together");
  }
  LowerHIRModuleToLLVM(module, tbContext);
  return module;
}

template <typename ThresholdType = double, typename ReturnType = double,
          typename FeatureIndexType = int32_t, typename NodeIndexType = int32_t,
          typename InputElementType = ThresholdType>
mlir::ModuleOp
ConstructLLVMDialectModuleFromXGBoostJSON(TreebeardContext &tbContext) {
  mlir::MLIRContext &context = tbContext.context;
  const std::string &modelJsonPath = tbContext.modelPath;
  const CompilerOptions &options = tbContext.options;

  // if (options.compileToGPU) {
  //   tbContext.forestConstructor = std::make_shared<
  //       XGBoostJSONParser<ThresholdType, ReturnType, FeatureIndexType,
  //                         NodeIndexType, InputElementType>>(
  //       context, modelJsonPath, tbContext.serializer,
  //       options.statsProfileCSVPath, options.batchSize);
  //   return ConstructGPUModuleFromTreebeardContext(tbContext);
  // }

  TreeBeard::XGBoostJSONParser<ThresholdType, ReturnType, FeatureIndexType,
                               NodeIndexType, InputElementType>
      xgBoostParser(context, modelJsonPath, tbContext.serializer,
                    options.statsProfileCSVPath, options.batchSize);
  return ConstructLLVMDialectModuleFromForestCreator(tbContext, xgBoostParser);
}

void InitializeMLIRContext(mlir::MLIRContext &context);

mlir::ModuleOp
ConstructLLVMDialectModuleFromXGBoostJSON(TreebeardContext &tbContext);
void ConvertONNXModelToLLVMIR(TreebeardContext &tbContext,
                              const std::string &llvmIRFilePath);
void ConvertXGBoostJSONToLLVMIR(TreebeardContext &tbContext,
                                const std::string &llvmIRFilePath);

void RunInferenceUsingSO(const std::string &soPath,
                         const std::string &modelGlobalsJSONPath,
                         const std::string &csvPath,
                         const CompilerOptions &options);
} // namespace TreeBeard

#endif // _COMPILEUTILS_H_