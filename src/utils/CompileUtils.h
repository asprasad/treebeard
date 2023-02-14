#ifndef _COMPILEUTILS_H_
#define _COMPILEUTILS_H_

#include "Dialect.h"
#include "xgboostparser.h"
#include "TreebeardContext.h"

namespace TreeBeard
{

template<typename ThresholdType=double, typename ReturnType=double, typename FeatureIndexType=int32_t, 
         typename NodeIndexType=int32_t, typename InputElementType=ThresholdType>
mlir::ModuleOp ConstructLLVMDialectModuleFromXGBoostJSON(mlir::MLIRContext& context, TreebeardContext& tbContext) {
  const std::string& modelJsonPath=tbContext.modelJSONPath;
  const CompilerOptions& options=tbContext.options;
  
  TreeBeard::XGBoostJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType>
                               xgBoostParser(context, modelJsonPath, tbContext.serializer, options.statsProfileCSVPath, options.batchSize);
  xgBoostParser.Parse();
  xgBoostParser.SetChildIndexBitWidth(options.childIndexBitWidth);
  auto module = xgBoostParser.GetEvaluationFunction();

  // TODO maybe all the manipulation before the lowering to mid-level IR can be a single custom function?
  if (options.tilingType==TilingType::kUniform)
    mlir::decisionforest::DoUniformTiling(context, module, options.tileSize, options.tileShapeBitWidth, options.makeAllLeavesSameDepth);
  else if (options.tilingType==TilingType::kHybrid)
    mlir::decisionforest::DoProbabilityBasedTiling(context, module, options.tileSize, options.tileShapeBitWidth);
  else if (options.tilingType==TilingType::kHybrid)
    mlir::decisionforest::DoHybridTiling(context, module, options.tileSize, options.tileShapeBitWidth);
  else
    assert (false && "Unknown tiling type");
  
  if (options.scheduleManipulator) {
    auto schedule = xgBoostParser.GetSchedule();
    options.scheduleManipulator->Run(schedule);
    assert (!options.reorderTreesByDepth && "Cannot have a custom schedule manipulator and the inbuilt one together");
  }

  // TODO this needs to change to something that knows how to do all schedule manipulation
  if (options.reorderTreesByDepth) {
    assert(options.pipelineSize == -1 || (options.pipelineSize <= options.batchSize));
    mlir::decisionforest::DoReorderTreesByDepth(context, module, options.pipelineSize, options.numberOfCores);
    assert (!options.scheduleManipulator && "Cannot have a custom schedule manipulator and the inbuilt one together");
  }
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(context, module);
  mlir::decisionforest::LowerEnsembleToMemrefs(context, module, tbContext.serializer, tbContext.representation);
  mlir::decisionforest::ConvertNodeTypeToIndexType(context, module);
  // module->dump();
  mlir::decisionforest::LowerToLLVM(context, module, tbContext.representation);
  // mlir::decisionforest::dumpLLVMIR(module, false);
  return module;
}

void InitializeMLIRContext(mlir::MLIRContext& context);

mlir::ModuleOp ConstructLLVMDialectModuleFromXGBoostJSON(mlir::MLIRContext& context, TreebeardContext& tbContext);
void ConvertXGBoostJSONToLLVMIR(TreebeardContext& tbContext, const std::string& llvmIRFilePath);

void RunInferenceUsingSO(const std::string&modelJsonPath, const std::string& soPath, const std::string& modelGlobalsJSONPath, 
                         const std::string& csvPath, const CompilerOptions& options);
}


#endif // _COMPILEUTILS_H_