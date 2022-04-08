#ifndef _COMPILEUTILS_H_
#define _COMPILEUTILS_H_

#include "Dialect.h"
#include "xgboostparser.h"

namespace TreeBeard
{

enum class TilingType { kUniform, kProbabilistic, kHybrid };

struct CompilerOptions {
  // optimization parameters
  int32_t batchSize;
  int32_t tileSize;

  // Type size parameters
  int32_t thresholdTypeWidth=32;
  int32_t returnTypeWidth=32;
  bool returnTypeFloatType=true;
  int32_t featureIndexTypeWidth=16;
  int32_t nodeIndexTypeWidth=16;
  int32_t inputElementTypeWidth=32;
  int32_t tileShapeBitWidth=16;
  int32_t childIndexBitWidth=16;
  TilingType tilingType=TilingType::kUniform;
  bool makeAllLeavesSameDepth=false;
  bool reorderTreesByDepth=false;

  mlir::decisionforest::ScheduleManipulator *scheduleManipulator=nullptr;
  std::string statsProfileCSVPath = "";
  int32_t pipelineWidth = -1;
  int32_t numberOfCores = -1;

  CompilerOptions() { }
  CompilerOptions(int32_t thresholdWidth, int32_t returnWidth, bool isReturnTypeFloat, int32_t featureIndexWidth, 
                  int32_t nodeIndexWidth, int32_t inputElementWidth, int32_t batchSz, int32_t tileSz,
                  int32_t tileShapeWidth, int32_t childIndexWidth, TilingType tileType, bool makeLeavesSameDepth, bool reorderTrees,
                  mlir::decisionforest::ScheduleManipulator* scheduleManip)
  : batchSize(batchSz), tileSize(tileSz), thresholdTypeWidth(thresholdWidth), returnTypeWidth(returnWidth), returnTypeFloatType(isReturnTypeFloat),
    featureIndexTypeWidth(featureIndexWidth), nodeIndexTypeWidth(nodeIndexWidth), inputElementTypeWidth(inputElementWidth),
    tileShapeBitWidth(tileShapeWidth), childIndexBitWidth(childIndexWidth), tilingType(tileType), makeAllLeavesSameDepth(makeLeavesSameDepth),
    reorderTreesByDepth(reorderTrees), scheduleManipulator(scheduleManip)
  { }
};

template<typename ThresholdType=double, typename ReturnType=double, typename FeatureIndexType=int32_t, 
         typename NodeIndexType=int32_t, typename InputElementType=ThresholdType>
mlir::ModuleOp ConstructLLVMDialectModuleFromXGBoostJSON(mlir::MLIRContext& context, const std::string& modelJsonPath, 
                                                         const std::string& modelGlobalsJSONPath, const CompilerOptions& options) {
  TreeBeard::XGBoostJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType>
                               xgBoostParser(context, modelJsonPath, modelGlobalsJSONPath, options.statsProfileCSVPath, options.batchSize);
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
    mlir::decisionforest::DoReorderTreesByDepth(context, module, options.pipelineWidth, options.numberOfCores);
    assert (!options.scheduleManipulator && "Cannot have a custom schedule manipulator and the inbuilt one together");
  }
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(context, module);
  mlir::decisionforest::LowerEnsembleToMemrefs(context, module);
  mlir::decisionforest::ConvertNodeTypeToIndexType(context, module);
  // module->dump();
  mlir::decisionforest::LowerToLLVM(context, module);
  // mlir::decisionforest::dumpLLVMIR(module, false);
  return module;
}

mlir::ModuleOp ConstructLLVMDialectModuleFromXGBoostJSON(mlir::MLIRContext& context, const std::string&modelJsonPath, 
                                                         const std::string& modelGlobalsJSONPath, const CompilerOptions& options);

void InitializeMLIRContext(mlir::MLIRContext& context);
void ConvertXGBoostJSONToLLVMIR(const std::string&modelJsonPath, const std::string& llvmIRFilePath, const std::string& modelGlobalsJSONPath, const CompilerOptions& options);

void RunInferenceUsingSO(const std::string&modelJsonPath, const std::string& soPath, const std::string& modelGlobalsJSONPath, 
                         const std::string& csvPath, const CompilerOptions& options);
}


#endif // _COMPILEUTILS_H_