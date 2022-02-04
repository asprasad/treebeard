#ifndef _COMPILEUTILS_H_
#define _COMPILEUTILS_H_

#include "Dialect.h"
#include "xgboostparser.h"

namespace TreeBeard
{

struct CompilerOptions {
  // optimization parameters
  int32_t batchSize;
  int32_t tileSize;

  // Type size parameters
  int32_t thresholdTypeWidth;
  int32_t returnTypeWidth;
  int32_t featureIndexTypeWidth;
  int32_t nodeIndexTypeWidth;
  int32_t inputElementTypeWidth;
  int32_t tileShapeBitWidth;
  int32_t childIndexBitWidth;

  mlir::decisionforest::ScheduleManipulator *scheduleManipulator;

  CompilerOptions(int32_t thresholdWidth, int32_t returnWidth, int32_t featureIndexWidth, 
                  int32_t nodeIndexWidth, int32_t inputElementWidth, int32_t batchSz, int32_t tileSz,
                  int32_t tileShapeWidth, int32_t childIndexWidth, mlir::decisionforest::ScheduleManipulator* scheduleManip)
  : batchSize(batchSz), tileSize(tileSz), thresholdTypeWidth(thresholdWidth), returnTypeWidth(returnWidth),
    featureIndexTypeWidth(featureIndexWidth), nodeIndexTypeWidth(nodeIndexWidth), inputElementTypeWidth(inputElementWidth),
    tileShapeBitWidth(tileShapeWidth), childIndexBitWidth(childIndexWidth), scheduleManipulator(scheduleManip)
  { }
};

template<typename ThresholdType=double, typename ReturnType=double, typename FeatureIndexType=int32_t, 
         typename NodeIndexType=int32_t, typename InputElementType=double>
mlir::ModuleOp ConstructLLVMDialectModuleFromXGBoostJSON(mlir::MLIRContext& context, const std::string& modelJsonPath, 
                                                         const std::string& modelGlobalsJSONPath, const CompilerOptions& options) {
  TreeBeard::XGBoostJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType> xgBoostParser(context, modelJsonPath, modelGlobalsJSONPath, options.batchSize);
  xgBoostParser.Parse();
  xgBoostParser.SetChildIndexBitWidth(options.childIndexBitWidth);
  auto module = xgBoostParser.GetEvaluationFunction();

  if (options.scheduleManipulator) {
    auto schedule = xgBoostParser.GetSchedule();
    options.scheduleManipulator->Run(schedule);
  }

  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(context, module);
  mlir::decisionforest::DoUniformTiling(context, module, options.tileSize, options.tileShapeBitWidth);
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