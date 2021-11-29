#ifndef _COMPILEUTILS_H_
#define _COMPILEUTILS_H_

#include "Dialect.h"
#include "xgboostparser.h"

namespace TreeBeard
{

template<typename ThresholdType=double, typename ReturnType=double, typename FeatureIndexType=int32_t, 
         typename NodeIndexType=int32_t, typename InputElementType=double>
mlir::ModuleOp ConstructLLVMDialectModuleFromXGBoostJSON(mlir::MLIRContext& context, const std::string& modelJsonPath, int32_t batchSize, int32_t tileSize=1) {
  TreeBeard::XGBoostJSONParser<ThresholdType, ReturnType, FeatureIndexType, NodeIndexType, InputElementType> xgBoostParser(context, modelJsonPath, batchSize);
  xgBoostParser.Parse();
  auto module = xgBoostParser.GetEvaluationFunction();
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(context, module);
  mlir::decisionforest::DoUniformTiling(context, module, tileSize);
  mlir::decisionforest::LowerEnsembleToMemrefs(context, module);
  mlir::decisionforest::ConvertNodeTypeToIndexType(context, module);
  // module->dump();
  mlir::decisionforest::LowerToLLVM(context, module);
  // mlir::decisionforest::dumpLLVMIR(module, false);
  return module;
}

mlir::ModuleOp ConstructLLVMDialectModuleFromXGBoostJSON(mlir::MLIRContext& context, const std::string&modelJsonPath,
                                                         int32_t thresholdTypeWidth, int32_t returnTypeWidth, int32_t featureIndexTypeWidth, 
                                                         int32_t nodeIndexTypeWidth, int32_t inputElementTypeWidth, int32_t batchSize, int32_t tileSize);

void InitializeMLIRContext(mlir::MLIRContext& context);
void ConvertXGBoostJSONToLLVMIR(const std::string&modelJsonPath, const std::string& llvmIRFilePath,
                                int32_t thresholdTypeWidth, int32_t returnTypeWidth, int32_t featureIndexTypeWidth, 
                                int32_t nodeIndexTypeWidth, int32_t inputElementTypeWidth, int32_t batchSize, int32_t tileSize);

void RunInferenceUsingSO(const std::string&modelJsonPath, const std::string& soPath, const std::string& csvPath,
                         int32_t thresholdTypeWidth, int32_t returnTypeWidth, int32_t featureIndexTypeWidth, 
                         int32_t nodeIndexTypeWidth, int32_t inputElementTypeWidth, int32_t batchSize, int32_t tileSize);
}


#endif // _COMPILEUTILS_H_