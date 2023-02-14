#include <iostream>
#include <fstream>
#include <json.hpp>
#include "ExecutionHelpers.h"
#include "CompileUtils.h"
#include "xgboostparser.h"
#include "schedule.h"
#include "ModelSerializers.h"
#include "Representations.h"

// ===-------------------------------------------------------------=== //
// Execution API
// ===-------------------------------------------------------------=== //

// Create a shared object inference runner and return an ID (Init)
//    -- SO name, globals JSON path 
extern "C" intptr_t InitializeInferenceRunner(const char* soPath, const char* modelGlobalsJSONPath) {
  using json = nlohmann::json;
  json globalsJSON;
  std::ifstream fin(modelGlobalsJSONPath);
  fin >> globalsJSON;

  auto tileSizeEntries = globalsJSON["TileSizeEntries"];
  assert (tileSizeEntries.size() == 1);
  int32_t tileSize = tileSizeEntries.front()["TileSize"];
  int32_t thresholdBitwidth = tileSizeEntries.front()["ThresholdBitWidth"];
  int32_t featureIndexBitwidth = tileSizeEntries.front()["FeatureIndexBitWidth"];
  auto serializer = mlir::decisionforest::ConstructModelSerializer(modelGlobalsJSONPath);
  auto inferenceRunner = new mlir::decisionforest::SharedObjectInferenceRunner(serializer, soPath, tileSize, 
                                                                               thresholdBitwidth, featureIndexBitwidth);
  return reinterpret_cast<intptr_t>(inferenceRunner);
}

// Run inference
//    -- inference runner, row, result
extern "C" void RunInference(intptr_t inferenceRunnerInt, void *inputs, void *results) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::InferenceRunnerBase*>(inferenceRunnerInt);
  // TODO The types in this template don't really matter. Maybe we should get rid of them? 
  inferenceRunner->RunInference<double, double>(reinterpret_cast<double*>(inputs), reinterpret_cast<double*>(results));
}

extern "C" void RunInferenceOnMultipleBatches(intptr_t inferenceRunnerInt, void *inputs, void *results, int32_t numRows) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::InferenceRunnerBase*>(inferenceRunnerInt);
  auto batchSize = inferenceRunner->GetBatchSize();
  auto rowSize = inferenceRunner->GetRowSize();

  assert (numRows % batchSize == 0);
  int32_t inputElementSize = inferenceRunner->GetInputElementBitWidth()/8;
  int32_t returnTypeSize = inferenceRunner->GetReturnTypeBitWidth()/8;
  for (int32_t batch=0 ; batch<numRows/batchSize ; ++batch) {
    auto batchPtr = reinterpret_cast<char*>(inputs) + (batch * (rowSize*batchSize) * inputElementSize);
    auto resultsPtr = reinterpret_cast<char*>(results) + (batch * batchSize * returnTypeSize);
    // TODO The types in this template don't really matter. Maybe we should get rid of them? 
    inferenceRunner->RunInference<double, double>(reinterpret_cast<double*>(batchPtr), reinterpret_cast<double*>(resultsPtr));
  }
}

extern "C" int32_t GetBatchSize(intptr_t inferenceRunnerInt) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::InferenceRunnerBase*>(inferenceRunnerInt);
  // TODO The types in this template don't really matter. Maybe we should get rid of them? 
  return inferenceRunner->GetBatchSize();
}

extern "C" int32_t GetRowSize(intptr_t inferenceRunnerInt) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::InferenceRunnerBase*>(inferenceRunnerInt);
  // TODO The types in this template don't really matter. Maybe we should get rid of them? 
  return inferenceRunner->GetRowSize();
}

extern "C" void DeleteInferenceRunner(intptr_t inferenceRunnerInt) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::InferenceRunnerBase*>(inferenceRunnerInt);
  delete inferenceRunner;
}

// ===-------------------------------------------------------------=== //
// Compilation API
// ===-------------------------------------------------------------=== //

extern "C" intptr_t CreateCompilerOptions() {
  return reinterpret_cast<intptr_t>(new TreeBeard::CompilerOptions);
}

extern "C" void DeleteCompilerOptions(intptr_t options) {
  TreeBeard::CompilerOptions *optionsPtr = reinterpret_cast<TreeBeard::CompilerOptions*>(options);
  delete optionsPtr->scheduleManipulator;
  delete optionsPtr;
}

#define COMPILER_OPTION_SETTER(propName, propType) \
extern "C" void Set_##propName(intptr_t options, propType val) { \
  TreeBeard::CompilerOptions *optionsPtr = reinterpret_cast<TreeBeard::CompilerOptions*>(options);  \
  optionsPtr->propName = reinterpret_cast<propType>(val); \
} 

COMPILER_OPTION_SETTER(batchSize, int32_t)
COMPILER_OPTION_SETTER(tileSize, int32_t)

COMPILER_OPTION_SETTER(thresholdTypeWidth, int32_t)
COMPILER_OPTION_SETTER(returnTypeWidth, int32_t)
COMPILER_OPTION_SETTER(returnTypeFloatType, int32_t)
COMPILER_OPTION_SETTER(featureIndexTypeWidth, int32_t)
COMPILER_OPTION_SETTER(nodeIndexTypeWidth, int32_t)
COMPILER_OPTION_SETTER(inputElementTypeWidth, int32_t)
COMPILER_OPTION_SETTER(tileShapeBitWidth, int32_t)
COMPILER_OPTION_SETTER(childIndexBitWidth, int32_t)
COMPILER_OPTION_SETTER(makeAllLeavesSameDepth, int32_t)
COMPILER_OPTION_SETTER(reorderTreesByDepth, int32_t)
COMPILER_OPTION_SETTER(statsProfileCSVPath,  const char*)
COMPILER_OPTION_SETTER(pipelineSize, int32_t)
COMPILER_OPTION_SETTER(numberOfCores, int32_t)

extern "C" void Set_tilingType(intptr_t options, int32_t val) {
  TreeBeard::CompilerOptions *optionsPtr = reinterpret_cast<TreeBeard::CompilerOptions*>(options);
  TreeBeard::TilingType tilingType;
  if (val == 0)
    tilingType = TreeBeard::TilingType::kUniform;
  else if (val == 1)
    tilingType = TreeBeard::TilingType::kProbabilistic;
  else if (val == 2)
    tilingType = TreeBeard::TilingType::kHybrid;
  else
    assert (false && "Invalid tiling type value");
  optionsPtr->tilingType = tilingType;
}

extern "C" void GenerateLLVMIRForXGBoostModel(const char* modelJSONPath, const char* llvmIRFilePath,
                                              const char* modelGlobalsJSONPath, intptr_t options) {
  TreeBeard::CompilerOptions *optionsPtr = reinterpret_cast<TreeBeard::CompilerOptions*>(options);
  TreeBeard::TreebeardContext tbContext{modelJSONPath, modelGlobalsJSONPath, *optionsPtr, 
                                        mlir::decisionforest::ConstructRepresentation(),
                                        mlir::decisionforest::ConstructModelSerializer(std::string(modelGlobalsJSONPath))};
  TreeBeard::ConvertXGBoostJSONToLLVMIR(tbContext, llvmIRFilePath);
}

extern "C" intptr_t CreateInferenceRunner(const char* modelJSONPath, const char* profileCSVPath,
                                          intptr_t options) {
  TreeBeard::CompilerOptions *optionsPtr = reinterpret_cast<TreeBeard::CompilerOptions*>(options);
  auto modelGlobalsJSONPath = TreeBeard::XGBoostJSONParser<>::ModelGlobalJSONFilePathFromJSONFilePath(modelJSONPath);
  mlir::MLIRContext context;
  TreeBeard::InitializeMLIRContext(context); 
  TreeBeard::TreebeardContext tbContext{modelJSONPath, modelGlobalsJSONPath, *optionsPtr, 
                                        mlir::decisionforest::ConstructRepresentation(),
                                        mlir::decisionforest::ConstructModelSerializer(modelGlobalsJSONPath)};
  auto module = TreeBeard::ConstructLLVMDialectModuleFromXGBoostJSON(context, tbContext);
  auto inferenceRunner = new mlir::decisionforest::InferenceRunner(tbContext.serializer, module, 
                                                                   optionsPtr->tileSize, optionsPtr->thresholdTypeWidth,
                                                                   optionsPtr->featureIndexTypeWidth);
  return reinterpret_cast<intptr_t>(inferenceRunner);
}

extern "C" void SetEnableSparseRepresentation(int32_t val) {
  mlir::decisionforest::UseSparseTreeRepresentation = val;
}

extern "C" int32_t IsSparseRepresentationEnabled() {
  return mlir::decisionforest::UseSparseTreeRepresentation;
}

extern "C" void SetPeeledCodeGenForProbabilityBasedTiling(int32_t val) {
  mlir::decisionforest::PeeledCodeGenForProbabiltyBasedTiling = val;
}

extern "C" int32_t IsPeeledCodeGenForProbabilityBasedTilingEnabled() {
  return mlir::decisionforest::PeeledCodeGenForProbabiltyBasedTiling;
}

// ===-------------------------------------------------------------=== //
// Predefined Schedule Manipulation API
// ===-------------------------------------------------------------=== //

extern "C" void SetOneTreeAtATimeSchedule(intptr_t options) {
  TreeBeard::CompilerOptions *optionsPtr = reinterpret_cast<TreeBeard::CompilerOptions*>(options);
  optionsPtr->scheduleManipulator = new mlir::decisionforest::ScheduleManipulationFunctionWrapper(mlir::decisionforest::OneTreeAtATimeSchedule);
}
