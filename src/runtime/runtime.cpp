#include <iostream>
#include <fstream>
#include <json.hpp>
#include "ExecutionHelpers.h"

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
  auto inferenceRunner = new mlir::decisionforest::SharedObjectInferenceRunner(modelGlobalsJSONPath, soPath, tileSize, 
                                                                               thresholdBitwidth, featureIndexBitwidth);
  return reinterpret_cast<intptr_t>(inferenceRunner);
}

// Run inference
//    -- inference runner, row, result
extern "C" void RunInference(intptr_t inferenceRunnerInt, void *inputs, void *results) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::SharedObjectInferenceRunner*>(inferenceRunnerInt);
  // TODO The types in this template don't really matter. Maybe we should get rid of them? 
  inferenceRunner->RunInference<double, double>(reinterpret_cast<double*>(inputs), reinterpret_cast<double*>(results));
}

// TODO We're assuming float as the type of the inputs and outputs, but we should change this based on values persisted in the JSON 
extern "C" void RunInferenceOnMultipleBatches(intptr_t inferenceRunnerInt, void *inputs, void *results, int32_t numRows) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::SharedObjectInferenceRunner*>(inferenceRunnerInt);
  auto batchSize = inferenceRunner->GetBatchSize();
  auto rowSize = inferenceRunner->GetRowSize();

  assert (numRows % batchSize == 0);
  for (int32_t batch=0 ; batch<numRows/batchSize ; ++batch) {
    auto batchPtr = reinterpret_cast<float*>(inputs) + batch * (rowSize*batchSize);
    auto resultsPtr = reinterpret_cast<float*>(results) + batch * batchSize;
    inferenceRunner->RunInference<float, float>(batchPtr, resultsPtr);
  }
}

extern "C" int32_t GetBatchSize(intptr_t inferenceRunnerInt) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::SharedObjectInferenceRunner*>(inferenceRunnerInt);
  // TODO The types in this template don't really matter. Maybe we should get rid of them? 
  return inferenceRunner->GetBatchSize();
}

extern "C" int32_t GetRowSize(intptr_t inferenceRunnerInt) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::SharedObjectInferenceRunner*>(inferenceRunnerInt);
  // TODO The types in this template don't really matter. Maybe we should get rid of them? 
  return inferenceRunner->GetRowSize();
}

extern "C" void DeleteInferenceRunner(intptr_t inferenceRunnerInt) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::SharedObjectInferenceRunner*>(inferenceRunnerInt);
  delete inferenceRunner;
}
