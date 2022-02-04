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

extern "C" void DeleteInferenceRunner(intptr_t inferenceRunnerInt) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::SharedObjectInferenceRunner*>(inferenceRunnerInt);
  delete inferenceRunner;
}