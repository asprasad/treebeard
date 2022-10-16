import os
import numpy
import pandas
import time
import treebeard

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(filepath)))

def CheckEqual(a, b) -> bool:
  if type(a) is numpy.float32:
    threshold = 1e-6
    return pow(a-b, 2) < threshold
  else:
    return a==b

def CheckArraysEqual(x, y) -> bool:
  assert len(x.shape) == 1
  assert len(y.shape) == 1
  if not x.shape[0] == y.shape[0]:
    return False
  for i in range(x.shape[0]):
    if not CheckEqual(x[i], y[i]):
      return False
  return True

def RunSingleTestJIT(modelJSONPath, csvPath, options, returnType) -> bool:
  data_df = pandas.read_csv(csvPath, header=None)
  data = numpy.array(data_df, order='C') # numpy.genfromtxt(csvPath, ',')
  inputs = numpy.array(data[:, :-1], numpy.float32, order='C')
  expectedOutputs = data[:, data.shape[1]-1]
  
  inferenceRunner = treebeard.TreebeardInferenceRunner.FromModelFile(modelJSONPath, "", options)
  start = time.time()
  for i in range(0, data.shape[0], 200):
    batch = inputs[i:i+200, :]
    results = inferenceRunner.RunInference(batch, returnType)
    if not CheckArraysEqual(results, expectedOutputs[i:i+200]):
      print("Failed")
      return False
  end = time.time()
  print("Passed (", end - start, "s )")
  return True

def RunTestOnSingleModelTestInputsJIT(modelName : str, options, testName : str, returnType=numpy.float32) -> bool:
  print("Running JIT test", testName, modelName, "...", end=" ")
  modelJSONPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  return RunSingleTestJIT(modelJSONPath, csvPath, options, returnType)

def SetStatsProfileCSVPath(options, modelName):
  profilesDir = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), "profiles") 
  csvPath = os.path.join(profilesDir, modelName + ".test.csv")
  options.SetStatsProfileCSVPath(csvPath)
  return options

def RunPipeliningTests():
  invertLoopsTileSize8Options = treebeard.CompilerOptions(200, 8)
  # invertLoopsTileSize8Options.SetOneTreeAtATimeSchedule()
  invertLoopsTileSize8Options.SetPipelineWidth(8)
  invertLoopsTileSize8Options.SetReorderTreesByDepth(True)
  invertLoopsTileSize8Options.SetMakeAllLeavesSameDepth(1)
  
  invertLoopsTileSize8MulticlassOptions = treebeard.CompilerOptions(200, 8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)
  # invertLoopsTileSize8MulticlassOptions.SetOneTreeAtATimeSchedule();
  invertLoopsTileSize8MulticlassOptions.SetPipelineWidth(8)
  invertLoopsTileSize8MulticlassOptions.SetReorderTreesByDepth(True)
  invertLoopsTileSize8MulticlassOptions.SetMakeAllLeavesSameDepth(1)

  assert RunTestOnSingleModelTestInputsJIT("abalone", invertLoopsTileSize8Options, "one-tree-pipeline8-sparse")
  assert RunTestOnSingleModelTestInputsJIT("airline", invertLoopsTileSize8Options, "one-tree-pipeline8-sparse")
  assert RunTestOnSingleModelTestInputsJIT("airline-ohe", invertLoopsTileSize8Options, "one-tree-pipeline8-sparse")
  assert RunTestOnSingleModelTestInputsJIT("covtype", invertLoopsTileSize8MulticlassOptions, "one-tree-pipeline8-sparse", numpy.int8)
  assert RunTestOnSingleModelTestInputsJIT("epsilon", invertLoopsTileSize8Options, "one-tree-pipeline8-sparse")
  assert RunTestOnSingleModelTestInputsJIT("higgs", invertLoopsTileSize8Options, "one-tree-pipeline8-sparse")
  assert RunTestOnSingleModelTestInputsJIT("letters", invertLoopsTileSize8MulticlassOptions, "one-tree-pipeline8-sparse", numpy.int8)
  assert RunTestOnSingleModelTestInputsJIT("year_prediction_msd", invertLoopsTileSize8Options, "one-tree-pipeline8-sparse")

def RunParallelTests():
  batchSize = 200
  num_cores = 4
  tile_size = 8
  pipeline_width = 8
  invertLoopsTileSize8Options = treebeard.CompilerOptions(batchSize, tile_size)
  invertLoopsTileSize8Options.SetPipelineWidth(pipeline_width)
  invertLoopsTileSize8Options.SetReorderTreesByDepth(True)
  invertLoopsTileSize8Options.SetMakeAllLeavesSameDepth(1)
  invertLoopsTileSize8Options.SetNumberOfCores(num_cores)

  invertLoopsTileSize8MulticlassOptions = treebeard.CompilerOptions(batchSize, tile_size)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)
  invertLoopsTileSize8MulticlassOptions.SetPipelineWidth(pipeline_width)
  invertLoopsTileSize8MulticlassOptions.SetReorderTreesByDepth(True)
  invertLoopsTileSize8MulticlassOptions.SetMakeAllLeavesSameDepth(1)
  invertLoopsTileSize8MulticlassOptions.SetNumberOfCores(num_cores)

  assert RunTestOnSingleModelTestInputsJIT("abalone", invertLoopsTileSize8Options, "one-tree-par-4cores")
  assert RunTestOnSingleModelTestInputsJIT("airline", invertLoopsTileSize8Options, "one-tree-par-4cores")
  assert RunTestOnSingleModelTestInputsJIT("airline-ohe", invertLoopsTileSize8Options, "one-tree-par-4cores")
  assert RunTestOnSingleModelTestInputsJIT("covtype", invertLoopsTileSize8MulticlassOptions, "one-tree-par-4cores", numpy.int8)
  assert RunTestOnSingleModelTestInputsJIT("epsilon", invertLoopsTileSize8Options, "one-tree-par-4cores")
  assert RunTestOnSingleModelTestInputsJIT("higgs", invertLoopsTileSize8Options, "one-tree-par-4cores")
  assert RunTestOnSingleModelTestInputsJIT("letters", invertLoopsTileSize8MulticlassOptions, "one-tree-par-4cores", numpy.int8)
  assert RunTestOnSingleModelTestInputsJIT("year_prediction_msd", invertLoopsTileSize8Options, "one-tree-par-4cores")

defaultTileSize8Options = treebeard.CompilerOptions(200, 8)
defaultTileSize8MulticlassOptions = treebeard.CompilerOptions(200, 8)
defaultTileSize8MulticlassOptions.SetReturnTypeWidth(8)
defaultTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)

assert RunTestOnSingleModelTestInputsJIT("abalone", defaultTileSize8Options, "default-array")
assert RunTestOnSingleModelTestInputsJIT("airline", defaultTileSize8Options, "default-array")
assert RunTestOnSingleModelTestInputsJIT("airline-ohe", defaultTileSize8Options, "default-array")
assert RunTestOnSingleModelTestInputsJIT("covtype", defaultTileSize8MulticlassOptions, "default-array", numpy.int8)
assert RunTestOnSingleModelTestInputsJIT("epsilon", defaultTileSize8Options, "default-array")
assert RunTestOnSingleModelTestInputsJIT("higgs", defaultTileSize8Options, "default-array")
assert RunTestOnSingleModelTestInputsJIT("letters", defaultTileSize8MulticlassOptions, "default-array", numpy.int8)
assert RunTestOnSingleModelTestInputsJIT("year_prediction_msd", defaultTileSize8Options, "default-array")

invertLoopsTileSize8Options = treebeard.CompilerOptions(200, 8)
invertLoopsTileSize8Options.SetOneTreeAtATimeSchedule()

invertLoopsTileSize8MulticlassOptions = treebeard.CompilerOptions(200, 8)
invertLoopsTileSize8MulticlassOptions.SetReturnTypeWidth(8)
invertLoopsTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)
invertLoopsTileSize8MulticlassOptions.SetOneTreeAtATimeSchedule();

assert RunTestOnSingleModelTestInputsJIT("abalone", invertLoopsTileSize8Options, "one-tree-array")
assert RunTestOnSingleModelTestInputsJIT("airline", invertLoopsTileSize8Options, "one-tree-array")
assert RunTestOnSingleModelTestInputsJIT("airline-ohe", invertLoopsTileSize8Options, "one-tree-array")
assert RunTestOnSingleModelTestInputsJIT("covtype", invertLoopsTileSize8MulticlassOptions, "one-tree-array", numpy.int8)
assert RunTestOnSingleModelTestInputsJIT("epsilon", invertLoopsTileSize8Options, "one-tree-array")
assert RunTestOnSingleModelTestInputsJIT("higgs", invertLoopsTileSize8Options, "one-tree-array")
assert RunTestOnSingleModelTestInputsJIT("letters", invertLoopsTileSize8MulticlassOptions, "one-tree-array", numpy.int8)
assert RunTestOnSingleModelTestInputsJIT("year_prediction_msd", invertLoopsTileSize8Options, "one-tree-array")

treebeard.SetEnableSparseRepresentation(1)
assert RunTestOnSingleModelTestInputsJIT("abalone", invertLoopsTileSize8Options, "one-tree-sparse")
assert RunTestOnSingleModelTestInputsJIT("airline", invertLoopsTileSize8Options, "one-tree-sparse")
assert RunTestOnSingleModelTestInputsJIT("airline-ohe", invertLoopsTileSize8Options, "one-tree-sparse")
assert RunTestOnSingleModelTestInputsJIT("covtype", invertLoopsTileSize8MulticlassOptions, "one-tree-sparse", numpy.int8)
assert RunTestOnSingleModelTestInputsJIT("epsilon", invertLoopsTileSize8Options, "one-tree-sparse")
assert RunTestOnSingleModelTestInputsJIT("higgs", invertLoopsTileSize8Options, "one-tree-sparse")
assert RunTestOnSingleModelTestInputsJIT("letters", invertLoopsTileSize8MulticlassOptions, "one-tree-sparse", numpy.int8)
assert RunTestOnSingleModelTestInputsJIT("year_prediction_msd", invertLoopsTileSize8Options, "one-tree-sparse")

probTilingOptions = treebeard.CompilerOptions(200, 8)
probTilingOptions.SetTilingType(2) # prob tiling
probTilingOptions.SetReorderTreesByDepth(True)

probTilingMulticlassOptions = treebeard.CompilerOptions(200, 8)
probTilingMulticlassOptions.SetReturnTypeWidth(8)
probTilingMulticlassOptions.SetReturnTypeIsFloatType(False)

assert RunTestOnSingleModelTestInputsJIT("abalone", SetStatsProfileCSVPath(probTilingOptions, "abalone"), "default-probtiling-sparse")
assert RunTestOnSingleModelTestInputsJIT("airline", SetStatsProfileCSVPath(probTilingOptions, "airline"), "default-probtiling-sparse")
assert RunTestOnSingleModelTestInputsJIT("airline-ohe", SetStatsProfileCSVPath(probTilingOptions, "airline-ohe"), "default-probtiling-sparse")
assert RunTestOnSingleModelTestInputsJIT("covtype", SetStatsProfileCSVPath(probTilingMulticlassOptions, "covtype"), "default-probtiling-sparse", numpy.int8)
assert RunTestOnSingleModelTestInputsJIT("epsilon", SetStatsProfileCSVPath(probTilingOptions, "epsilon"), "default-probtiling-sparse")
assert RunTestOnSingleModelTestInputsJIT("higgs", SetStatsProfileCSVPath(probTilingOptions, "higgs"), "default-probtiling-sparse")
assert RunTestOnSingleModelTestInputsJIT("letters", SetStatsProfileCSVPath(probTilingMulticlassOptions, "letters"), "default-probtiling-sparse", numpy.int8)
assert RunTestOnSingleModelTestInputsJIT("year_prediction_msd", SetStatsProfileCSVPath(probTilingOptions, "year_prediction_msd"), "default-probtiling-sparse")

RunPipeliningTests()
RunParallelTests()

treebeard.SetEnableSparseRepresentation(0)

