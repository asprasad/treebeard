import os
import numpy
import pandas
import time
import treebeard
from functools import partial

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(filepath)))

def CheckEqual(a, b) -> bool:
  if type(a) is numpy.float32:
    scaledThreshold = max(abs(a), abs(b))/1e8
    threshold = max(1e-6, scaledThreshold)
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
      print(x[i], y[i])
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

def RunSingleTestJIT_TBContext(modelJSONPath, csvPath, options, returnType, representation, inputType) -> bool:
  data_df = pandas.read_csv(csvPath, header=None)
  data = numpy.array(data_df, order='C') # numpy.genfromtxt(csvPath, ',')
  inputs = numpy.array(data[:, :-1], numpy.float32, order='C')
  expectedOutputs = data[:, data.shape[1]-1]
  
  globalsPath = modelJSONPath + ".treebeard-globals.json"
  tbContext = treebeard.TreebeardContext(modelJSONPath, globalsPath, options)
  tbContext.SetRepresentationType(representation)
  tbContext.SetInputFiletype(inputType)

  inferenceRunner = treebeard.TreebeardInferenceRunner.FromTBContext(tbContext)
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

def RunSingleTestJIT_ScheduleManipulation(modelJSONPath, 
                                          csvPath,
                                          options,
                                          returnType,
                                          representation,
                                          inputType,
                                          scheduleManipulator) -> bool:
  data_df = pandas.read_csv(csvPath, header=None)
  data = numpy.array(data_df, order='C') # numpy.genfromtxt(csvPath, ',')
  inputs = numpy.array(data[:, :-1], numpy.float32, order='C')
  expectedOutputs = data[:, data.shape[1]-1]
  
  globalsPath = modelJSONPath + ".treebeard-globals.json"
  tbContext = treebeard.TreebeardContext(modelJSONPath, globalsPath, options)
  tbContext.SetRepresentationType(representation)
  tbContext.SetInputFiletype(inputType)

  tbContext.BuildHIRRepresentation()
  schedule = tbContext.GetSchedule()
  scheduleManipulator(schedule)
  inferenceRunner = tbContext.ConstructInferenceRunnerFromHIR()
  
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

def RunTestOnSingleModelTestInputsJIT(modelName : str, options, testName : str, returnType=numpy.float32, testFunc=RunSingleTestJIT) -> bool:
  print("JIT ", testName, modelName, "...", end=" ")
  modelJSONPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  return testFunc(modelJSONPath, csvPath, options, returnType)

def SetStatsProfileCSVPath(options, modelName):
  profilesDir = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), "profiles") 
  csvPath = os.path.join(profilesDir, modelName + ".test.csv")
  options.SetStatsProfileCSVPath(csvPath)
  return options

def RunAllTests(testName, options, multiclassOptions, singleTestRunner):
  assert RunTestOnSingleModelTestInputsJIT("abalone", options, testName, numpy.float32, singleTestRunner)
  assert RunTestOnSingleModelTestInputsJIT("airline", options, testName, numpy.float32, singleTestRunner)
  assert RunTestOnSingleModelTestInputsJIT("airline-ohe", options, testName, numpy.float32, singleTestRunner)
  assert RunTestOnSingleModelTestInputsJIT("covtype", multiclassOptions, testName, numpy.int8, singleTestRunner)
  assert RunTestOnSingleModelTestInputsJIT("epsilon", options, testName, numpy.float32, singleTestRunner)
  assert RunTestOnSingleModelTestInputsJIT("higgs", options, testName, numpy.float32, singleTestRunner)
  assert RunTestOnSingleModelTestInputsJIT("letters", multiclassOptions, testName, numpy.int8, singleTestRunner)
  assert RunTestOnSingleModelTestInputsJIT("year_prediction_msd", options, testName, numpy.float32, singleTestRunner)


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

  RunAllTests("one-tree-pipeline8-sparse", invertLoopsTileSize8Options, invertLoopsTileSize8MulticlassOptions, RunSingleTestJIT)

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

  RunAllTests("one-tree-par-4cores", invertLoopsTileSize8Options, invertLoopsTileSize8MulticlassOptions, RunSingleTestJIT)

def RunProbBasedTilingTests():
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

def RunBasicTests():
  defaultTileSize8Options = treebeard.CompilerOptions(200, 8)
  defaultTileSize8MulticlassOptions = treebeard.CompilerOptions(200, 8)
  defaultTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  defaultTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)

  RunAllTests("default-array", defaultTileSize8Options, defaultTileSize8MulticlassOptions, RunSingleTestJIT)

  invertLoopsTileSize8Options = treebeard.CompilerOptions(200, 8)
  invertLoopsTileSize8Options.SetOneTreeAtATimeSchedule()

  invertLoopsTileSize8MulticlassOptions = treebeard.CompilerOptions(200, 8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)
  invertLoopsTileSize8MulticlassOptions.SetOneTreeAtATimeSchedule()
  
  RunAllTests("one-tree-array", invertLoopsTileSize8Options, invertLoopsTileSize8MulticlassOptions, RunSingleTestJIT)

  treebeard.SetEnableSparseRepresentation(1)
  
  RunAllTests("one-tree-sparse", invertLoopsTileSize8Options, invertLoopsTileSize8MulticlassOptions, RunSingleTestJIT)
  RunProbBasedTilingTests()
  
  treebeard.SetEnableSparseRepresentation(0)

def RunTBContextTests():
  defaultTileSize8Options = treebeard.CompilerOptions(200, 8)
  defaultTileSize8MulticlassOptions = treebeard.CompilerOptions(200, 8)
  defaultTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  defaultTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)

  arrayRepSingleTestRunner = partial(RunSingleTestJIT_TBContext, representation="array", inputType="xgboost_json")
  sparseRepSingleTestRunner = partial(RunSingleTestJIT_TBContext, representation="sparse", inputType="xgboost_json")

  RunAllTests("default-array-tbcontext", defaultTileSize8Options, defaultTileSize8MulticlassOptions, arrayRepSingleTestRunner)

  RunAllTests("default-sparse-tbcontext", defaultTileSize8Options, defaultTileSize8MulticlassOptions, sparseRepSingleTestRunner)

  invertLoopsTileSize8Options = treebeard.CompilerOptions(200, 8)
  invertLoopsTileSize8Options.SetOneTreeAtATimeSchedule()

  invertLoopsTileSize8MulticlassOptions = treebeard.CompilerOptions(200, 8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)
  invertLoopsTileSize8MulticlassOptions.SetOneTreeAtATimeSchedule();

  RunAllTests("one-tree-array-tbcontext", invertLoopsTileSize8Options, invertLoopsTileSize8MulticlassOptions, arrayRepSingleTestRunner)

  RunAllTests("one-tree-sparse-tbcontext", invertLoopsTileSize8Options, invertLoopsTileSize8MulticlassOptions, sparseRepSingleTestRunner)

def TileBatchLoopSchedule(schedule: treebeard.Schedule):
  batchIndex = schedule.GetBatchIndex()
  outerIndex = schedule.NewIndexVariable("b0")
  innerIndex = schedule.NewIndexVariable("b1")
  schedule.Tile(batchIndex, outerIndex, innerIndex, tileSize=4)

def TileTreeLoopSchedule(schedule: treebeard.Schedule):
  treeIndex = schedule.GetTreeIndex()
  outerIndex = schedule.NewIndexVariable("b0")
  innerIndex = schedule.NewIndexVariable("b1")
  schedule.Tile(treeIndex, outerIndex, innerIndex, tileSize=4)

def RunTileBatchLoopTests():
  defaultTileSize8Options = treebeard.CompilerOptions(200, 8)
  defaultTileSize8MulticlassOptions = treebeard.CompilerOptions(200, 8)
  defaultTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  defaultTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)

  arrayRepSingleTestRunner_TileBatch = partial(RunSingleTestJIT_ScheduleManipulation,
                                               representation="array",
                                               inputType="xgboost_json",
                                               scheduleManipulator=TileBatchLoopSchedule)

  RunAllTests("tiled_batch-array-tbcontext", defaultTileSize8Options, defaultTileSize8MulticlassOptions, arrayRepSingleTestRunner_TileBatch)

def RunTileTreeLoopTests():
  defaultTileSize8Options = treebeard.CompilerOptions(200, 8)
  defaultTileSize8MulticlassOptions = treebeard.CompilerOptions(200, 8)
  defaultTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  defaultTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)

  arrayRepSingleTestRunner_TileTree = partial(RunSingleTestJIT_ScheduleManipulation,
                                               representation="array",
                                               inputType="xgboost_json",
                                               scheduleManipulator=TileTreeLoopSchedule)

  RunAllTests("tiled_tree-array-tbcontext", defaultTileSize8Options, defaultTileSize8MulticlassOptions, arrayRepSingleTestRunner_TileTree)

def ScheduleTest():
  RunTileBatchLoopTests()
  RunTileTreeLoopTests()

ScheduleTest()
RunTBContextTests()
RunBasicTests()

treebeard.SetEnableSparseRepresentation(1)

RunPipeliningTests()
RunParallelTests()

treebeard.SetEnableSparseRepresentation(0)

