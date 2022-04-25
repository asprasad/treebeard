import os
import sys

# This is a hack to make XGBoost run single threaded. 
# os.environ['OMP_NUM_THREADS'] = "1"
os.environ['TREELITE_BIND_THREADS']="0"

import numpy
import pandas
import time
import math
from scipy.stats.mstats import gmean
import treelite
import treelite_runtime

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(filepath)))
# print(treebeard_repo_dir)
treebeard_runtime_dir = os.path.join(os.path.join(treebeard_repo_dir, "src"), "python")
# print(treebeard_runtime_dir)

sys.path.append(treebeard_runtime_dir)

import treebeard
import xgboost as xgb

num_repeats = 100
use_inplace_predict = True
run_on_same_array_repeatedly = True
run_parallel = False

# def RunSingleTest_Multibatch(soPath, globalsJSONPath, csvPath, returnType) -> float:
#   data_df = pandas.read_csv(csvPath, header=None)
#   data = numpy.array(data_df, order='C') # numpy.genfromtxt(csvPath, ',')

#   inferenceRunner = treebeard.TreebeardInferenceRunner(soPath, globalsJSONPath)
#   batch_size = inferenceRunner.batchSize
#   num_batches = int(math.floor(data.shape[0]/inferenceRunner.batchSize))
#   numRows = int(math.floor(data.shape[0]/inferenceRunner.batchSize) * inferenceRunner.batchSize)
#   inputs = numpy.array(data[0:numRows, :-1], numpy.float32, order='C')
#   repeated_inputs = numpy.tile(inputs, (num_repeats, 1))

#   start = time.time()
#   if run_on_same_array_repeatedly:
#     for i in range(num_repeats):
#       # results = inferenceRunner.RunInferenceOnMultipleBatches(inputs, returnType)
#       for j in range(0, num_batches):
#         start_index = j*batch_size
#         stop_index = start_index + batch_size
#         batch = inputs[start_index:stop_index, :]
#         results = inferenceRunner.RunInference(batch, returnType)
#   else:
#     results = inferenceRunner.RunInferenceOnMultipleBatches(repeated_inputs, returnType)
#   end = time.time()

#   print("(", end - start, "s )")
#   return (end - start)

# def RunTestOnSingleModelTestInputs_Multibatch(modelName : str, sparse, invert, returnType=numpy.float32) -> float:
#   print("TreeBeard ",  modelName, "...", end=" ")
#   invert_str = "_invert" if invert else ""
#   sparse_str = "_sparse" if sparse else ""
#   soPath = os.path.join(os.path.join(treebeard_repo_dir, "runtime_test_binaries"), modelName + "_t8_b64_f_i16" + sparse_str + invert_str + ".so")
#   # print(soPath)
#   globalsJSONPath = soPath + ".treebeard-globals.json"
#   csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
#   return RunSingleTest_Multibatch(soPath, globalsJSONPath, csvPath, returnType)

def RunSingleTest_XGBoost(modelJSONPath, csvPath) -> float:
  booster = xgb.Booster(model_file=modelJSONPath)
  numFeatures = booster.num_features()
  data_df = pandas.read_csv(csvPath, header=None)
  full_test_array = numpy.array(data_df, order='C') # numpy.genfromtxt(csvPath, ',')

  # x_numpy = full_test_array[:, 0:numFeatures]
  num_batches = int(math.floor(full_test_array.shape[0]/batchSize))
  numRows = int(math.floor(full_test_array.shape[0]/batchSize) * batchSize)
  if not run_parallel:
    booster.set_param('nthread', 1)

  inputs = numpy.array(full_test_array[0:numRows, :-1], numpy.float32, order='C')
  repeated_inputs = numpy.tile(inputs, (num_repeats, 1))
  x = xgb.DMatrix(repeated_inputs, feature_names=booster.feature_names, )

  start = time.time()
  if run_on_same_array_repeatedly:
    for i in range(num_repeats):
      # pred = booster.inplace_predict(inputs, validate_features=False)
      for j in range(0, num_batches):
        start_index = j*batchSize
        stop_index = start_index + batchSize
        batch = inputs[start_index:stop_index, :]
        pred = booster.inplace_predict(batch, validate_features=False)

  else:
    if use_inplace_predict:
      pred = booster.inplace_predict(repeated_inputs, validate_features=False)
    else:
      pred = booster.predict(x, validate_features=False)
  
  end = time.time()
  # print("(", end - start, "s )")
  return (end-start)/(batchSize * num_batches * num_repeats)


def RunTestOnSingleModelTestInputs_XGBoost(modelName : str) -> None:
  # print("XGBoost ",  modelName, "...", end=" ")
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  return RunSingleTest_XGBoost(modelJSON, csvPath)

def RunSingleTestJIT_Multibatch(modelJSONPath, csvPath, options, returnType) -> float:
  data_df = pandas.read_csv(csvPath, header=None)
  data = numpy.array(data_df, order='C') # numpy.genfromtxt(csvPath, ',')

  inferenceRunner = treebeard.TreebeardInferenceRunner.FromModelFile(modelJSONPath, "", options)
  batch_size = inferenceRunner.batchSize
  num_batches = int(math.floor(data.shape[0]/inferenceRunner.batchSize))
  numRows = int(math.floor(data.shape[0]/inferenceRunner.batchSize) * inferenceRunner.batchSize)
  inputs = numpy.array(data[0:numRows, :-1], numpy.float32, order='C')
  repeated_inputs = numpy.tile(inputs, (num_repeats, 1))

  start = time.time()
  if run_on_same_array_repeatedly:
    for i in range(num_repeats):
      # results = inferenceRunner.RunInferenceOnMultipleBatches(inputs, returnType)
      for j in range(0, num_batches):
        start_index = j*batch_size
        stop_index = start_index + batch_size
        batch = inputs[start_index:stop_index, :]
        results = inferenceRunner.RunInference(batch, returnType)
  else:
    results = inferenceRunner.RunInferenceOnMultipleBatches(repeated_inputs, returnType)
  end = time.time()

  # print("(", end - start, "s )")
  return (end - start)/(batchSize * num_batches * num_repeats)

def RunTestOnSingleModelTestInputsJIT_Multibatch(modelName : str, sparse, options, returnType=numpy.float32) -> float:
  treebeard.SetEnableSparseRepresentation(1 if sparse else 0)
  # print("TreeBeard ",  modelName, "...", end=" ")
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  execTime = RunSingleTestJIT_Multibatch(modelJSON, csvPath, options, returnType)
  treebeard.SetEnableSparseRepresentation(0)
  return execTime

def RunSingleTest_Treelite(modelJSONPath, csvPath, modelName) -> float:
  booster = xgb.Booster(model_file=modelJSONPath)
  treeliteModel = treelite.Model.from_xgboost(booster)
  generated_code_dir = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/test/python/treelite_generated_code"
  model_code_dir = os.path.join(generated_code_dir, modelName)
  if not os.path.exists(model_code_dir):
    os.makedirs(model_code_dir)
    compile_options = dict()
    compile_options["parallel_comp"] = 100
    treeliteModel.compile(model_code_dir, params=compile_options)
    treelite_so_path = treelite.create_shared("clang", model_code_dir)
  else:
    treelite_so_path = os.path.join(model_code_dir, "predictor.so")

  if run_parallel:
    predictor = treelite_runtime.Predictor(treelite_so_path)
  else:
    predictor = treelite_runtime.Predictor(treelite_so_path, nthread=1)

  data_df = pandas.read_csv(csvPath, header=None)
  full_test_array = numpy.array(data_df, order='C') # numpy.genfromtxt(csvPath, ',')

  # x_numpy = full_test_array[:, 0:numFeatures]
  num_batches = int(math.floor(full_test_array.shape[0]/batchSize))
  numRows = int(math.floor(full_test_array.shape[0]/batchSize) * batchSize)

  inputs = numpy.array(full_test_array[0:numRows, :-1], numpy.float32, order='C')
  repeated_inputs = numpy.tile(inputs, (num_repeats, 1))

  start = time.time()
  if run_on_same_array_repeatedly:
    for i in range(num_repeats):
      # dmat = treelite_runtime.DMatrix(inputs)
      # predictor.predict(dmat, pred_margin=False)
      for j in range(0, num_batches):
        start_index = j*batchSize
        stop_index = start_index + batchSize
        batch = inputs[start_index:stop_index, :]
        dmat = treelite_runtime.DMatrix(batch)
        predictor.predict(dmat, pred_margin=False)

  end = time.time()
  # print("(", end - start, "s )")
  return (end-start)/(batchSize * num_batches * num_repeats)

def RunTestOnSingleModelTestInputs_Treelite(modelName : str) -> None:
  # print("Treelite ",  modelName, "...", end=" ")
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  return RunSingleTest_Treelite(modelJSON, csvPath, modelName)

modelNames = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "letters", "higgs", "year_prediction_msd"]
sparse = [True, True, True, True, True, True, True, True ]
invert = [True, True, True, True, False, True, True, True ]
return_types = [numpy.float32, numpy.float32, numpy.float32, numpy.int8, numpy.float32, numpy.int8, numpy.float32, numpy.float32 ]

def ConstructCompilerOptionsList():
  num_cores = 16
  invertLoopsTileSize8Options = treebeard.CompilerOptions(batchSize, 8)
  # invertLoopsTileSize8Options.SetOneTreeAtATimeSchedule()
  invertLoopsTileSize8Options.SetPipelineWidth(8)
  invertLoopsTileSize8Options.SetReorderTreesByDepth(True)
  invertLoopsTileSize8Options.SetMakeAllLeavesSameDepth(1)
  if run_parallel:
    invertLoopsTileSize8Options.SetNumberOfCores(num_cores)

  invertLoopsTileSize8MulticlassOptions = treebeard.CompilerOptions(batchSize, 8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)
  # invertLoopsTileSize8MulticlassOptions.SetOneTreeAtATimeSchedule();
  invertLoopsTileSize8MulticlassOptions.SetPipelineWidth(8)
  invertLoopsTileSize8MulticlassOptions.SetReorderTreesByDepth(True)
  invertLoopsTileSize8MulticlassOptions.SetMakeAllLeavesSameDepth(1)
  if run_parallel:
    invertLoopsTileSize8MulticlassOptions.SetNumberOfCores(num_cores)

  optionsList = []
  for model in modelNames:
    if model=="covtype" or model=="letters":
      optionsList.append(invertLoopsTileSize8MulticlassOptions)
    else:
      optionsList.append(invertLoopsTileSize8Options)
  return optionsList

def print_result_list(label, values):
  print(label, batchSize, sep=", ", end="")
  for val in values:
    print(", ", val, end="")
  print(" ")

def run_benchmarks_for_batchsize():
  treebeard_times = []
  xgboost_times = []
  treelite_times = []
  xgboost_speedups = []
  treelite_speedups = []
  options = ConstructCompilerOptionsList()
  i = 0
  for model in modelNames:
    treebeardTime = RunTestOnSingleModelTestInputsJIT_Multibatch(model, sparse[i], options[i], return_types[i])
    treebeard_times.append(treebeardTime)
    xgBoostTime = RunTestOnSingleModelTestInputs_XGBoost(model)
    xgboost_times.append(xgBoostTime)
    treeliteTime = RunTestOnSingleModelTestInputs_Treelite(model)
    treelite_times.append(treeliteTime)
    speedup = xgBoostTime/treebeardTime
    xgboost_speedups.append(speedup)
    treelite_speedup = treeliteTime/treebeardTime
    treelite_speedups.append(treelite_speedup)
    i += 1
    # print("XGBoost Speedup : ", speedup)
    # print("Treelite Speedup : ", treelite_speedup)

  # print("XGBoost geomean speedup : ", gmean(xgboost_speedups))
  # print("Treelite geomean speedup : ", gmean(treelite_speedups))
  print_result_list("treebeard", treebeard_times)
  print_result_list("xgboost", xgboost_times)
  print_result_list("treelite", treelite_times)
  print_result_list("xgboost speedup", xgboost_speedups)
  print_result_list("treelite speedup", treelite_speedups)

batchSizes = [64, 128, 256, 512, 1024, 2000]
for batchSize in batchSizes:
  run_benchmarks_for_batchsize()