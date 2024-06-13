import os
import argparse

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

import treebeard
import xgboost as xgb

num_repeats = 1000
run_parallel = True
# Treelite complains if we specify a number higher than the number of cores.
# So just limiting the number of cores in case machine has fewer than 16. 
num_cores = min(16, os.cpu_count())

def RunSingleTest_XGBoost(modelJSONPath, csvPath) -> float:
  booster = xgb.Booster(model_file=modelJSONPath)
  data_df = pandas.read_csv(csvPath, header=None)
  full_test_array = numpy.array(data_df, order='C') 

  num_batches = int(math.floor(full_test_array.shape[0]/batchSize))
  numRows = int(math.floor(full_test_array.shape[0]/batchSize) * batchSize)
  if run_parallel:
    booster.set_param('nthread', num_cores)

  inputs = numpy.array(full_test_array[0:numRows, :-1], numpy.float32, order='C')

  start = time.time()
  for i in range(num_repeats):
    for j in range(0, num_batches):
      start_index = j*batchSize
      stop_index = start_index + batchSize
      batch = inputs[start_index:stop_index, :]
      pred = booster.inplace_predict(batch, validate_features=False)

  end = time.time()
  return (end-start)/(batchSize * num_batches * num_repeats)


def RunTestOnSingleModelTestInputs_XGBoost(modelName : str) -> None:
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

  start = time.time()
  for i in range(num_repeats):
    for j in range(0, num_batches):
      start_index = j*batch_size
      stop_index = start_index + batch_size
      batch = inputs[start_index:stop_index, :]
      results = inferenceRunner.RunInference(batch, returnType)
  end = time.time()
  return (end - start)/(batchSize * num_batches * num_repeats)

def RunTestOnSingleModelTestInputsJIT_Multibatch(modelName : str, sparse, options, returnType=numpy.float32) -> float:
  treebeard.SetEnableSparseRepresentation(1 if sparse else 0)
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  execTime = RunSingleTestJIT_Multibatch(modelJSON, csvPath, options, returnType)
  treebeard.SetEnableSparseRepresentation(0)
  return execTime

def RunSingleTest_Treelite(modelJSONPath, csvPath, modelName) -> float:
  booster = xgb.Booster(model_file=modelJSONPath)
  treeliteModel = treelite.Model.from_xgboost(booster)
  generated_code_dir = os.path.join(os.path.join(os.path.join(treebeard_repo_dir, "test"), "python"), "treelite_generated_code")
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
    predictor = treelite_runtime.Predictor(treelite_so_path, nthread=num_cores)
  else:
    predictor = treelite_runtime.Predictor(treelite_so_path, nthread=1)

  data_df = pandas.read_csv(csvPath, header=None)
  full_test_array = numpy.array(data_df, order='C')

  num_batches = int(math.floor(full_test_array.shape[0]/batchSize))
  numRows = int(math.floor(full_test_array.shape[0]/batchSize) * batchSize)

  inputs = numpy.array(full_test_array[0:numRows, :-1], numpy.float32, order='C')

  start = time.time()
  for i in range(num_repeats):
    for j in range(0, num_batches):
      start_index = j*batchSize
      stop_index = start_index + batchSize
      batch = inputs[start_index:stop_index, :]
      dmat = treelite_runtime.DMatrix(batch)
      predictor.predict(dmat, pred_margin=False)

  end = time.time()
  return (end-start)/(batchSize * num_batches * num_repeats)

def RunTestOnSingleModelTestInputs_Treelite(modelName : str) -> None:
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  return RunSingleTest_Treelite(modelJSON, csvPath, modelName)

modelNames = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "letters", "higgs", "year_prediction_msd"]
sparse = [True, True, True, True, True, True, True, True ]
invert = [True, True, True, True, False, True, True, True ]
return_types = [numpy.float32, numpy.float32, numpy.float32, numpy.int8, numpy.float32, numpy.int8, numpy.float32, numpy.float32 ]

def ConstructCompilerOptionsList(tile_size, pipeline_width):
  invertLoopsTileSize8Options = treebeard.CompilerOptions(batchSize, tile_size)
  invertLoopsTileSize8Options.SetPipelineWidth(pipeline_width)
  invertLoopsTileSize8Options.SetReorderTreesByDepth(True)
  invertLoopsTileSize8Options.SetMakeAllLeavesSameDepth(1)
  if run_parallel:
    invertLoopsTileSize8Options.SetNumberOfCores(num_cores)

  invertLoopsTileSize8MulticlassOptions = treebeard.CompilerOptions(batchSize, tile_size)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)
  invertLoopsTileSize8MulticlassOptions.SetPipelineWidth(pipeline_width)
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

def ConstructTreeParallelCompilerOptionsList(tile_size, pipeline_width):
  num_par_tree_sets = 10
  invertLoopsTileSize8Options = treebeard.CompilerOptions(batchSize, tile_size)
  invertLoopsTileSize8Options.SetPipelineWidth(pipeline_width)
  invertLoopsTileSize8Options.SetReorderTreesByDepth(True)
  invertLoopsTileSize8Options.SetMakeAllLeavesSameDepth(1)
  if run_parallel:
    # invertLoopsTileSize8Options.SetNumberOfCores(int(num_cores/num_par_tree_sets))
    invertLoopsTileSize8Options.SetNumberOfParallelTreeBatches(num_par_tree_sets)

  invertLoopsTileSize8MulticlassOptions = treebeard.CompilerOptions(batchSize, tile_size)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)
  invertLoopsTileSize8MulticlassOptions.SetPipelineWidth(pipeline_width)
  invertLoopsTileSize8MulticlassOptions.SetReorderTreesByDepth(True)
  invertLoopsTileSize8MulticlassOptions.SetMakeAllLeavesSameDepth(1)
  if run_parallel:
    # invertLoopsTileSize8MulticlassOptions.SetNumberOfCores(int(num_cores/num_par_tree_sets))
    invertLoopsTileSize8MulticlassOptions.SetNumberOfParallelTreeBatches(num_par_tree_sets)

  optionsList = []
  for model in modelNames:
    if model=="covtype" or model=="letters":
      optionsList.append(invertLoopsTileSize8MulticlassOptions)
    else:
      optionsList.append(invertLoopsTileSize8Options)
  return optionsList

def print_title_row():
  print("Title", "Batch Size", sep=", ", end="")
  for model in modelNames:
    print(", ", model, end="")
  print(" ")

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
  options = ConstructCompilerOptionsList(8, 8)
  optionsTreePar = ConstructTreeParallelCompilerOptionsList(8, 8)
  i = 0
  for model in modelNames:
    treebeardTime = RunTestOnSingleModelTestInputsJIT_Multibatch(model, sparse[i], options[i], return_types[i])
    treebeard_times.append(treebeardTime)
    xgBoostTime = RunTestOnSingleModelTestInputsJIT_Multibatch(model, sparse[i], optionsTreePar[i], return_types[i]) # RunTestOnSingleModelTestInputs_XGBoost(model)
    xgboost_times.append(xgBoostTime)
    # treeliteTime = RunTestOnSingleModelTestInputs_Treelite(model)
    # treelite_times.append(treeliteTime)
    speedup = xgBoostTime/treebeardTime
    xgboost_speedups.append(speedup)
    # treelite_speedup = treeliteTime/treebeardTime
    # treelite_speedups.append(treelite_speedup)
    i += 1

  print_title_row()
  print_result_list("Treebeard (s)", treebeard_times)
  print_result_list("XGBoost (s)", xgboost_times)
  # print_result_list("Treelite (s)", treelite_times)
  print_result_list("TB speedup vs XGBoost", xgboost_speedups)
  # print_result_list("TB speedup vs Treelite", treelite_speedups)
  print("Treebeard geomean speedup vs XGBoost: ", gmean(xgboost_speedups))
  # print("Treebeard geomean speedup vs Treelite: ", gmean(treelite_speedups))

def run_benchmarks_for_several_configs():
  print("Running configuration exploration")
  treebeard_times = []
  xgboost_times = []
  treelite_times = []
  xgboost_speedups = []
  treelite_speedups = []
  list_of_options = [ConstructCompilerOptionsList(4, 2), ConstructCompilerOptionsList(4, 4), ConstructCompilerOptionsList(4, 8), 
                     ConstructCompilerOptionsList(8, 2), ConstructCompilerOptionsList(8, 4), ConstructCompilerOptionsList(8, 8)]
  i = 0
  for model in modelNames:
    model_treebeard_times = []
    for options in list_of_options:
      treebeardTime = RunTestOnSingleModelTestInputsJIT_Multibatch(model, sparse[i], options[i], return_types[i])
      model_treebeard_times.append(treebeardTime)
    
    treebeard_times.append(min(model_treebeard_times))
    xgBoostTime = RunTestOnSingleModelTestInputs_XGBoost(model)
    xgboost_times.append(xgBoostTime)
    treeliteTime = RunTestOnSingleModelTestInputs_Treelite(model)
    treelite_times.append(treeliteTime)
    speedup = xgBoostTime/treebeardTime
    xgboost_speedups.append(speedup)
    treelite_speedup = treeliteTime/treebeardTime
    treelite_speedups.append(treelite_speedup)
    i += 1

  print_title_row()
  print_result_list("Treebeard (s)", treebeard_times)
  print_result_list("XGBoost (s)", xgboost_times)
  print_result_list("Treelite (s)", treelite_times)
  print_result_list("TB speedup vs XGBoost", xgboost_speedups)
  print_result_list("TB speedup vs Treelite", treelite_speedups)
  print("Treebeard geomean speedup vs XGBoost: ", gmean(xgboost_speedups))
  print("Treebeard geomean speedup vs Treelite: ", gmean(treelite_speedups))

# batchSizes = [64, 128, 256, 512, 1024, 2000]
batchSize = -1
batchSizes = [32, 64]

def run_benchmarks(benchmark_func):
  global batchSize
  for b in batchSizes:
    batchSize = b
    benchmark_func()

parser = argparse.ArgumentParser()
parser.add_argument("--explore", help = "Explore a fixed set of configurations and find the best", action="store_true")
parser.add_argument("--num_cores", help = "Number of cores to use")
args = parser.parse_args()

if (args.num_cores):
  print("Using", args.num_cores, "cores")
  num_cores=int(args.num_cores)

if (args.explore):
  run_benchmarks(run_benchmarks_for_several_configs)
else:
  run_benchmarks(run_benchmarks_for_batchsize)
