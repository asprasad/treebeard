import os
import sys

# This is a hack to make XGBoost run single threaded. 
os.environ['OMP_NUM_THREADS'] = "1"

import numpy
import pandas
import time
import math
from scipy.stats.mstats import gmean

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

def RunSingleTest_Multibatch(soPath, globalsJSONPath, csvPath) -> float:
  data_df = pandas.read_csv(csvPath, header=None)
  data = numpy.array(data_df, order='C') # numpy.genfromtxt(csvPath, ',')

  inferenceRunner = treebeard.TreebeardInferenceRunner(soPath, globalsJSONPath)

  numRows = int(math.floor(data.shape[0]/inferenceRunner.batchSize) * inferenceRunner.batchSize)
  inputs = numpy.array(data[0:numRows, :-1], numpy.float32, order='C')
  repeated_inputs = numpy.tile(inputs, (num_repeats, 1))

  start = time.time()
  if run_on_same_array_repeatedly:
    for i in range(num_repeats):
      results = inferenceRunner.RunInferenceOnMultipleBatches(inputs)
  else:
    results = inferenceRunner.RunInferenceOnMultipleBatches(repeated_inputs)
  end = time.time()

  print("(", end - start, "s )")
  return (end - start)

def RunTestOnSingleModelTestInputs_Multibatch(modelName : str, sparse, invert) -> float:
  print("TreeBeard ",  modelName, "...", end=" ")
  invert_str = "_invert" if invert else ""
  sparse_str = "_sparse" if sparse else ""
  soPath = os.path.join(os.path.join(treebeard_repo_dir, "runtime_test_binaries"), modelName + "_t8_b64_f_i16" + sparse_str + invert_str + ".so")
  # print(soPath)
  globalsJSONPath = soPath + ".treebeard-globals.json"
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  return RunSingleTest_Multibatch(soPath, globalsJSONPath, csvPath)

def RunSingleTest_XGBoost(modelJSONPath, csvPath) -> float:
  booster = xgb.Booster(model_file=modelJSONPath)
  numFeatures = booster.num_features()
  data_df = pandas.read_csv(csvPath, header=None)
  full_test_array = numpy.array(data_df, order='C') # numpy.genfromtxt(csvPath, ',')

  # x_numpy = full_test_array[:, 0:numFeatures]
  batchSize = 64
  numRows = int(math.floor(full_test_array.shape[0]/batchSize) * batchSize)
  booster.set_param('nthread', 1)

  inputs = numpy.array(full_test_array[0:numRows, :-1], numpy.float32, order='C')
  repeated_inputs = numpy.tile(inputs, (num_repeats, 1))
  x = xgb.DMatrix(repeated_inputs, feature_names=booster.feature_names, )

  start = time.time()
  if run_on_same_array_repeatedly:
    for i in range(num_repeats):
      pred = booster.inplace_predict(inputs, validate_features=False)
  else:
    if use_inplace_predict:
      pred = booster.inplace_predict(repeated_inputs, validate_features=False)
    else:
      pred = booster.predict(x, validate_features=False)
  
  end = time.time()
  print("(", end - start, "s )")
  return end-start


def RunTestOnSingleModelTestInputs_XGBoost(modelName : str) -> None:
  print("XGBoost ",  modelName, "...", end=" ")
  modelJSON = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json")
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  return RunSingleTest_XGBoost(modelJSON, csvPath)

modelNames = ["abalone", "airline", "airline-ohe", "epsilon", "higgs", "year_prediction_msd"]
# modelNames = ["epsilon"]
speedups = []

for model in modelNames:
  sparse = False
  invert = True
  if model == "epsilon":
     sparse = True
     invert = False
  treebeardTime = RunTestOnSingleModelTestInputs_Multibatch(model, sparse, invert)
  xgBoostTime = RunTestOnSingleModelTestInputs_XGBoost(model)
  speedup = xgBoostTime/treebeardTime
  speedups.append(speedup)
  print("Speedup : ", speedup)

print("Geomean speedup : ", gmean(speedups))




