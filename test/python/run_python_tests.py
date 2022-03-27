import sys
import os
from xmlrpc.client import Boolean
import numpy
import pandas
import time

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(filepath)))
# print(treebeard_repo_dir)
treebeard_runtime_dir = os.path.join(os.path.join(treebeard_repo_dir, "src"), "python")
# print(treebeard_runtime_dir)

sys.path.append(treebeard_runtime_dir)

import treebeard

def CheckEqual(a, b) -> Boolean:
  if type(a) is numpy.float32:
    threshold = 1e-6
    return pow(a-b, 2) < threshold
  else:
    return a==b

def CheckArraysEqual(x, y) -> Boolean:
  assert len(x.shape) == 1
  assert len(y.shape) == 1
  if not x.shape[0] == y.shape[0]:
    return False
  for i in range(x.shape[0]):
    if not CheckEqual(x[i], y[i]):
      return False
  return True

def RunSingleTest(soPath, globalsJSONPath, csvPath, returnType) -> Boolean:
  data_df = pandas.read_csv(csvPath, header=None)
  data = numpy.array(data_df, order='C') # numpy.genfromtxt(csvPath, ',')
  inputs = numpy.array(data[:, :-1], numpy.float32, order='C')
  expectedOutputs = data[:, data.shape[1]-1]
  
  inferenceRunner = treebeard.TreebeardInferenceRunner(soPath, globalsJSONPath)
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

def RunTestOnSingleModelRandomInputs(modelName : str, returnType=numpy.float32) -> Boolean:
  print("Running random input test", modelName, "...", end=" ")
  soPath = os.path.join(os.path.join(treebeard_repo_dir, "runtime_test_binaries"), modelName + "_t8_b200_f_i16_invert.so")
  globalsJSONPath = soPath + ".treebeard-globals.json"
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.csv")
  return RunSingleTest(soPath, globalsJSONPath, csvPath, returnType)

def RunTestOnSingleModelTestInputs(modelName : str, returnType=numpy.float32) -> Boolean:
  print("Running actual input test",  modelName, "...", end=" ")
  soPath = os.path.join(os.path.join(treebeard_repo_dir, "runtime_test_binaries"), modelName + "_t8_b200_f_i16_invert.so")
  globalsJSONPath = soPath + ".treebeard-globals.json"
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  return RunSingleTest(soPath, globalsJSONPath, csvPath, returnType)

def RunSingleTest_Multibatch(soPath, globalsJSONPath, csvPath, returnType=numpy.float32) -> Boolean:
  data_df = pandas.read_csv(csvPath, header=None)
  data = numpy.array(data_df, order='C') # numpy.genfromtxt(csvPath, ',')
  inputs = numpy.array(data[:, :-1], numpy.float32, order='C')
  expectedOutputs = data[:, data.shape[1]-1]
  inferenceRunner = treebeard.TreebeardInferenceRunner(soPath, globalsJSONPath)

  start = time.time()
  results = inferenceRunner.RunInferenceOnMultipleBatches(inputs, returnType)
  if not CheckArraysEqual(results, expectedOutputs):
      print("Failed")
      return False
  end = time.time()
  print("Passed (", end - start, "s )")
  return True

def RunTestOnSingleModelRandomInputs_Multibatch(modelName : str, returnType=numpy.float32) -> Boolean:
  print("Running multibatch random input test", modelName, "...", end=" ")
  soPath = os.path.join(os.path.join(treebeard_repo_dir, "runtime_test_binaries"), modelName + "_t8_b200_f_i16_invert.so")
  globalsJSONPath = soPath + ".treebeard-globals.json"
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.csv")
  return RunSingleTest_Multibatch(soPath, globalsJSONPath, csvPath, returnType)

def RunTestOnSingleModelTestInputs_Multibatch(modelName : str, returnType=numpy.float32) -> Boolean:
  print("Running multibatch actual input test",  modelName, "...", end=" ")
  soPath = os.path.join(os.path.join(treebeard_repo_dir, "runtime_test_binaries"), modelName + "_t8_b200_f_i16_invert.so")
  globalsJSONPath = soPath + ".treebeard-globals.json"
  csvPath = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), modelName + "_xgb_model_save.json.test.sampled.csv")
  return RunSingleTest_Multibatch(soPath, globalsJSONPath, csvPath, returnType)

assert RunTestOnSingleModelRandomInputs_Multibatch("abalone")
assert RunTestOnSingleModelRandomInputs_Multibatch("airline")
assert RunTestOnSingleModelRandomInputs_Multibatch("airline-ohe")
assert RunTestOnSingleModelRandomInputs_Multibatch("bosch")
assert RunTestOnSingleModelRandomInputs_Multibatch("covtype", numpy.int8)
assert RunTestOnSingleModelRandomInputs_Multibatch("epsilon")
assert RunTestOnSingleModelRandomInputs_Multibatch("higgs")
assert RunTestOnSingleModelRandomInputs_Multibatch("year_prediction_msd")

assert RunTestOnSingleModelRandomInputs("abalone")
assert RunTestOnSingleModelRandomInputs("airline")
assert RunTestOnSingleModelRandomInputs("airline-ohe")
assert RunTestOnSingleModelRandomInputs("bosch")
assert RunTestOnSingleModelRandomInputs("covtype", numpy.int8)
assert RunTestOnSingleModelRandomInputs("epsilon")
assert RunTestOnSingleModelRandomInputs("higgs")
assert RunTestOnSingleModelRandomInputs("year_prediction_msd")

assert RunTestOnSingleModelTestInputs_Multibatch("abalone")
assert RunTestOnSingleModelTestInputs_Multibatch("airline")
assert RunTestOnSingleModelTestInputs_Multibatch("airline-ohe")
assert RunTestOnSingleModelTestInputs_Multibatch("covtype", numpy.int8)
assert RunTestOnSingleModelTestInputs_Multibatch("epsilon")
assert RunTestOnSingleModelTestInputs_Multibatch("higgs")
assert RunTestOnSingleModelTestInputs_Multibatch("letters", numpy.int8)
assert RunTestOnSingleModelTestInputs_Multibatch("year_prediction_msd")

assert RunTestOnSingleModelTestInputs("abalone")
assert RunTestOnSingleModelTestInputs("airline")
assert RunTestOnSingleModelTestInputs("airline-ohe")
assert RunTestOnSingleModelTestInputs("covtype", numpy.int8)
assert RunTestOnSingleModelTestInputs("epsilon")
assert RunTestOnSingleModelTestInputs("higgs")
assert RunTestOnSingleModelTestInputs("letters", numpy.int8)
assert RunTestOnSingleModelTestInputs("year_prediction_msd")
