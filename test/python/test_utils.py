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

def RunSingleGPUTestJIT(modelJSONPath, 
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
  inferenceRunner = tbContext.ConstructGPUInferenceRunnerFromHIR()
  
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

def RunSingleGPUTestAutoSchedule(modelJSONPath, 
                                 csvPath,
                                 options,
                                 returnType,
                                 representation,
                                 inputType,
                                 gpuAutoScheduleOptions: treebeard.GPUAutoScheduleOptions) -> bool:
  data_df = pandas.read_csv(csvPath, header=None)
  data = numpy.array(data_df, order='C') # numpy.genfromtxt(csvPath, ',')
  inputs = numpy.array(data[:, :-1], numpy.float32, order='C')
  expectedOutputs = data[:, data.shape[1]-1]
  
  globalsPath = modelJSONPath + ".treebeard-globals.json"
  tbContext = treebeard.TreebeardContext(modelJSONPath, globalsPath, options, gpuAutoScheduleOptions=gpuAutoScheduleOptions)
  tbContext.SetRepresentationType(representation)
  tbContext.SetInputFiletype(inputType)

  inferenceRunner = treebeard.TreebeardInferenceRunner.GPUInferenceRunnerFromTBContext(tbContext)
  
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

def RunSingleGPUTestAutoTuneHeuristic(modelJSONPath, 
                                      csvPath,
                                      options,
                                      returnType,
                                      inputType) -> bool:
  data_df = pandas.read_csv(csvPath, header=None)
  data = numpy.array(data_df, order='C') # numpy.genfromtxt(csvPath, ',')
  inputs = numpy.array(data[:, :-1], numpy.float32, order='C')
  expectedOutputs = data[:, data.shape[1]-1]
  
  globalsPath = modelJSONPath + ".treebeard-globals.json"
  tbContext = treebeard.TreebeardContext(modelJSONPath, globalsPath, options)
  tbContext.SetInputFiletype(inputType)

  inferenceRunner = treebeard.TreebeardInferenceRunner.AutoScheduleGPUInferenceRunnerFromTBContext(tbContext)
  
  batch_size = 256
  start = time.time()
  for i in range(batch_size, data.shape[0], batch_size):
    batch = inputs[i-batch_size:i, :]
    results = inferenceRunner.RunInference(batch, returnType)
    if not CheckArraysEqual(results, expectedOutputs[i - batch_size : i]):
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

def RunTestOnSingleModelTestInputsJITIfSelected(modelName : str, options, testName : str, returnType=numpy.float32, testFunc=RunSingleTestJIT, selectedTests=[]) -> bool:
  if modelName in selectedTests:
    return RunTestOnSingleModelTestInputsJIT(modelName, options, testName, returnType, testFunc)
  else:
    return True
  
def RunSelectedTests(testName, options, multiclassOptions, singleTestRunner, selectedTests):
  assert RunTestOnSingleModelTestInputsJITIfSelected("abalone", options, testName, numpy.float32, singleTestRunner, selectedTests)
  assert RunTestOnSingleModelTestInputsJITIfSelected("airline", options, testName, numpy.float32, singleTestRunner, selectedTests)
  assert RunTestOnSingleModelTestInputsJITIfSelected("airline-ohe", options, testName, numpy.float32, singleTestRunner, selectedTests)
  assert RunTestOnSingleModelTestInputsJITIfSelected("covtype", multiclassOptions, testName, numpy.int8, singleTestRunner, selectedTests)
  assert RunTestOnSingleModelTestInputsJITIfSelected("epsilon", options, testName, numpy.float32, singleTestRunner, selectedTests)
  assert RunTestOnSingleModelTestInputsJITIfSelected("higgs", options, testName, numpy.float32, singleTestRunner, selectedTests)
  assert RunTestOnSingleModelTestInputsJITIfSelected("letters", multiclassOptions, testName, numpy.int8, singleTestRunner, selectedTests)
  assert RunTestOnSingleModelTestInputsJITIfSelected("year_prediction_msd", options, testName, numpy.float32, singleTestRunner, selectedTests)

def RunAllTests(testName, options, multiclassOptions, singleTestRunner):
  assert RunTestOnSingleModelTestInputsJIT("abalone", options, testName, numpy.float32, singleTestRunner)
  assert RunTestOnSingleModelTestInputsJIT("airline", options, testName, numpy.float32, singleTestRunner)
  assert RunTestOnSingleModelTestInputsJIT("airline-ohe", options, testName, numpy.float32, singleTestRunner)
  assert RunTestOnSingleModelTestInputsJIT("covtype", multiclassOptions, testName, numpy.int8, singleTestRunner)
  assert RunTestOnSingleModelTestInputsJIT("epsilon", options, testName, numpy.float32, singleTestRunner)
  assert RunTestOnSingleModelTestInputsJIT("higgs", options, testName, numpy.float32, singleTestRunner)
  assert RunTestOnSingleModelTestInputsJIT("letters", multiclassOptions, testName, numpy.int8, singleTestRunner)
  assert RunTestOnSingleModelTestInputsJIT("year_prediction_msd", options, testName, numpy.float32, singleTestRunner)
