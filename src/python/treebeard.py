import os
import ctypes
import numpy

filepath = os.path.abspath(__file__)
treebeard_runtime_dir = os.path.dirname(filepath)
treebeard_runtime_so_name = "libtreebeard-runtime.so"
treebeard_runtime_path = os.path.join(treebeard_runtime_dir, treebeard_runtime_so_name)

# We expect the runtime so and this python file to be in the same directory
assert os.path.exists(treebeard_runtime_path)

class TreebeardAPI:
  def __init__(self) -> None:
    try:
      self.runtime_lib = ctypes.CDLL(treebeard_runtime_path)
      
      self.runtime_lib.InitializeInferenceRunner.argtypes = (ctypes.c_char_p, ctypes.c_char_p)
      self.runtime_lib.InitializeInferenceRunner.restype = ctypes.c_int64
      
      self.runtime_lib.RunInference.argtypes = (ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p)
      self.runtime_lib.RunInference.restype = None

      self.runtime_lib.RunInferenceOnMultipleBatches.argtypes = (ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32)
      self.runtime_lib.RunInferenceOnMultipleBatches.restype = None
      
      self.runtime_lib.GetBatchSize.argtypes = [ctypes.c_int64]
      self.runtime_lib.GetBatchSize.restype = ctypes.c_int32

      self.runtime_lib.GetRowSize.argtypes = [ctypes.c_int64]
      self.runtime_lib.GetRowSize.restype = ctypes.c_int32

      self.runtime_lib.DeleteInferenceRunner.argtypes = [ctypes.c_int64]
      self.runtime_lib.DeleteInferenceRunner.restype = None
    except Exception as e:
      print("Loading the TreeBeard runtime failed with exception :", e)
  
  def InitializeInferenceRunner(self, modelSOPath : str, modelGlobalsJSONPath : str) -> int:
    soPath = modelSOPath.encode('ascii')
    globalsJSONPath = modelGlobalsJSONPath.encode('ascii')
    return int(self.runtime_lib.InitializeInferenceRunner(ctypes.c_char_p(soPath), ctypes.c_char_p(globalsJSONPath)))
  
  def GetRowSize(self, inferenceRunner : int) -> int:
    return int(self.runtime_lib.GetRowSize(inferenceRunner))

  def GetBatchSize(self, inferenceRunner : int) -> int:
    return int(self.runtime_lib.GetBatchSize(inferenceRunner))
  
  def RunInference(self, inferenceRunner : int, inputs : ctypes.c_void_p, results : ctypes.c_void_p) -> None:
    self.runtime_lib.RunInference(inferenceRunner, inputs, results)

  def RunInferenceOnMultipleBatches(self, inferenceRunner : int, inputs : ctypes.c_void_p, results : ctypes.c_void_p, numRows : int) -> None:
    self.runtime_lib.RunInferenceOnMultipleBatches(inferenceRunner, inputs, results, numRows)

  def DeleteInferenceRunner(self, inferenceRunner : int) -> None:
    self.runtime_lib.DeleteInferenceRunner(inferenceRunner)

treebeardAPI = TreebeardAPI()

class TreebeardInferenceRunner:
  def __init__(self, modelSOPath : str, modelGlobalsJSONPath : str) -> None:
    self.treebeardAPI = treebeardAPI
    self.inferenceRunner = treebeardAPI.InitializeInferenceRunner(modelSOPath, modelGlobalsJSONPath)
    self.rowSize = treebeardAPI.GetRowSize(self.inferenceRunner)
    self.batchSize = treebeardAPI.GetBatchSize(self.inferenceRunner)

  def __del__(self):
    self.treebeardAPI.DeleteInferenceRunner(self.inferenceRunner)
  
  def RunInference(self, inputs, resultType=numpy.float32):
    assert type(inputs) is numpy.ndarray
    inputs_np = inputs
    results = numpy.zeros((self.batchSize), resultType)
    self.treebeardAPI.RunInference(self.inferenceRunner, inputs_np.ctypes.data_as(ctypes.c_void_p), results.ctypes.data_as(ctypes.c_void_p))
    return results

  def RunInferenceOnMultipleBatches(self, inputs, resultType=numpy.float32):
    assert type(inputs) is numpy.ndarray
    numRows = inputs.shape[0]
    results = numpy.zeros((numRows), resultType)
    self.treebeardAPI.RunInferenceOnMultipleBatches(self.inferenceRunner, inputs.ctypes.data_as(ctypes.c_void_p), results.ctypes.data_as(ctypes.c_void_p), numRows)
    return results
  
