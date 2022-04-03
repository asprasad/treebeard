import os
import ctypes
from statistics import mode
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

      self.runtime_lib.CreateCompilerOptions.argtypes = None
      self.runtime_lib.CreateCompilerOptions.restype = ctypes.c_int64

      self.runtime_lib.DeleteCompilerOptions.argtypes = [ctypes.c_int64]
      self.runtime_lib.DeleteCompilerOptions.restype = None

      self.runtime_lib.Set_batchSize.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_batchSize.restype = None

      self.runtime_lib.Set_tileSize.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_tileSize.restype = None

      self.runtime_lib.Set_thresholdTypeWidth.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_thresholdTypeWidth.restype = None

      self.runtime_lib.Set_returnTypeWidth.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_returnTypeWidth.restype = None

      self.runtime_lib.Set_returnTypeFloatType.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_returnTypeFloatType.restype = None

      self.runtime_lib.Set_featureIndexTypeWidth.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_featureIndexTypeWidth.restype = None

      self.runtime_lib.Set_nodeIndexTypeWidth.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_nodeIndexTypeWidth.restype = None
      
      self.runtime_lib.Set_inputElementTypeWidth.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_inputElementTypeWidth.restype = None

      self.runtime_lib.Set_tileShapeBitWidth.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_tileShapeBitWidth.restype = None

      self.runtime_lib.Set_childIndexBitWidth.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_childIndexBitWidth.restype = None

      self.runtime_lib.Set_makeAllLeavesSameDepth.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_makeAllLeavesSameDepth.restype = None

      self.runtime_lib.Set_reorderTreesByDepth.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_reorderTreesByDepth.restype = None

      self.runtime_lib.Set_tilingType.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_tilingType.restype = None

      self.runtime_lib.GenerateLLVMIRForXGBoostModel.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64)
      self.runtime_lib.GenerateLLVMIRForXGBoostModel.restype = None
 
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

class CompilerOptions:
  def __init__(self, batchSize, tileSize) -> None:
    self.optionsPtr = treebeardAPI.runtime_lib.CreateCompilerOptions()
    treebeardAPI.runtime_lib.Set_batchSize(self.optionsPtr, batchSize)
    treebeardAPI.runtime_lib.Set_tileSize(self.optionsPtr, tileSize)

  def __del__(self):
    treebeardAPI.runtime_lib.DeleteCompilerOptions(self.optionsPtr)


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
  
def GenerateLLVMIRForXGBoostModel(modelJSONPathStr, llvmIRPathStr, modelGlobalsJSONPathStr, options):
  modelJSONPath = modelJSONPathStr.encode('ascii')
  llvmIRPath = llvmIRPathStr.encode('ascii')
  modelGlobalsJSONPath = modelGlobalsJSONPathStr.encode('ascii')
  treebeardAPI.runtime_lib.GenerateLLVMIRForXGBoostModel(ctypes.c_char_p(modelJSONPath), ctypes.c_char_p(llvmIRPath), ctypes.c_char_p(modelGlobalsJSONPath), options.optionsPtr)
