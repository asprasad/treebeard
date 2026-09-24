import os
import ctypes
from enum import IntEnum


class TreebeardErrorCode(IntEnum):
  SUCCESS = 0
  UNSUPPORTED_MODEL = 1
  UNSUPPORTED_CONFIGURATION = 2
  COMPILATION = 3
  RUNTIME = 4


class TreebeardNativeError(RuntimeError):
  def __init__(self, code, message):
    super().__init__(message)
    self.code = code

filepath = os.path.abspath(__file__)
treebeard_runtime_dir = os.path.dirname(filepath)
treebeard_runtime_so_name = "libtreebeard-runtime.so"
treebeard_runtime_path = os.path.join(treebeard_runtime_dir, treebeard_runtime_so_name)

if not os.path.exists(treebeard_runtime_path):
  raise ImportError(
      f"Treebeard native runtime was not found at {treebeard_runtime_path}")

class TreebeardAPI:
  def __init__(self) -> None:
    try:
      self.runtime_lib = ctypes.CDLL(treebeard_runtime_path)

      self.runtime_lib.TreebeardGetLastError.argtypes = None
      self.runtime_lib.TreebeardGetLastError.restype = ctypes.c_char_p
      self.runtime_lib.TreebeardGetLastErrorCode.argtypes = None
      self.runtime_lib.TreebeardGetLastErrorCode.restype = ctypes.c_int32

      self.runtime_lib.CreateInferenceRunner.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64)
      self.runtime_lib.CreateInferenceRunner.restype = ctypes.c_int64

      self.runtime_lib.CreateInferenceRunnerForONNXModel.argtypes = (ctypes.c_char_p, ctypes.c_int64)
      self.runtime_lib.CreateInferenceRunnerForONNXModel.restype = ctypes.c_int64

      self.runtime_lib.InitializeInferenceRunner.argtypes = (ctypes.c_char_p, ctypes.c_char_p)
      self.runtime_lib.InitializeInferenceRunner.restype = ctypes.c_int64
      
      self.runtime_lib.RunInference.argtypes = (ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p)
      self.runtime_lib.RunInference.restype = ctypes.c_bool

      self.runtime_lib.RunInferenceOnMultipleBatches.argtypes = (ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32)
      self.runtime_lib.RunInferenceOnMultipleBatches.restype = ctypes.c_bool
      
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

      self.runtime_lib.Set_pipelineSize.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_pipelineSize.restype = None

      self.runtime_lib.Set_numberOfFeatures.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_numberOfFeatures.restype = None

      self.runtime_lib.Set_numberOfCores.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_numberOfCores.restype = None

      self.runtime_lib.Set_numParallelTreeBatches.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_numParallelTreeBatches.restype = None

      self.runtime_lib.Set_statsProfileCSVPath.argtypes = [ctypes.c_int64, ctypes.c_char_p]
      self.runtime_lib.Set_statsProfileCSVPath.restype = None

      # self.runtime_lib.Set_compileToGPU.argtypes = [ctypes.c_int64, ctypes.c_bool]
      # self.runtime_lib.Set_compileToGPU.restype = None

      self.runtime_lib.SetOneTreeAtATimeSchedule.argtypes = [ctypes.c_int64]
      self.runtime_lib.SetOneTreeAtATimeSchedule.restype = None

      # self.runtime_lib.SetBasicGPUSchedule.argtypes = [ctypes.c_int64, ctypes.c_int32]
      # self.runtime_lib.SetBasicGPUSchedule.restype = None

      self.runtime_lib.GenerateLLVMIRForXGBoostModel.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64)
      self.runtime_lib.GenerateLLVMIRForXGBoostModel.restype = None

      self.runtime_lib.SetEnableSparseRepresentation.argtypes = [ctypes.c_int32]
      self.runtime_lib.SetEnableSparseRepresentation.restype = None

      self.runtime_lib.IsSparseRepresentationEnabled.argtypes = None
      self.runtime_lib.IsSparseRepresentationEnabled.restype = ctypes.c_int32

      self.runtime_lib.SetPeeledCodeGenForProbabilityBasedTiling.argtypes = [ctypes.c_int32]
      self.runtime_lib.SetPeeledCodeGenForProbabilityBasedTiling.restype = None

      self.runtime_lib.IsPeeledCodeGenForProbabilityBasedTilingEnabled.argtypes = None
      self.runtime_lib.IsPeeledCodeGenForProbabilityBasedTilingEnabled.restype = ctypes.c_int32

      self.runtime_lib.Schedule_NewIndexVariable.argtypes = [ctypes.c_int64, ctypes.c_char_p]
      self.runtime_lib.Schedule_NewIndexVariable.restype = ctypes.c_int64

      self.runtime_lib.Schedule_NewIndexVariable2.argtypes = [ctypes.c_int64, ctypes.c_int64]
      self.runtime_lib.Schedule_NewIndexVariable2.restype = ctypes.c_int64

      self.runtime_lib.Schedule_Tile.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Schedule_Tile.restype = None

      self.runtime_lib.Schedule_Reorder.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_int32]
      self.runtime_lib.Schedule_Reorder.restype = None

      self.runtime_lib.Schedule_Split.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int64]
      self.runtime_lib.Schedule_Split.restype = None

      # intptr_t Schedule_Specialize(intptr_t schedPtr, intptr_t indexPtr) {
      self.runtime_lib.Schedule_Specialize.argtypes = [ctypes.c_int64, ctypes.c_int64]
      self.runtime_lib.Schedule_Specialize.restype = ctypes.c_int64

      # int64_t GetSpecializationInfoNumIterations(intptr_t infoPtr) {
      self.runtime_lib.GetSpecializationInfoNumIterations.argtypes = [ctypes.c_int64]
      self.runtime_lib.GetSpecializationInfoNumIterations.restype = ctypes.c_int64

      # int64_t GetSpecializationInfoNumEntries(intptr_t infoPtr) {
      self.runtime_lib.GetSpecializationInfoNumEntries.argtypes = [ctypes.c_int64]
      self.runtime_lib.GetSpecializationInfoNumEntries.restype = ctypes.c_int64

      # void GetSpecializationInfoEntries(intptr_t infoPtr, intptr_t lengths, intptr_t indices)
      self.runtime_lib.GetSpecializationInfoEntries.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
      self.runtime_lib.GetSpecializationInfoEntries.restype = None

      self.runtime_lib.Schedule_Pipeline.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Schedule_Pipeline.restype = None

      self.runtime_lib.Schedule_Simdize.argtypes = [ctypes.c_int64, ctypes.c_int64]

      self.runtime_lib.Schedule_Parallel.argtypes = [ctypes.c_int64, ctypes.c_int64]

      self.runtime_lib.Schedule_Unroll.argtypes = [ctypes.c_int64, ctypes.c_int64]

      self.runtime_lib.Schedule_PeelWalk.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32]

      self.runtime_lib.Schedule_Cache.argtypes = [ctypes.c_int64, ctypes.c_int64]
      
      # void Schedule_AtomicReduce(intptr_t schedPtr, intptr_t indexVarPtr)
      self.runtime_lib.Schedule_AtomicReduce.argtypes = [ctypes.c_int64, ctypes.c_int64]

      # void Schedule_VectorReduce(intptr_t schedPtr, intptr_t indexVarPtr
      self.runtime_lib.Schedule_VectorReduce.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32]

      # void Schedule_SharedReduce(intptr_t schedPtr, intptr_t indexVarPtr)
      self.runtime_lib.Schedule_SharedReduce.argtypes = [ctypes.c_int64, ctypes.c_int64]

      self.runtime_lib.Schedule_GetRootIndex.restype = ctypes.c_int64
      self.runtime_lib.Schedule_GetRootIndex.argtypes = [ctypes.c_int64]

      self.runtime_lib.Schedule_GetBatchIndex.restype = ctypes.c_int64
      self.runtime_lib.Schedule_GetBatchIndex.argtypes = [ctypes.c_int64]

      self.runtime_lib.Schedule_GetTreeIndex.argtypes = [ctypes.c_int64]
      self.runtime_lib.Schedule_GetTreeIndex.restype = ctypes.c_int64

      self.runtime_lib.Schedule_PrintToString.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_int32]
      self.runtime_lib.Schedule_PrintToString.restype = ctypes.c_int32

      self.runtime_lib.Schedule_GetBatchSize.argtypes = [ctypes.c_int64]
      self.runtime_lib.Schedule_GetBatchSize.restype = ctypes.c_int32

      self.runtime_lib.Schedule_GetForestSize.argtypes = [ctypes.c_int64]
      self.runtime_lib.Schedule_GetForestSize.restype = ctypes.c_int32

      self.runtime_lib.Schedule_IsDefaultSchedule.argtypes = [ctypes.c_int64]
      self.runtime_lib.Schedule_IsDefaultSchedule.restype = ctypes.c_bool

      self.runtime_lib.Schedule_Finalize.argtypes = [ctypes.c_int64]
      self.runtime_lib.Schedule_Finalize.restype = None

      self.runtime_lib.ConstructRepresentation.argtypes = [ctypes.c_char_p]
      self.runtime_lib.ConstructRepresentation.restype = ctypes.c_void_p

      self.runtime_lib.DestroyRepresentation.argtypes = [ctypes.c_void_p]
      
      self.runtime_lib.ConstructTreebeardContext.restype = ctypes.c_int64
      self.runtime_lib.ConstructTreebeardContext.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64]

      self.gpu_support = hasattr(self.runtime_lib, "ConstructGPUInferenceRunnerFromHIR")
      if self.gpu_support:
        self.runtime_lib.ConstructTreebeardContextFromGPUAutoscheduleOptions.restype = ctypes.c_int64
        self.runtime_lib.ConstructTreebeardContextFromGPUAutoscheduleOptions.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64]

      self.runtime_lib.DestroyTreebeardContext.argtypes = [ctypes.c_int64]
      self.runtime_lib.DestroyTreebeardContext.restype = None

      self.runtime_lib.SetForestCreatorType.argtypes = [ctypes.c_int64, ctypes.c_char_p]
      self.runtime_lib.SetForestCreatorType.restype = ctypes.c_bool

      self.runtime_lib.SetRepresentationAndSerializer.argtypes = [ctypes.c_int64, ctypes.c_char_p]
      self.runtime_lib.SetRepresentationAndSerializer.restype = ctypes.c_bool

      self.runtime_lib.GetScheduleFromTBContext.restype = ctypes.c_int64
      self.runtime_lib.GetScheduleFromTBContext.argtypes = [ctypes.c_int64]

      self.runtime_lib.BuildHIRRepresentation.argtypes = [ctypes.c_int64]
      self.runtime_lib.BuildHIRRepresentation.restype = ctypes.c_bool

      self.runtime_lib.LowerToLLVMAndDumpIR.restype = ctypes.c_bool
      self.runtime_lib.LowerToLLVMAndDumpIR.argtypes = [ctypes.c_int64, ctypes.c_char_p]

      self.runtime_lib.ConstructInferenceRunnerFromHIR.restype = ctypes.c_int64
      self.runtime_lib.ConstructInferenceRunnerFromHIR.argtypes = [ctypes.c_int64]

      if self.gpu_support:
        self.runtime_lib.ConstructGPUInferenceRunnerFromHIR.restype = ctypes.c_int64
        self.runtime_lib.ConstructGPUInferenceRunnerFromHIR.argtypes = [ctypes.c_int64]

        self.runtime_lib.ConstructGPUInferenceRunnerFromTBContext.restype = ctypes.c_int64
        self.runtime_lib.ConstructGPUInferenceRunnerFromTBContext.argtypes = [ctypes.c_int64]

      # void IndexVariable_SetGPUThreadDim(intptr_t indexVarPtr, int32_t construct, int32_t dim)
      self.runtime_lib.IndexVariable_SetGPUThreadDim.restype = None
      self.runtime_lib.IndexVariable_SetGPUThreadDim.argtypes = [ctypes.c_int64, ctypes.c_int32, ctypes.c_int32]

      if self.gpu_support:
        self.runtime_lib.CreateGPUAutoScheduleOptions.restype = ctypes.c_int64
        self.runtime_lib.CreateGPUAutoScheduleOptions.argtypes = None

        self.runtime_lib.DeleteGPUAutoScheduleOptions.restype = None
        self.runtime_lib.DeleteGPUAutoScheduleOptions.argtypes = [ctypes.c_int64]

        self.runtime_lib.findBestGPUScheduleAndCompileModel.restype = ctypes.c_int64
        self.runtime_lib.findBestGPUScheduleAndCompileModel.argtypes = [ctypes.c_int64]

        self.runtime_lib.Set_numRowsPerTB.argtypes = [ctypes.c_int64, ctypes.c_int32]
        self.runtime_lib.Set_numRowsPerThread.argtypes = [ctypes.c_int64, ctypes.c_int32]
        self.runtime_lib.Set_rowTileSize.argtypes = [ctypes.c_int64, ctypes.c_int32]
        self.runtime_lib.Set_numTreeThreads.argtypes = [ctypes.c_int64, ctypes.c_int32]
        self.runtime_lib.Set_numTreesAtATime.argtypes = [ctypes.c_int64, ctypes.c_int32]
        self.runtime_lib.Set_cacheRows.argtypes = [ctypes.c_int64, ctypes.c_int32]
        self.runtime_lib.Set_cacheTrees.argtypes = [ctypes.c_int64, ctypes.c_int32]
        self.runtime_lib.Set_unrollTreeWalks.argtypes = [ctypes.c_int64, ctypes.c_int32]
        self.runtime_lib.Set_treeWalkInterleaveFactor.argtypes = [ctypes.c_int64, ctypes.c_int32]
        self.runtime_lib.Set_sharedMemoryReduce.argtypes = [ctypes.c_int64, ctypes.c_int32]

        self.runtime_lib.Set_numRowsPerTB.restype = None
        self.runtime_lib.Set_numRowsPerThread.restype = None
        self.runtime_lib.Set_rowTileSize.restype = None
        self.runtime_lib.Set_numTreeThreads.restype = None
        self.runtime_lib.Set_numTreesAtATime.restype = None
        self.runtime_lib.Set_cacheRows.restype = None
        self.runtime_lib.Set_cacheTrees.restype = None
        self.runtime_lib.Set_unrollTreeWalks.restype = None
        self.runtime_lib.Set_treeWalkInterleaveFactor.restype = None
        self.runtime_lib.Set_sharedMemoryReduce.restype = None

        self.runtime_lib.SetEnableMeasureGpuKernelTime.argtypes = [ctypes.c_int8]
        self.runtime_lib.SetEnableMeasureGpuKernelTime.restype = None

        self.runtime_lib.GetGPUKernelExecutionTime.argtypes = [ctypes.c_int64]
        self.runtime_lib.GetGPUKernelExecutionTime.restype = ctypes.c_int64

    except Exception as e:
      raise ImportError(f"loading the Treebeard runtime failed: {e}") from e

  def _raise_last_error(self):
    message = self.runtime_lib.TreebeardGetLastError()
    if message:
      message = message.decode("utf-8", errors="replace")
    else:
      message = "native Treebeard operation failed"
    raise TreebeardNativeError(
        int(self.runtime_lib.TreebeardGetLastErrorCode()), message)
  
  def InitializeInferenceRunner(self, modelSOPath : str, modelGlobalsJSONPath : str) -> int:
    soPath = modelSOPath.encode('ascii')
    globalsJSONPath = modelGlobalsJSONPath.encode('ascii')
    return int(self.runtime_lib.InitializeInferenceRunner(ctypes.c_char_p(soPath), ctypes.c_char_p(globalsJSONPath)))
  
  def GetRowSize(self, inferenceRunner : int) -> int:
    return int(self.runtime_lib.GetRowSize(inferenceRunner))

  def GetBatchSize(self, inferenceRunner : int) -> int:
    return int(self.runtime_lib.GetBatchSize(inferenceRunner))
  
  def RunInference(self, inferenceRunner : int, inputs : ctypes.c_void_p, results : ctypes.c_void_p) -> None:
    if not self.runtime_lib.RunInference(inferenceRunner, inputs, results):
      self._raise_last_error()

  def RunInferenceOnMultipleBatches(self, inferenceRunner : int, inputs : ctypes.c_void_p, results : ctypes.c_void_p, numRows : int) -> None:
    if not self.runtime_lib.RunInferenceOnMultipleBatches(
        inferenceRunner, inputs, results, numRows):
      self._raise_last_error()

  def DeleteInferenceRunner(self, inferenceRunner : int) -> None:
    self.runtime_lib.DeleteInferenceRunner(inferenceRunner)

  def Schedule_NewIndexVariable(self, schedPtr, name):
      return self.runtime_lib.Schedule_NewIndexVariable(schedPtr, name.encode('utf-8'))

  def Schedule_NewIndexVariable2(self, schedPtr, indexVarPtr):
      return self.runtime_lib.Schedule_NewIndexVariable2(schedPtr, indexVarPtr)

  def Schedule_Tile(self, schedPtr, indexPtr, outerPtr, innerPtr, tileSize):
      self.runtime_lib.Schedule_Tile(schedPtr, indexPtr, outerPtr, innerPtr, tileSize)

  def Schedule_Reorder(self, schedPtr, indicesPtr, numIndices):
      self.runtime_lib.Schedule_Reorder(schedPtr, indicesPtr, numIndices)

  def Schedule_Split(self, schedPtr, indexPtr, firstPtr, secondPtr, splitIteration, indexMapPtr):
      self.runtime_lib.Schedule_Split(schedPtr, indexPtr, firstPtr, secondPtr, splitIteration, indexMapPtr)

  def Schedule_Pipeline(self, schedPtr, indexPtr, stepSize):
      self.runtime_lib.Schedule_Pipeline(schedPtr, indexPtr, stepSize)
  
  def Schedule_Simdize(self, schedPtr, indexPtr):
      self.runtime_lib.Schedule_Simdize(schedPtr, ctypes.c_int64(indexPtr))

  def Schedule_Parallel(self, schedPtr, indexPtr):
      self.runtime_lib.Schedule_Parallel(schedPtr, ctypes.c_int64(indexPtr))

  def Schedule_Unroll(self, schedPtr, indexVarPtr):
      self.runtime_lib.Schedule_Unroll(schedPtr, ctypes.c_int64(indexVarPtr))

  def Schedule_PeelWalk(self, schedPtr, indexVarPtr, numberOfIterations):
      self.runtime_lib.Schedule_PeelWalk(ctypes.c_int64(schedPtr), ctypes.c_int64(indexVarPtr), ctypes.c_int32(numberOfIterations))

  def Schedule_Cache(self, schedPtr, indexVarPtr):
      self.runtime_lib.Schedule_Cache(schedPtr, ctypes.c_int64(indexVarPtr))
  
  def Schedule_AtomicReduce(self, schedPtr, indexVarPtr):
      self.runtime_lib.Schedule_AtomicReduce(schedPtr, ctypes.c_int64(indexVarPtr))
  
  def Schedule_VectorReduce(self, schedPtr, indexVarPtr, vectorWidth):
      self.runtime_lib.Schedule_VectorReduce(schedPtr, ctypes.c_int64(indexVarPtr), ctypes.c_int32(vectorWidth))
  
  def Schedule_SharedReduce(self, schedPtr, indexVarPtr):
      self.runtime_lib.Schedule_SharedReduce(schedPtr, ctypes.c_int64(indexVarPtr))

  def Schedule_GetRootIndex(self, schedPtr):
      return self.runtime_lib.Schedule_GetRootIndex(schedPtr)

  def Schedule_GetBatchIndex(self, schedPtr):
      return self.runtime_lib.Schedule_GetBatchIndex(schedPtr)

  def Schedule_GetTreeIndex(self, schedPtr):
      return self.runtime_lib.Schedule_GetTreeIndex(schedPtr)

  def Schedule_PrintToString(self, schedPtr, string, strLen):
      return self.runtime_lib.Schedule_PrintToString(schedPtr, string, strLen)

  def Schedule_GetBatchSize(self, schedPtr):
      return self.runtime_lib.Schedule_GetBatchSize(schedPtr)

  def Schedule_GetForestSize(self, schedPtr):
      return self.runtime_lib.Schedule_GetForestSize(schedPtr)

  def Schedule_IsDefaultSchedule(self, schedPtr):
      return self.runtime_lib.Schedule_IsDefaultSchedule(schedPtr)

  def Schedule_Finalize(self, schedPtr):
      self.runtime_lib.Schedule_Finalize(schedPtr)
  
  def ConstructTreebeardContext(self, model_path, model_globals_path, options_ptr):
    model_path_ascii = model_path.encode('ascii')
    model_globals_path_ascii = model_globals_path.encode('ascii')
    tbContext = self.runtime_lib.ConstructTreebeardContext(model_path_ascii, model_globals_path_ascii, options_ptr)
    if not tbContext:
      self._raise_last_error()
    return tbContext
  
  def ConstructTreebeardContextFromGPUAutoscheduleOptions(self, model_path, model_globals_path, options_ptr, gpu_autoschedule_options_ptr):
    model_path_ascii = model_path.encode('ascii')
    model_globals_path_ascii = model_globals_path.encode('ascii')
    tbContext = self.runtime_lib.ConstructTreebeardContextFromGPUAutoscheduleOptions(model_path_ascii, model_globals_path_ascii, options_ptr, gpu_autoschedule_options_ptr)
    # print("TBContext_Create: ", tbContext, type(tbContext))
    return tbContext

  def DestroyTreebeardContext(self, treebeard_context_ptr):
    return self.runtime_lib.DestroyTreebeardContext(ctypes.c_int64(treebeard_context_ptr))

  def SetForestCreatorType(self, treebeard_context_ptr, file_type):
    file_type_ascii = file_type.encode('ascii')
    if not self.runtime_lib.SetForestCreatorType(
        ctypes.c_int64(treebeard_context_ptr), file_type_ascii):
      self._raise_last_error()

  def LowerToLLVMAndDumpIR(self, treebeard_context_ptr, output_path):
    output_path_utf8 = output_path.encode('utf-8')
    self.runtime_lib.LowerToLLVMAndDumpIR(treebeard_context_ptr, output_path_utf8)

  def SetRepresentationAndSerializer(self, treebeard_context_ptr, rep_type):
    rep_type_ascii = rep_type.encode('ascii')
    if not self.runtime_lib.SetRepresentationAndSerializer(
        ctypes.c_int64(treebeard_context_ptr), rep_type_ascii):
      self._raise_last_error()

  def GetScheduleFromTBContext(self, treebeard_context_ptr):
    schedule_ptr = self.runtime_lib.GetScheduleFromTBContext(treebeard_context_ptr)
    if not schedule_ptr:
      self._raise_last_error()
    return ctypes.c_int64(schedule_ptr)

  def BuildHIRRepresentation(self, treebeard_context_ptr):
    if not self.runtime_lib.BuildHIRRepresentation(treebeard_context_ptr):
      self._raise_last_error()

  def ConstructInferenceRunnerFromHIR(self, treebeard_context_ptr):
    runner = self.runtime_lib.ConstructInferenceRunnerFromHIR(
        treebeard_context_ptr)
    if not runner:
      self._raise_last_error()
    return int(runner)
