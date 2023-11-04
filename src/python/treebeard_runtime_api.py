import os
import ctypes

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

      self.runtime_lib.CreateInferenceRunner.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64)
      self.runtime_lib.CreateInferenceRunner.restype = ctypes.c_int64

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

      self.runtime_lib.Set_pipelineSize.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_pipelineSize.restype = None

      self.runtime_lib.Set_numberOfCores.argtypes = [ctypes.c_int64, ctypes.c_int32]
      self.runtime_lib.Set_numberOfCores.restype = None

      self.runtime_lib.Set_statsProfileCSVPath.argtypes = [ctypes.c_int64, ctypes.c_char_p]
      self.runtime_lib.Set_statsProfileCSVPath.restype = None

      self.runtime_lib.SetOneTreeAtATimeSchedule.argtypes = [ctypes.c_int64]
      self.runtime_lib.SetOneTreeAtATimeSchedule.restype = None

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
      self.runtime_lib.GetSpecializationInfoEntries.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
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

      self.runtime_lib.DestroyTreebeardContext.argTypes = [ctypes.c_int64]

      self.runtime_lib.SetForestCreatorType.argTypes = [ctypes.c_int64, ctypes.c_char_p]

      self.runtime_lib.SetRepresentationAndSerializer.argTypes = [ctypes.c_int64, ctypes.c_char_p]

      self.runtime_lib.GetScheduleFromTBContext.restype = ctypes.c_int64
      self.runtime_lib.GetScheduleFromTBContext.argtypes = [ctypes.c_int64]

      self.runtime_lib.BuildHIRRepresentation.argtypes = [ctypes.c_int64]

      self.runtime_lib.LowerToLLVMAndDumpIR.restype = ctypes.c_bool
      self.runtime_lib.LowerToLLVMAndDumpIR.argtypes = [ctypes.c_int64, ctypes.c_char_p]

      self.runtime_lib.ConstructInferenceRunnerFromHIR.restype = ctypes.c_int64
      self.runtime_lib.ConstructInferenceRunnerFromHIR.argtypes = [ctypes.c_int64]

      # extern "C" void *ConstructGPUInferenceRunnerFromHIR(void *tbContext)
      self.runtime_lib.ConstructGPUInferenceRunnerFromHIR.restype = ctypes.c_int64
      self.runtime_lib.ConstructGPUInferenceRunnerFromHIR.argtypes = [ctypes.c_int64]

      # void IndexVariable_SetGPUThreadDim(intptr_t indexVarPtr, int32_t construct, int32_t dim)
      self.runtime_lib.IndexVariable_SetGPUThreadDim.restype = None
      self.runtime_lib.IndexVariable_SetGPUThreadDim.argtypes = [ctypes.c_int64, ctypes.c_int32, ctypes.c_int32]

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
      self.runtime_lib.Schedule_AtomicReduce(ctypes.c_int64(schedPtr), ctypes.c_int64(indexVarPtr))
  
  def Schedule_VectorReduce(self, schedPtr, indexVarPtr, vectorWidth):
      self.runtime_lib.Schedule_VectorReduce(ctypes.c_int64(schedPtr), ctypes.c_int64(indexVarPtr), ctypes.c_int32(vectorWidth))
  
  def Schedule_SharedReduce(self, schedPtr, indexVarPtr):
      self.runtime_lib.Schedule_SharedReduce(ctypes.c_int64(schedPtr), ctypes.c_int64(indexVarPtr))

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
    # print("TBContext_Create: ", tbContext, type(tbContext))
    return tbContext

  def DestroyTreebeardContext(self, treebeard_context_ptr):
    return self.runtime_lib.DestroyTreebeardContext(ctypes.c_int64(treebeard_context_ptr))

  def SetForestCreatorType(self, treebeard_context_ptr, file_type):
    file_type_ascii = file_type.encode('ascii')
    # print("TBContext_SetTypeAPI: ", treebeard_context_ptr, type(treebeard_context_ptr))
    self.runtime_lib.SetForestCreatorType(ctypes.c_int64(treebeard_context_ptr), file_type_ascii)

  def LowerToLLVMAndDumpIR(self, treebeard_context_ptr, output_path):
    output_path_utf8 = output_path.encode('utf-8')
    self.runtime_lib.LowerToLLVMAndDumpIR(treebeard_context_ptr, output_path_utf8)

  def SetRepresentationAndSerializer(self, treebeard_context_ptr, rep_type):
    rep_type_ascii = rep_type.encode('ascii')
    self.runtime_lib.SetRepresentationAndSerializer(ctypes.c_int64(treebeard_context_ptr), rep_type_ascii)

  def GetScheduleFromTBContext(self, treebeard_context_ptr):
    schedule_ptr = self.runtime_lib.GetScheduleFromTBContext(treebeard_context_ptr)
    return ctypes.c_int64(schedule_ptr)
