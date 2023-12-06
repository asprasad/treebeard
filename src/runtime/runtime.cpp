#include <cstdint>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <json.hpp>
#include <string>
#include "DecisionForest.h"
#include "Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "tbruntime.h"
#include "ExecutionHelpers.h"
#include "CompileUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "xgboostparser.h"
#include "schedule.h"
#include "ModelSerializers.h"
#include "Representations.h"
#include "onnxmodelparser.h"

// ===-------------------------------------------------------------=== //
// Execution API
// ===-------------------------------------------------------------=== //

// Create a shared object inference runner and return an ID (Init)
//    -- SO name, globals JSON path 
extern "C" intptr_t InitializeInferenceRunner(const char* soPath, const char* modelGlobalsJSONPath) {
  using json = nlohmann::json;
  json globalsJSON;
  std::ifstream fin(modelGlobalsJSONPath);
  fin >> globalsJSON;

  auto tileSizeEntries = globalsJSON["TileSizeEntries"];
  assert (tileSizeEntries.size() == 1);
  int32_t tileSize = tileSizeEntries.front()["TileSize"];
  int32_t thresholdBitwidth = tileSizeEntries.front()["ThresholdBitWidth"];
  int32_t featureIndexBitwidth = tileSizeEntries.front()["FeatureIndexBitWidth"];
  auto serializer = mlir::decisionforest::ConstructModelSerializer(modelGlobalsJSONPath);
  auto inferenceRunner = new mlir::decisionforest::SharedObjectInferenceRunner(serializer, soPath, tileSize, 
                                                                               thresholdBitwidth, featureIndexBitwidth);
  return reinterpret_cast<intptr_t>(inferenceRunner);
}

// Run inference
//    -- inference runner, row, result
extern "C" void RunInference(intptr_t inferenceRunnerInt, void *inputs, void *results) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::InferenceRunnerBase*>(inferenceRunnerInt);
  // TODO The types in this template don't really matter. Maybe we should get rid of them? 
  inferenceRunner->RunInference<double, double>(reinterpret_cast<double*>(inputs), reinterpret_cast<double*>(results));
}

extern "C" void RunInferenceOnMultipleBatches(intptr_t inferenceRunnerInt, void *inputs, void *results, int32_t numRows) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::InferenceRunnerBase*>(inferenceRunnerInt);
  auto batchSize = inferenceRunner->GetBatchSize();
  auto rowSize = inferenceRunner->GetRowSize();

  assert (numRows % batchSize == 0);
  int32_t inputElementSize = inferenceRunner->GetInputElementBitWidth()/8;
  int32_t returnTypeSize = inferenceRunner->GetReturnTypeBitWidth()/8;
  for (int32_t batch=0 ; batch<numRows/batchSize ; ++batch) {
    auto batchPtr = reinterpret_cast<char*>(inputs) + (batch * (rowSize*batchSize) * inputElementSize);
    auto resultsPtr = reinterpret_cast<char*>(results) + (batch * batchSize * returnTypeSize);
    // TODO The types in this template don't really matter. Maybe we should get rid of them? 
    inferenceRunner->RunInference<double, double>(reinterpret_cast<double*>(batchPtr), reinterpret_cast<double*>(resultsPtr));
  }
}

extern "C" int32_t GetBatchSize(intptr_t inferenceRunnerInt) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::InferenceRunnerBase*>(inferenceRunnerInt);
  // TODO The types in this template don't really matter. Maybe we should get rid of them? 
  return inferenceRunner->GetBatchSize();
}

extern "C" int32_t GetRowSize(intptr_t inferenceRunnerInt) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::InferenceRunnerBase*>(inferenceRunnerInt);
  // TODO The types in this template don't really matter. Maybe we should get rid of them? 
  return inferenceRunner->GetRowSize();
}

extern "C" void DeleteInferenceRunner(intptr_t inferenceRunnerInt) {
  auto inferenceRunner = reinterpret_cast<mlir::decisionforest::InferenceRunnerBase*>(inferenceRunnerInt);
  delete inferenceRunner;
}

// ===-------------------------------------------------------------=== //
// CompilerOptions API
// ===-------------------------------------------------------------=== //

extern "C" intptr_t CreateCompilerOptions() {
  return reinterpret_cast<intptr_t>(new TreeBeard::CompilerOptions);
}

extern "C" void DeleteCompilerOptions(intptr_t options) {
  TreeBeard::CompilerOptions *optionsPtr = reinterpret_cast<TreeBeard::CompilerOptions*>(options);
  delete optionsPtr->scheduleManipulator;
  delete optionsPtr;
}


#define COMPILER_OPTION_SETTER(propName, propType) \
extern "C" void Set_##propName(intptr_t options, propType val) { \
  TreeBeard::CompilerOptions *optionsPtr = reinterpret_cast<TreeBeard::CompilerOptions*>(options);  \
  optionsPtr->propName = reinterpret_cast<propType>(val); \
} 

COMPILER_OPTION_SETTER(numberOfFeatures, int32_t);
COMPILER_OPTION_SETTER(batchSize, int32_t)
COMPILER_OPTION_SETTER(tileSize, int32_t)

COMPILER_OPTION_SETTER(thresholdTypeWidth, int32_t)
COMPILER_OPTION_SETTER(returnTypeWidth, int32_t)
COMPILER_OPTION_SETTER(returnTypeFloatType, int32_t)
COMPILER_OPTION_SETTER(featureIndexTypeWidth, int32_t)
COMPILER_OPTION_SETTER(nodeIndexTypeWidth, int32_t)
COMPILER_OPTION_SETTER(inputElementTypeWidth, int32_t)
COMPILER_OPTION_SETTER(tileShapeBitWidth, int32_t)
COMPILER_OPTION_SETTER(childIndexBitWidth, int32_t)
COMPILER_OPTION_SETTER(makeAllLeavesSameDepth, int32_t)
COMPILER_OPTION_SETTER(reorderTreesByDepth, int32_t)
COMPILER_OPTION_SETTER(statsProfileCSVPath,  const char*)
COMPILER_OPTION_SETTER(pipelineSize, int32_t)
COMPILER_OPTION_SETTER(numberOfCores, int32_t)

extern "C" void Set_tilingType(intptr_t options, int32_t val) {
  TreeBeard::CompilerOptions *optionsPtr = reinterpret_cast<TreeBeard::CompilerOptions*>(options);
  TreeBeard::TilingType tilingType;
  if (val == 0)
    tilingType = TreeBeard::TilingType::kUniform;
  else if (val == 1)
    tilingType = TreeBeard::TilingType::kProbabilistic;
  else if (val == 2)
    tilingType = TreeBeard::TilingType::kHybrid;
  else
    assert (false && "Invalid tiling type value");
  optionsPtr->tilingType = tilingType;
}

// ===-------------------------------------------------------------=== //
// Compilation API
// ===-------------------------------------------------------------=== //

extern "C" void GenerateLLVMIRForXGBoostModel(const char* modelJSONPath, const char* llvmIRFilePath,
                                              const char* modelGlobalsJSONPath, intptr_t options) {
  TreeBeard::CompilerOptions *optionsPtr = reinterpret_cast<TreeBeard::CompilerOptions*>(options);
  TreeBeard::TreebeardContext tbContext(modelJSONPath,
                                        modelGlobalsJSONPath,
                                        *optionsPtr, 
                                        mlir::decisionforest::ConstructRepresentation(),
                                        mlir::decisionforest::ConstructModelSerializer(std::string(modelGlobalsJSONPath)),
                                        nullptr  /*TODO_ForestCreator*/);
  TreeBeard::ConvertXGBoostJSONToLLVMIR(tbContext, llvmIRFilePath);
}

extern "C" intptr_t CreateInferenceRunner(const char* modelJSONPath, const char* profileCSVPath,
                                          intptr_t options) {
  TreeBeard::CompilerOptions *optionsPtr = reinterpret_cast<TreeBeard::CompilerOptions*>(options);
  auto modelGlobalsJSONPath = TreeBeard::XGBoostJSONParser<>::ModelGlobalJSONFilePathFromJSONFilePath(modelJSONPath);
  TreeBeard::TreebeardContext tbContext(modelJSONPath,
                                        modelGlobalsJSONPath,
                                        *optionsPtr, 
                                        mlir::decisionforest::ConstructRepresentation(),
                                        mlir::decisionforest::ConstructModelSerializer(modelGlobalsJSONPath),
                                        nullptr  /*TODO_ForestCreator*/);
  auto module = TreeBeard::ConstructLLVMDialectModuleFromXGBoostJSON(tbContext);
  auto inferenceRunner = new mlir::decisionforest::InferenceRunner(tbContext.serializer, module, 
                                                                   optionsPtr->tileSize, optionsPtr->thresholdTypeWidth,
                                                                   optionsPtr->featureIndexTypeWidth);
  return reinterpret_cast<intptr_t>(inferenceRunner);
}

mlir::decisionforest::PredictionTransformation GetPredictionTransformation(const std::string& s)
{
  if (s == "softmax") return mlir::decisionforest::PredictionTransformation::kSoftMax;
  if(s == "id") return mlir::decisionforest::PredictionTransformation::kIdentity;
  if(s == "logistic") return mlir::decisionforest::PredictionTransformation::kSigmoid;
  
  assert(false && "Invalid prediction transformation");
}

extern "C" void CreateLLVMIRForONNXModel(const char *modelPath, const char* llvmIrPath, const char* modelGlobalsJSONPath, intptr_t options)
{
  TreeBeard::CompilerOptions *optionsPtr = reinterpret_cast<TreeBeard::CompilerOptions *>(options);

  TreeBeard::TreebeardContext tbContext(modelPath,
                                        modelGlobalsJSONPath,
                                        *optionsPtr,
                                        mlir::decisionforest::ConstructRepresentation(),
                                        mlir::decisionforest::ConstructModelSerializer(modelGlobalsJSONPath),
                                        nullptr /*TODO_ForestCreator*/ );

  TreeBeard::ConvertONNXModelToLLVMIR(tbContext, llvmIrPath);
}

extern "C" intptr_t CreateInferenceRunnerForONNXModelInputs(int32_t numFeatures, 
    int64_t inputAndThresholdSize, int64_t numNodes, const char *predTransform,
    double baseValue, const int64_t *treeIds, const int64_t *nodeIds,
    const int64_t *featureIds, void *thresholds, const int64_t *leftChildIds,
    const int64_t *rightChildIds, int64_t numberOfClasses,
    const int64_t *targetClassIds, const int64_t *targetClassTreeId,
    const int64_t *targetClassNodeId, const float *targetWeights,
    int64_t numWeights, int64_t batchSize, intptr_t options) {

  auto modelGlobalsJSONPath = std::filesystem::temp_directory_path() / "modelGlobals.json";

  TreeBeard::CompilerOptions *optionsPtr =
      reinterpret_cast<TreeBeard::CompilerOptions *>(options);
  TreeBeard::TreebeardContext tbContext("",
                                        modelGlobalsJSONPath,
                                        *optionsPtr,
                                        mlir::decisionforest::ConstructRepresentation(),
                                        mlir::decisionforest::ConstructModelSerializer(modelGlobalsJSONPath),
                                        nullptr  /*TODO_ForestCreator*/);

  auto& context = tbContext.context;
  mlir::ModuleOp module;

  if (inputAndThresholdSize == 8) {
    auto onnxModelConverter = TreeBeard::ONNXModelConverter<double>(
    tbContext.serializer,
    context,
    numFeatures,
    baseValue,
    GetPredictionTransformation(predTransform),
    mlir::arith::CmpFPredicate::ULE, // TODO - Hardcoded for now
    numNodes,
    treeIds,
    nodeIds,
    featureIds,
    (double *)thresholds,
    leftChildIds,
    rightChildIds,
    numberOfClasses,
    targetClassTreeId,
    targetClassNodeId,
    targetClassIds,
    targetWeights,
    numWeights,
    batchSize);

    module = TreeBeard::ConstructLLVMDialectModuleFromForestCreator(tbContext, onnxModelConverter);

  }
  else if (inputAndThresholdSize == 4) {
    auto onnxModelConverter = TreeBeard::ONNXModelConverter<float>(
    tbContext.serializer,
    context,
    numFeatures,
    baseValue,
    GetPredictionTransformation(predTransform),
    mlir::arith::CmpFPredicate::ULE, // TODO - Hardcoded for now
    numNodes,
    treeIds,
    nodeIds,
    featureIds,
    (float *)thresholds,
    leftChildIds,
    rightChildIds,
    numberOfClasses,
    targetClassTreeId,
    targetClassNodeId,
    targetClassIds,
    targetWeights,
    numWeights,
    batchSize);

    module = TreeBeard::ConstructLLVMDialectModuleFromForestCreator(tbContext, onnxModelConverter);
  }
  
  auto *inferenceRunner = new mlir::decisionforest::InferenceRunner(tbContext.serializer, module, 
                                                                   optionsPtr->tileSize, optionsPtr->thresholdTypeWidth,
                                                                   optionsPtr->featureIndexTypeWidth);
  return reinterpret_cast<intptr_t>(inferenceRunner);
}

extern "C" void SetEnableSparseRepresentation(int32_t val) {
  mlir::decisionforest::UseSparseTreeRepresentation = val;
}

extern "C" int32_t IsSparseRepresentationEnabled() {
  return mlir::decisionforest::UseSparseTreeRepresentation;
}

extern "C" void SetPeeledCodeGenForProbabilityBasedTiling(int32_t val) {
  mlir::decisionforest::PeeledCodeGenForProbabiltyBasedTiling = val;
}

extern "C" int32_t IsPeeledCodeGenForProbabilityBasedTilingEnabled() {
  return mlir::decisionforest::PeeledCodeGenForProbabiltyBasedTiling;
}

// ===-------------------------------------------------------------=== //
// Representation API
// ===-------------------------------------------------------------=== //

namespace 
{
// This set is needed to prevent the shared_ptr we construct from going out of
// scope and destroying the representation object.
std::set<std::shared_ptr<mlir::decisionforest::IRepresentation>> constructedRepresentations;
}

extern "C" void* ConstructRepresentation(const char* repName) {
  auto rep = mlir::decisionforest::RepresentationFactory::Get().GetRepresentation(repName);
  constructedRepresentations.insert(rep);
  return reinterpret_cast<void*>(rep.get());
}

extern "C" void DestroyRepresentation(void *rep) {
  auto iter = constructedRepresentations.begin();
  for (; iter!=constructedRepresentations.end() ; ++iter) {
    if (reinterpret_cast<void*>(iter->get()) == rep)
      break;
  }
  if (iter == constructedRepresentations.end())
    return;
  constructedRepresentations.erase(iter);
  return;
}

// ===-------------------------------------------------------------=== //
// TreebeardContext API
// ===-------------------------------------------------------------=== //
extern "C" intptr_t ConstructTreebeardContext(const char *modelPath, 
                                           const char* modelGlobalsJSONPath,
                                           intptr_t options) {
  TreeBeard::CompilerOptions *optionsPtr = reinterpret_cast<TreeBeard::CompilerOptions *>(options);
  auto* tbContext = new TreeBeard::TreebeardContext(modelPath, 
                                                    modelGlobalsJSONPath,
                                                    *optionsPtr);
  return reinterpret_cast<intptr_t>(tbContext);
}

extern "C" void DestroyTreebeardContext(intptr_t tbContext) {
  TreeBeard::TreebeardContext* tbContextPtr = reinterpret_cast<TreeBeard::TreebeardContext*>(tbContext);
  delete tbContextPtr;
}

extern "C" void SetForestCreatorType(intptr_t tbContext, const char* creatorType) {
  TreeBeard::TreebeardContext* tbContextPtr = reinterpret_cast<TreeBeard::TreebeardContext*>(tbContext);
  tbContextPtr->SetForestCreatorType(creatorType);
}

extern "C" void SetRepresentationAndSerializer(intptr_t tbContext, const char* repType) {
  TreeBeard::TreebeardContext* tbContextPtr = reinterpret_cast<TreeBeard::TreebeardContext*>(tbContext);
  tbContextPtr->SetRepresentationAndSerializer(repType);
}

extern "C" void* GetScheduleFromTBContext(intptr_t tbContext) {
  TreeBeard::TreebeardContext* tbContextPtr = reinterpret_cast<TreeBeard::TreebeardContext*>(tbContext);
  return tbContextPtr->forestConstructor->GetSchedule();
}

// ===-------------------------------------------------------------=== //
// Generic Compilation API
// ===-------------------------------------------------------------=== //

extern "C" void BuildHIRRepresentation(void* tbContext) {
  TreeBeard::TreebeardContext* tbContextPtr = reinterpret_cast<TreeBeard::TreebeardContext*>(tbContext);
  TreeBeard::BuildHIRModule(*tbContextPtr, *tbContextPtr->forestConstructor);
}

inline mlir::ModuleOp LowerToLLVM(void* tbContext) {
  TreeBeard::TreebeardContext* tbContextPtr = reinterpret_cast<TreeBeard::TreebeardContext*>(tbContext);
  auto module = tbContextPtr->forestConstructor->GetModule();
  TreeBeard::DoTilingTransformation(module, *tbContextPtr);
  TreeBeard::LowerHIRModuleToLLVM(module, *tbContextPtr);

  return module;
}

extern "C" bool LowerToLLVMAndDumpIR(void* tbContext, const char* fileName) {
  auto module = LowerToLLVM(tbContext);
  return mlir::decisionforest::dumpLLVMIRToFile(module, fileName) == 0;
}

extern "C" void* ConstructInferenceRunnerFromHIR(void *tbContext) {
  TreeBeard::TreebeardContext* tbContextPtr = reinterpret_cast<TreeBeard::TreebeardContext*>(tbContext);
  auto module = LowerToLLVM(tbContext);
  auto *inferenceRunner = new mlir::decisionforest::InferenceRunner(tbContextPtr->serializer,
                                                                   module, 
                                                                   tbContextPtr->options.tileSize,
                                                                   tbContextPtr->options.thresholdTypeWidth,
                                                                   tbContextPtr->options.featureIndexTypeWidth);
  return reinterpret_cast<void*>(inferenceRunner);  
}

// ===-------------------------------------------------------------=== //
// Predefined Schedule Manipulation API
// ===-------------------------------------------------------------=== //

extern "C" void SetOneTreeAtATimeSchedule(intptr_t options) {
  TreeBeard::CompilerOptions *optionsPtr = reinterpret_cast<TreeBeard::CompilerOptions*>(options);
  optionsPtr->scheduleManipulator = new mlir::decisionforest::ScheduleManipulationFunctionWrapper(mlir::decisionforest::OneTreeAtATimeSchedule);
}

// ===-------------------------------------------------------------=== //
// Schedule API
// ===-------------------------------------------------------------=== //
using Schedule = mlir::decisionforest::Schedule;
using IndexVariable = mlir::decisionforest::IndexVariable;

extern "C" {

intptr_t Schedule_NewIndexVariable(intptr_t schedPtr, const char* name);
intptr_t Schedule_NewIndexVariable2(intptr_t schedPtr, intptr_t indexVarPtr);
void Schedule_Tile(intptr_t schedPtr, intptr_t indexPtr, intptr_t outerPtr, intptr_t innerPtr, int32_t tileSize);
void Schedule_Reorder(intptr_t schedPtr, intptr_t indicesPtr, int32_t numIndices);
void Schedule_Split(intptr_t schedPtr, intptr_t indexPtr, intptr_t firstPtr, intptr_t secondPtr, int32_t splitIteration, intptr_t indexMapPtr);
void Schedule_Pipeline(intptr_t schedPtr, intptr_t indexPtr, int32_t stepSize);
void Schedule_Simdize(intptr_t schedPtr, intptr_t indexPtr);
void Schedule_Parallel(intptr_t schedPtr, intptr_t indexPtr);
void Schedule_Unroll(intptr_t schedPtr, intptr_t indexVarPtr);
void Schedule_PeelWalk(intptr_t schedPtr, intptr_t indexVarPtr, int32_t numberOfIterations);
void Schedule_Cache(intptr_t schedPtr, intptr_t indexVarPtr);
intptr_t Schedule_GetRootIndex(intptr_t schedPtr);
intptr_t Schedule_GetBatchIndex(intptr_t schedPtr);
intptr_t Schedule_GetTreeIndex(intptr_t schedPtr);
int32_t Schedule_PrintToString(intptr_t schedPtr, intptr_t str, int32_t strLen);
int32_t Schedule_GetBatchSize(intptr_t schedPtr);
int32_t Schedule_GetForestSize(intptr_t schedPtr);
bool Schedule_IsDefaultSchedule(intptr_t schedPtr);
void Schedule_Finalize(intptr_t schedPtr);

// Loop Modifiers
intptr_t Schedule_NewIndexVariable(intptr_t schedPtr, const char* name) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  return (intptr_t)(&sched->NewIndexVariable(std::string(name)));
}

intptr_t Schedule_NewIndexVariable2(intptr_t schedPtr, intptr_t indexVarPtr) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  IndexVariable* indexVar = reinterpret_cast<IndexVariable*>(indexVarPtr);
  return (intptr_t)(&sched->NewIndexVariable(*indexVar));
}

void Schedule_Tile(intptr_t schedPtr, intptr_t indexPtr, intptr_t outerPtr, intptr_t innerPtr, int32_t tileSize) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  IndexVariable* index = reinterpret_cast<IndexVariable*>(indexPtr);
  IndexVariable* outer = reinterpret_cast<IndexVariable*>(outerPtr);
  IndexVariable* inner = reinterpret_cast<IndexVariable*>(innerPtr);
  sched->Tile(*index, *outer, *inner, tileSize);
}

void Schedule_Reorder(intptr_t schedPtr, intptr_t indicesPtr, int32_t numIndices) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  std::vector<IndexVariable*> indices(numIndices);
  for (int i = 0; i < numIndices; i++) {
    indices[i] = ((IndexVariable**)indicesPtr)[i];
  }
  sched->Reorder(indices);
}

void Schedule_Split(intptr_t schedPtr, intptr_t indexPtr, intptr_t firstPtr, intptr_t secondPtr, int32_t splitIteration, intptr_t indexMapPtr) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  IndexVariable* index = reinterpret_cast<IndexVariable*>(indexPtr);
  IndexVariable* first = reinterpret_cast<IndexVariable*>(firstPtr);
  IndexVariable* second = reinterpret_cast<IndexVariable*>(secondPtr);
  Schedule::IndexVariableMapType indexMap;
  sched->Split(*index, *first, *second, splitIteration, indexMap);
  int32_t i=0;
  IndexVariable** indexMapArr = reinterpret_cast<IndexVariable**>(indexMapPtr);
  for (auto& indexPair: indexMap) {
    indexMapArr[i] = indexPair.first;
    indexMapArr[i+1] = indexPair.second.first;
    indexMapArr[i+2] = indexPair.second.second;
    i+=3;
  }
}

// Optimizations
void Schedule_Pipeline(intptr_t schedPtr, intptr_t indexPtr, int32_t stepSize) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  IndexVariable* index = reinterpret_cast<IndexVariable*>(indexPtr);
  sched->Pipeline(*index, stepSize);
}

void Schedule_Simdize(intptr_t schedPtr, intptr_t indexPtr) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  IndexVariable* index = reinterpret_cast<IndexVariable*>(indexPtr);
  sched->Simdize(*index);
}

void Schedule_Parallel(intptr_t schedPtr, intptr_t indexPtr) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  IndexVariable* index = (IndexVariable*)indexPtr;
  sched->Parallel(*index);
}

// Wrapper function for Schedule::Unroll
void Schedule_Unroll(intptr_t schedPtr, intptr_t indexVarPtr) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  IndexVariable* indexVar = reinterpret_cast<IndexVariable*>(indexVarPtr);
  sched->Unroll(*indexVar);
}

// Wrapper function for Schedule::PeelWalk
void Schedule_PeelWalk(intptr_t schedPtr, intptr_t indexVarPtr, int32_t numberOfIterations) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  IndexVariable* indexVar = reinterpret_cast<IndexVariable*>(indexVarPtr);
  sched->PeelWalk(*indexVar, numberOfIterations);
}

// Wrapper function for Schedule::Cache
void Schedule_Cache(intptr_t schedPtr, intptr_t indexVarPtr) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  IndexVariable* indexVar = reinterpret_cast<IndexVariable*>(indexVarPtr);
  sched->Cache(*indexVar);
}

// Wrapper function for Schedule::GetRootIndex
intptr_t Schedule_GetRootIndex(intptr_t schedPtr) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  const IndexVariable* rootIndex = sched->GetRootIndex();
  return reinterpret_cast<intptr_t>(rootIndex);
}

// Wrapper function for Schedule::GetBatchIndex
intptr_t Schedule_GetBatchIndex(intptr_t schedPtr) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  IndexVariable& batchIndex = sched->GetBatchIndex();
  return reinterpret_cast<intptr_t>(&batchIndex);
}

// Wrapper function for Schedule::GetTreeIndex
intptr_t Schedule_GetTreeIndex(intptr_t schedPtr) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  IndexVariable& treeIndex = sched->GetTreeIndex();
  return reinterpret_cast<intptr_t>(&treeIndex);
}

// Wrapper function for Schedule::PrintToString
int32_t Schedule_PrintToString(intptr_t schedPtr, intptr_t str, int32_t strLen) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  std::string scheduleString = sched->PrintToString();
  auto *strPtr = reinterpret_cast<char*>(str);
  if (strPtr == nullptr)
    return (int32_t)scheduleString.size();
  strncpy(strPtr, scheduleString.c_str(), strLen);
  return std::min((int32_t)scheduleString.size(), strLen);
}

// Wrapper function for Schedule::GetBatchSize
int32_t Schedule_GetBatchSize(intptr_t schedPtr) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  return sched->GetBatchSize();
}

// Wrapper function for Schedule::GetForestSize
int32_t Schedule_GetForestSize(intptr_t schedPtr) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  return sched->GetForestSize();
}

// Wrapper function for Schedule::IsDefaultSchedule
bool Schedule_IsDefaultSchedule(intptr_t schedPtr) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  return sched->IsDefaultSchedule();
}

// Wrapper function for Schedule::Finalize
void Schedule_Finalize(intptr_t schedPtr) {
  Schedule* sched = reinterpret_cast<Schedule*>(schedPtr);
  sched->Finalize();
}
} // extern "C"
