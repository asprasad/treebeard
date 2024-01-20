#ifndef _TREEBEARD_CONTEXT_H_
#define _TREEBEARD_CONTEXT_H_

#include "DecisionForest.h"
#include "Dialect.h"
#include "ExecutionHelpers.h"
#include "TreeTilingUtils.h"
#include <string>

#define CONCAT(a, b, c) CONCAT_INNER(a, b, c)
#define CONCAT_INNER(a, b, c) a##b##c

#define UNIQUE_NAME(base) CONCAT(base, __COUNTER__, __LINE__)

namespace mlir {
namespace decisionforest {

class IRepresentation;

class IModelSerializer {
protected:
  std::string m_filepath;
  InferenceRunnerBase *m_inferenceRunner = nullptr;

  template <typename FuncType>
  FuncType GetFunctionAddress(const std::string &funcName) {
    return reinterpret_cast<FuncType>(
        m_inferenceRunner->GetFunctionAddress(funcName));
  }

  virtual void InitializeBuffersImpl() = 0;

public:
  IModelSerializer(const std::string &filepath) : m_filepath(filepath) {}
  virtual ~IModelSerializer() {}
  virtual void Persist(mlir::decisionforest::DecisionForest &forest,
                       mlir::decisionforest::TreeEnsembleType forestType) = 0;
  virtual void ReadData() = 0;

  virtual void CallPredictionMethod(void *predictFuncPtr,
                                    Memref<double, 2> inputs,
                                    Memref<double, 1> results) {}
  virtual bool HasCustomPredictionMethod() { return false; }

  virtual void CleanupBuffers() {}

  void InitializeBuffers(InferenceRunnerBase *inferenceRunner) {
    m_inferenceRunner = inferenceRunner;
    this->InitializeBuffersImpl();
  }

  const std::string &GetFilePath() const { return m_filepath; }
};

} // namespace decisionforest
} // namespace mlir

namespace TreeBeard {

class ForestCreator;

enum class TilingType { kUniform, kProbabilistic, kHybrid };

struct CompilerOptions {
  // model parameters
  int32_t numberOfFeatures = -1; // TODO: Currently used only by ONNX.

  // optimization parameters
  int32_t batchSize;
  int32_t tileSize;

  // Type size parameters
  int32_t thresholdTypeWidth = 32;
  int32_t returnTypeWidth = 32;
  bool returnTypeFloatType = true;
  int32_t featureIndexTypeWidth = 16;
  int32_t nodeIndexTypeWidth = 16;
  int32_t inputElementTypeWidth = 32;
  int32_t tileShapeBitWidth = 16;
  int32_t childIndexBitWidth = 16;
  TilingType tilingType = TilingType::kUniform;
  bool makeAllLeavesSameDepth = false;
  bool reorderTreesByDepth = false;
  int32_t pipelineSize = -1;

  std::string statsProfileCSVPath = "";
  int32_t numberOfCores = -1;
  int32_t numParallelTreeBatches = -1;

  mlir::decisionforest::ScheduleManipulator *scheduleManipulator = nullptr;

  CompilerOptions() {}
  CompilerOptions(int32_t thresholdWidth, int32_t returnWidth,
                  bool isReturnTypeFloat, int32_t featureIndexWidth,
                  int32_t nodeIndexWidth, int32_t inputElementWidth,
                  int32_t batchSz, int32_t tileSz, int32_t tileShapeWidth,
                  int32_t childIndexWidth, TilingType tileType,
                  bool makeLeavesSameDepth, bool reorderTrees,
                  int32_t pipelineSz, int32_t numCores,
                  int32_t numParTreeBatches,
                  mlir::decisionforest::ScheduleManipulator *scheduleManip)
      : batchSize(batchSz), tileSize(tileSz),
        thresholdTypeWidth(thresholdWidth), returnTypeWidth(returnWidth),
        returnTypeFloatType(isReturnTypeFloat),
        featureIndexTypeWidth(featureIndexWidth),
        nodeIndexTypeWidth(nodeIndexWidth),
        inputElementTypeWidth(inputElementWidth),
        tileShapeBitWidth(tileShapeWidth), childIndexBitWidth(childIndexWidth),
        tilingType(tileType), makeAllLeavesSameDepth(makeLeavesSameDepth),
        reorderTreesByDepth(reorderTrees), pipelineSize(pipelineSz),
        numberOfCores(numCores), numParallelTreeBatches(numParTreeBatches),
        scheduleManipulator(scheduleManip) {}

  CompilerOptions(int32_t thresholdWidth, int32_t returnWidth,
                  bool isReturnTypeFloat, int32_t featureIndexWidth,
                  int32_t nodeIndexWidth, int32_t inputElementWidth,
                  int32_t batchSz, int32_t tileSz, int32_t tileShapeWidth,
                  int32_t childIndexWidth, TilingType tileType,
                  bool makeLeavesSameDepth, bool reorderTrees,
                  mlir::decisionforest::ScheduleManipulator *scheduleManip)
      : batchSize(batchSz), tileSize(tileSz),
        thresholdTypeWidth(thresholdWidth), returnTypeWidth(returnWidth),
        returnTypeFloatType(isReturnTypeFloat),
        featureIndexTypeWidth(featureIndexWidth),
        nodeIndexTypeWidth(nodeIndexWidth),
        inputElementTypeWidth(inputElementWidth),
        tileShapeBitWidth(tileShapeWidth), childIndexBitWidth(childIndexWidth),
        tilingType(tileType), makeAllLeavesSameDepth(makeLeavesSameDepth),
        reorderTreesByDepth(reorderTrees), scheduleManipulator(scheduleManip) {}

  CompilerOptions(const std::string &configJSONFilePath);

  void SetPipelineSize(int32_t pipelineSize) {
    this->pipelineSize = pipelineSize;
  }
};

struct GPUAutoScheduleOptions {
  // Maps to TB.x
  int32_t numRowsPerTB;
  int32_t numRowsPerThread;

  // Hpw many rows do we process together?
  // Pick one tree and process this many rows before moving to the next row
  int32_t rowTileSize;

  // How many threads do we divide the trees across?
  // Maps to TB.y
  int32_t numTreeThreads;

  // How many trees do we process at a time? Only useful if single threaded
  // TODO Do we really need this? Can't we always do one tree at a time
  int32_t numTreesAtATime;

  bool cacheRows = false;
  bool cacheTrees = false;

  bool unrollTreeWalks = false;
  int32_t treeWalkInterleaveFactor = -1;

  bool sharedMemoryReduce = false;
};

struct GPUCompileInfo {
  int32_t sharedMemoryInBytes = 0;
};

void InitializeMLIRContext(mlir::MLIRContext &context);

struct TreebeardContext {
private:
  bool m_ownContext = false;
  bool m_gpuScheduleOptionsSet = false;

  void initializeGPUAutoScheduleOptions(
      GPUAutoScheduleOptions *gpuScheduleOptionsPtr) {
    if (gpuScheduleOptionsPtr == nullptr)
      return;
    m_gpuScheduleOptionsSet = true;
    gpuScheduleOptions = *gpuScheduleOptionsPtr;
  }

public:
  mlir::MLIRContext &context;
  std::string modelPath;
  std::string modelGlobalsJSONPath;
  CompilerOptions options;
  std::shared_ptr<mlir::decisionforest::IRepresentation> representation =
      nullptr;
  std::shared_ptr<mlir::decisionforest::IModelSerializer> serializer = nullptr;
  std::shared_ptr<ForestCreator> forestConstructor = nullptr;
  GPUAutoScheduleOptions gpuScheduleOptions;
  GPUCompileInfo gpuCompileInfo;

  TreebeardContext(
      const std::string &modelFilePath, const std::string &globalsJSONPath,
      CompilerOptions &compilerOptions,
      std::shared_ptr<mlir::decisionforest::IRepresentation> rep = nullptr,
      std::shared_ptr<mlir::decisionforest::IModelSerializer> ser = nullptr,
      std::shared_ptr<ForestCreator> constructor = nullptr,
      GPUAutoScheduleOptions *gpuScheduleOptionsPtr = nullptr)
      : m_ownContext(true), context(*new mlir::MLIRContext()),
        modelPath(modelFilePath), modelGlobalsJSONPath(globalsJSONPath),
        options(compilerOptions), representation(rep), serializer(ser),
        forestConstructor(constructor) {
    InitializeMLIRContext(context);
    initializeGPUAutoScheduleOptions(gpuScheduleOptionsPtr);
  }

  TreebeardContext(
      mlir::MLIRContext &ctxt, const std::string &modelFilePath,
      const std::string &globalsJSONPath, CompilerOptions &compilerOptions,
      std::shared_ptr<mlir::decisionforest::IRepresentation> rep = nullptr,
      std::shared_ptr<mlir::decisionforest::IModelSerializer> ser = nullptr,
      std::shared_ptr<ForestCreator> constructor = nullptr,
      GPUAutoScheduleOptions *gpuScheduleOptionsPtr = nullptr)
      : m_ownContext(false), context(ctxt), modelPath(modelFilePath),
        modelGlobalsJSONPath(globalsJSONPath), options(compilerOptions),
        representation(rep), serializer(ser), forestConstructor(constructor) {
    InitializeMLIRContext(context);
    initializeGPUAutoScheduleOptions(gpuScheduleOptionsPtr);
  }

  TreebeardContext() : m_ownContext(true), context(*new mlir::MLIRContext()) {
    InitializeMLIRContext(context);
  }

  ~TreebeardContext() {
    if (m_ownContext)
      delete &context;
  }

  void SetForestCreatorType(const std::string &creatorName);
  void SetRepresentationAndSerializer(const std::string &repName);

  bool shouldUseGPUAutoSchedule() { return m_gpuScheduleOptionsSet; }
};

} // namespace TreeBeard

#endif // _TREEBEARD_CONTEXT_H_