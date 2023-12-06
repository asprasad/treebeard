#ifndef _TREEBEARD_CONTEXT_H_
#define _TREEBEARD_CONTEXT_H_

#include <string>
#include "DecisionForest.h"
#include "TreeTilingUtils.h"
#include "Dialect.h"
#include "ExecutionHelpers.h"

#define CONCAT(a, b, c) CONCAT_INNER(a, b, c)
#define CONCAT_INNER(a, b, c) a##b##c

#define UNIQUE_NAME(base) CONCAT(base, __COUNTER__, __LINE__)

namespace mlir
{
namespace decisionforest
{

class IRepresentation;

class IModelSerializer {
protected:
  std::string m_filepath;
  InferenceRunnerBase *m_inferenceRunner=nullptr;

  template<typename FuncType>
  FuncType GetFunctionAddress(const std::string& funcName) {
    return reinterpret_cast<FuncType>(m_inferenceRunner->GetFunctionAddress(funcName));
  }

  virtual void InitializeBuffersImpl()=0;
public:
  IModelSerializer(const std::string& filepath)
    :m_filepath(filepath)
  { }
  virtual ~IModelSerializer() { }
  virtual void Persist(mlir::decisionforest::DecisionForest& forest, mlir::decisionforest::TreeEnsembleType forestType)=0;
  virtual void ReadData()=0;

  virtual void CallPredictionMethod(void* predictFuncPtr,
                                    Memref<double, 2> inputs,
                                    Memref<double, 1> results) { }
  virtual bool HasCustomPredictionMethod() { return false; }    
  
  virtual void CleanupBuffers() { }

  void InitializeBuffers(InferenceRunnerBase* inferenceRunner) {
    m_inferenceRunner = inferenceRunner;
    this->InitializeBuffersImpl();
  }
  
  const std::string& GetFilePath() const { return m_filepath; }
};

} // decisionforest
} // mlir

namespace TreeBeard
{

class ForestCreator;

enum class TilingType { kUniform, kProbabilistic, kHybrid };

struct CompilerOptions {
  // model parameters
  int32_t numberOfFeatures = -1; // TODO: Currently used only by ONNX.

  // optimization parameters
  int32_t batchSize;
  int32_t tileSize;

  // Type size parameters
  int32_t thresholdTypeWidth=32;
  int32_t returnTypeWidth=32;
  bool returnTypeFloatType=true;
  int32_t featureIndexTypeWidth=16;
  int32_t nodeIndexTypeWidth=16;
  int32_t inputElementTypeWidth=32;
  int32_t tileShapeBitWidth=16;
  int32_t childIndexBitWidth=16;
  TilingType tilingType=TilingType::kUniform;
  bool makeAllLeavesSameDepth=false;
  bool reorderTreesByDepth=false;
  int32_t pipelineSize = -1;

  mlir::decisionforest::ScheduleManipulator *scheduleManipulator=nullptr;
  std::string statsProfileCSVPath = "";
  int32_t numberOfCores = -1;

  CompilerOptions() { }
  CompilerOptions(int32_t thresholdWidth, int32_t returnWidth, bool isReturnTypeFloat, int32_t featureIndexWidth, 
                  int32_t nodeIndexWidth, int32_t inputElementWidth, int32_t batchSz, int32_t tileSz,
                  int32_t tileShapeWidth, int32_t childIndexWidth, TilingType tileType, bool makeLeavesSameDepth, bool reorderTrees,
                  mlir::decisionforest::ScheduleManipulator* scheduleManip)
  : batchSize(batchSz), tileSize(tileSz), thresholdTypeWidth(thresholdWidth), returnTypeWidth(returnWidth), returnTypeFloatType(isReturnTypeFloat),
    featureIndexTypeWidth(featureIndexWidth), nodeIndexTypeWidth(nodeIndexWidth), inputElementTypeWidth(inputElementWidth),
    tileShapeBitWidth(tileShapeWidth), childIndexBitWidth(childIndexWidth), tilingType(tileType), makeAllLeavesSameDepth(makeLeavesSameDepth),
    reorderTreesByDepth(reorderTrees), scheduleManipulator(scheduleManip)
  { }
  CompilerOptions(const std::string& configJSONFilePath);

  void SetPipelineSize(int32_t pipelineSize) { this->pipelineSize = pipelineSize; }
};

void InitializeMLIRContext(mlir::MLIRContext& context);

struct TreebeardContext {
  mlir::MLIRContext context;
  std::string modelPath;
  std::string modelGlobalsJSONPath;
  CompilerOptions options;
  std::shared_ptr<mlir::decisionforest::IRepresentation>  representation = nullptr;
  std::shared_ptr<mlir::decisionforest::IModelSerializer> serializer = nullptr;
  std::shared_ptr<ForestCreator> forestConstructor = nullptr;

  TreebeardContext(const std::string& modelFilePath, 
                   const std::string& globalsJSONPath,
                   CompilerOptions& compilerOptions,
                   std::shared_ptr<mlir::decisionforest::IRepresentation>  rep = nullptr,
                   std::shared_ptr<mlir::decisionforest::IModelSerializer> ser = nullptr,
                   std::shared_ptr<ForestCreator> constructor = nullptr) 
    : modelPath(modelFilePath), 
      modelGlobalsJSONPath(globalsJSONPath), 
      options(compilerOptions),
      representation(rep),
      serializer(ser),
      forestConstructor(constructor)
  {
    InitializeMLIRContext(context);
  }

  TreebeardContext() {
    InitializeMLIRContext(context);
  }

  void SetForestCreatorType(const std::string& creatorName);
  void SetRepresentationAndSerializer(const std::string& repName);
};

} // namespace TreeBeard

#endif // _TREEBEARD_CONTEXT_H_