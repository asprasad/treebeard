#ifndef _TREEBEARD_CONTEXT_H_
#define _TREEBEARD_CONTEXT_H_

#include <string>
#include "DecisionForest.h"
#include "TreeTilingUtils.h"
#include "Dialect.h"

namespace mlir
{
namespace decisionforest
{

class IRepresentation;

class IModelSerializer {
protected:
  std::string m_filepath;
  int32_t m_batchSize=-1;
  int32_t m_rowSize=-1;
  int32_t m_inputTypeBitWidth=-1;
  int32_t m_returnTypeBitwidth=-1;
public:
  IModelSerializer(const std::string& filepath)
    :m_filepath(filepath)
  { }
  virtual ~IModelSerializer() { }
  virtual void Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType)=0;
  virtual void ReadData()=0;

  const std::string& GetFilePath() const { return m_filepath; }
  
  virtual void SetBatchSize(int32_t value)=0;
  virtual void SetRowSize(int32_t value)=0;
  virtual void SetInputTypeBitWidth(int32_t value)=0;
  virtual void SetReturnTypeBitWidth(int32_t value)=0;

  int32_t GetBatchSize() { return m_batchSize; }
  int32_t GetRowSize() { return m_rowSize; }
  int32_t GetInputTypeBitWidth() { return m_inputTypeBitWidth; }
  int32_t GetReturnTypeBitWidth() { return m_returnTypeBitwidth; }
};

} // decisionforest
} // mlir

namespace TreeBeard
{

enum class TilingType { kUniform, kProbabilistic, kHybrid };

struct CompilerOptions {
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

struct TreebeardContext {
  std::string modelJSONPath;
  std::string modelGlobalsJSONPath;
  CompilerOptions options;
  std::shared_ptr<mlir::decisionforest::IRepresentation>  representation = nullptr;
  std::shared_ptr<mlir::decisionforest::IModelSerializer> serializer = nullptr;
};

} // namespace TreeBeard

#endif // _TREEBEARD_CONTEXT_H_