#ifdef TREEBEARD_GPU_SUPPORT

#ifndef _REORG_FOREST_REPRESENTATION_H_
#define _REORG_FOREST_REPRESENTATION_H_

#include "Representations.h"
#include "TreebeardContext.h"
#include "json.hpp"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"


using json = nlohmann::json;

namespace mlir {
namespace decisionforest {

// ===---------------------------------------------------=== //
// Reorg forest serializer
// ===---------------------------------------------------=== //

class ReorgForestSerializer : public IModelSerializer {
protected:
  int32_t m_thresholdBitWidth;
  int32_t m_featureIndexBitWidth;
  int32_t m_numberOfTrees;
  int32_t m_numberOfClasses;

  std::vector<double> m_thresholds;
  std::vector<int32_t> m_featureIndices;
  std::vector<int8_t> m_classIds;

  json m_json;

  Memref<double, 1> m_thresholdMemref;
  Memref<int32_t, 1> m_featureIndexMemref;
  Memref<int8_t, 1> m_classIDMemref;

  void
  WriteSingleTreeIntoReorgBuffer(mlir::decisionforest::DecisionForest &forest,
                                 int32_t treeIndex);
  void WriteJSONFile();

  template <typename VectorElemType, typename MemrefElemType>
  void InitializeSingleBuffer(const std::string &initFuncName,
                              const std::vector<VectorElemType> &vals,
                              Memref<VectorElemType, 1> &gpuMemref);
  void InitializeThresholds();
  void InitializeFeatureIndices();
  void InitializeBuffersImpl() override;

public:
  ReorgForestSerializer(const std::string &jsonFilename);
  ~ReorgForestSerializer() {}
  void Persist(mlir::decisionforest::DecisionForest &forest,
               mlir::decisionforest::TreeEnsembleType forestType) override;
  void ReadData() override;
  void CleanupBuffers() override;

  void CallPredictionMethod(void *predictFuncPtr, Memref<double, 2> inputs,
                            Memref<double, 1> results) override;
  bool HasCustomPredictionMethod() override;

  size_t GetNumberOfNodes() {
    assert(m_thresholds.size() == m_featureIndices.size());
    return m_thresholds.size();
  }

  Memref<double, 1> GetThresholdMemref() { return m_thresholdMemref; }
  Memref<int32_t, 1> GetFeatureIndexMemref() { return m_featureIndexMemref; }
  size_t GetNumberOfElements() { return m_thresholds.size(); }

  template <typename T> std::vector<T> GetThresholds() {
    std::vector<T> thresholds(m_thresholds.begin(), m_thresholds.end());
    return thresholds;
  }

  template <typename T> std::vector<T> GetFeatureIndices() {
    std::vector<T> indices(m_featureIndices.begin(), m_featureIndices.end());
    return indices;
  }
};

// ===---------------------------------------------------=== //
// Reorg forest representation
// ===---------------------------------------------------=== //

class ReorgForestRepresentation : public IRepresentation {
protected:
  int32_t m_tileSize = -1;
  mlir::Type m_thresholdType;
  mlir::Type m_featureIndexType;

  int32_t m_thresholdMemrefArgIndex = -1;
  int32_t m_featureIndexMemrefArgIndex = -1;
  int32_t m_classInfoMemrefArgIndex = -1;

  mlir::Value m_thresholdMemref;
  mlir::Value m_featureIndexMemref;
  mlir::Value m_classInfoMemref;

  int32_t m_numTrees = -1;
  struct CacheGlobalValues {
    Value thresholdCache;
    Value featureIndexCache;
  };
  CacheGlobalValues AddTreeCacheGlobals(ConversionPatternRewriter &rewriter,
                                        Operation *op, int32_t bufferLen,
                                        int64_t numTreesToCache);

  struct TreeCacheInfo {
    Value thresholdMemref;
    Value featureIndexMemref;
    int64_t numCachedTrees;
  };
  std::map<Operation *, TreeCacheInfo> m_treeCacheMap;

  struct GetTreeInfo {
    Value correctedIndex;
    Value thresholdBuffer;
    Value featureIndexBuffer;
  };
  std::map<Operation *, GetTreeInfo> m_getTreeFromEnsembleMap;
  std::map<int64_t, std::string> m_cacheBufferNamesMap;

public:
  ~ReorgForestRepresentation() {}
  void InitRepresentation() override { m_getTreeFromEnsembleMap.clear(); }
  mlir::LogicalResult GenerateModelGlobals(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter,
      std::shared_ptr<decisionforest::IModelSerializer> m_serializer) override;
  mlir::Value GetThresholdsMemref(mlir::Value treeValue) override;
  mlir::Value GetFeatureIndexMemref(mlir::Value treeValue) override;
  mlir::Value GetTileShapeMemref(mlir::Value treeValue) override {
    return Value();
  }

  std::vector<mlir::Value>
  GenerateExtraLoads(mlir::Location location,
                     ConversionPatternRewriter &rewriter, mlir::Value tree,
                     mlir::Value nodeIndex) override {
    return std::vector<mlir::Value>();
  }
  mlir::Value GenerateMoveToChild(
      mlir::Location location, ConversionPatternRewriter &rewriter,
      mlir::Value nodeIndex, mlir::Value childNumber, int32_t tileSize,
      std::vector<mlir::Value> &extraLoads) override;
  void GenerateTreeMemref(mlir::ConversionPatternRewriter &rewriter,
                          mlir::Operation *op, Value ensemble,
                          Value treeIndex) override;
  mlir::Value GenerateGetTreeClassId(mlir::ConversionPatternRewriter &rewriter,
                                     mlir::Operation *op, Value ensemble,
                                     Value treeIndex) override;
  mlir::Value GenerateGetLeafValueOp(ConversionPatternRewriter &rewriter,
                                     mlir::Operation *op, mlir::Value treeValue,
                                     mlir::Value nodeIndex) override;
  mlir::Value GenerateIsLeafOp(ConversionPatternRewriter &rewriter,
                               mlir::Operation *op, mlir::Value treeValue,
                               mlir::Value nodeIndex) override;
  mlir::Value GenerateIsLeafTileOp(ConversionPatternRewriter &rewriter,
                                   mlir::Operation *op, mlir::Value treeValue,
                                   mlir::Value nodeIndex) override;

  int32_t GetTileSize() override {
    assert(m_tileSize != -1 && "Tile size is not initialized");
    assert(m_tileSize == 1 && "Reorg forest doesn't support vectorization");
    return m_tileSize;
  }

  mlir::Type GetIndexElementType() override { return m_featureIndexType; }
  mlir::Type GetThresholdElementType() override { return m_thresholdType; }

  int32_t getTypeBitWidth(mlir::Type type) override;

  // This should never be needed because this representation only
  // supports scalar code generation. Returning a default Type
  // object so that anyone trying to use it crashes.
  mlir::Type GetTileShapeType() override {
    assert(
        false &&
        "Unimplemented function ReorgForestRepresentation::GetTileShapeType()");
    return mlir::Type();
  }
  mlir::Value GetTreeIndex(Value tree) override;

  void AddTypeConversions(mlir::MLIRContext &context,
                          LLVMTypeConverter &typeConverter) override;
  void AddTypeConversions(mlir::MLIRContext &context,
                          SPIRVTypeConverter &typeConverter);
  void AddLLVMConversionPatterns(LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns) override;

  void LowerCacheTreeOp(
      ConversionPatternRewriter &rewriter, mlir::Operation *op,
      ArrayRef<Value> operands,
      std::shared_ptr<decisionforest::IModelSerializer> m_serializer) override;

  void LowerCacheRowsOp(ConversionPatternRewriter &rewriter,
                        mlir::Operation *op, ArrayRef<Value> operands) override;
};

} // namespace decisionforest
} // namespace mlir

#endif // _REORG_FOREST_REPRESENTATION_H_

#endif // TREEBEARD_GPU_SUPPORT