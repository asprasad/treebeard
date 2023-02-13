#ifndef _REORG_FOREST_REPRESENTATION_H_
#define _REORG_FOREST_REPRESENTATION_H_

#include "mlir/Transforms/DialectConversion.h"
#include "TreebeardContext.h"

namespace mlir
{
namespace decisionforest
{

class ReorgForestSerializer : public IModelSerializer {
  int32_t m_inputElementBitwidth;
  int32_t m_returnTypeBitWidth;
  int32_t m_thresholdBitWidth;
  int32_t m_featureIndexBitWidth;
  int32_t m_rowSize;
  int32_t m_batchSize;
  int32_t m_numberOfTrees;
  int32_t m_numberOfClasses;
  
  std::vector<double> m_thresholds;
  std::vector<int32_t> m_featureIndices;
  std::vector<int32_t> m_classIds;

  void WriteSingleTreeIntoReorgBuffer(mlir::decisionforest::DecisionForest<>& forest, int32_t treeIndex);
public:
  ReorgForestSerializer(const std::string& jsonFilename);
  ~ReorgForestSerializer() {}
  void Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) override;
};

class ReorgForestRepresentation : public IRepresentation {
protected:
  int32_t m_tileSize=-1;
  mlir::Type m_thresholdType;
  mlir::Type m_featureIndexType;
  mlir::Type m_tileShapeType;

  void GenModelMemrefInitFunctionBody(MemRefType memrefType, Value memrefValue,
                                      mlir::OpBuilder &rewriter, Location location, Value tileIndex,
                                      Value thresholdMemref, Value indexMemref, Value tileShapeIdMemref);

  void AddModelMemrefInitFunction(mlir::ModuleOp module, std::string globalName, MemRefType memrefType, 
                                  ConversionPatternRewriter &rewriter, Location location);

public:
  virtual ~ReorgForestRepresentation() { }
  void InitRepresentation() override;
  mlir::LogicalResult GenerateModelGlobals(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter,
                                           std::shared_ptr<decisionforest::IModelSerializer> m_serializer) override;
  virtual mlir::Value GetThresholdsMemref(mlir::Value treeValue) override;
  virtual mlir::Value GetFeatureIndexMemref(mlir::Value treeValue) override;
  virtual mlir::Value GetTileShapeMemref(mlir::Value treeValue) override;

  std::vector<mlir::Value> GenerateExtraLoads(mlir::Location location, 
                                              ConversionPatternRewriter &rewriter,
                                              mlir::Value tree, 
                                              mlir::Value nodeIndex) override { return std::vector<mlir::Value>(); }
  mlir::Value GenerateMoveToChild(mlir::Location location, ConversionPatternRewriter &rewriter, mlir::Value nodeIndex, 
                                  mlir::Value childNumber, int32_t tileSize, std::vector<mlir::Value>& extraLoads) override;
  void GenerateTreeMemref(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) override;
  mlir::Value GenerateGetTreeClassId(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) override;
  mlir::Value GenerateGetLeafValueOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, 
                                     mlir::Value nodeIndex) override;
  mlir::Value GenerateIsLeafOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex) override;
  mlir::Value GenerateIsLeafTileOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex) override;

  int32_t GetTileSize() override {
    assert (m_tileSize != -1 && "Tile size is not initialized");
    assert (m_tileSize == 1 && "Reorg forest doesn't support vectorization");
    return m_tileSize;
  }
  mlir::Type GetIndexElementType() override {
    return m_featureIndexType;
  }
  mlir::Type GetThresholdElementType() override {
    return m_thresholdType;
  }
  mlir::Type GetTileShapeType() override {
    return m_tileShapeType;
  }
  void AddTypeConversions(mlir::MLIRContext& context, LLVMTypeConverter& typeConverter) override;
  void AddLLVMConversionPatterns(LLVMTypeConverter &converter, RewritePatternSet &patterns) override;
};

}
}

#endif // _REORG_FOREST_REPRESENTATION_H_
