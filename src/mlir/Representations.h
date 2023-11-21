#ifndef _REPRESENTATIONS_H_
#define _REPRESENTATIONS_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "TreebeardContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace decisionforest {

class IRepresentation {
public:
  virtual ~IRepresentation() {}
  virtual void InitRepresentation() = 0;
  virtual mlir::LogicalResult GenerateModelGlobals(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter,
      std::shared_ptr<decisionforest::IModelSerializer> m_serializer) = 0;
  virtual mlir::Value GetThresholdsMemref(mlir::Value treeValue) = 0;
  virtual mlir::Value GetFeatureIndexMemref(mlir::Value treeValue) = 0;
  virtual mlir::Value GetTileShapeMemref(mlir::Value treeValue) = 0;
  virtual std::vector<mlir::Value>
  GenerateExtraLoads(mlir::Location location,
                     ConversionPatternRewriter &rewriter, mlir::Value tree,
                     mlir::Value nodeIndex) = 0;
  virtual mlir::Value GenerateMoveToChild(
      mlir::Location location, ConversionPatternRewriter &rewriter,
      mlir::Value nodeIndex, mlir::Value childNumber, int32_t tileSize,
      std::vector<mlir::Value> &extraLoads) = 0;
  virtual void GenerateTreeMemref(mlir::ConversionPatternRewriter &rewriter,
                                  mlir::Operation *op, Value ensemble,
                                  Value treeIndex) = 0;
  virtual mlir::Value
  GenerateGetTreeClassId(mlir::ConversionPatternRewriter &rewriter,
                         mlir::Operation *op, Value ensemble,
                         Value treeIndex) = 0;
  virtual mlir::Value
  GenerateGetLeafValueOp(ConversionPatternRewriter &rewriter,
                         mlir::Operation *op, mlir::Value treeValue,
                         mlir::Value nodeIndex) = 0;
  virtual mlir::Value GenerateIsLeafOp(ConversionPatternRewriter &rewriter,
                                       mlir::Operation *op,
                                       mlir::Value treeValue,
                                       mlir::Value nodeIndex) = 0;
  virtual mlir::Value GenerateIsLeafTileOp(ConversionPatternRewriter &rewriter,
                                           mlir::Operation *op,
                                           mlir::Value treeValue,
                                           mlir::Value nodeIndex) = 0;

  virtual int32_t GetTileSize() = 0;
  virtual mlir::Type GetIndexElementType() = 0;
  virtual mlir::Type GetThresholdElementType() = 0;
  virtual mlir::Type GetTileShapeType() = 0;
  virtual mlir::Type GetThresholdFieldType() {
    if (GetTileSize() == 1)
      return GetThresholdElementType();
    else
      return mlir::VectorType::get({GetTileSize()}, GetThresholdElementType());
  }
  virtual mlir::Value GetTreeIndex(Value tree) = 0;

  virtual mlir::Type GetIndexFieldType() {
    if (GetTileSize() == 1)
      return GetIndexElementType();
    else
      return mlir::VectorType::get({GetTileSize()}, GetIndexElementType());
  }

  virtual int32_t getTypeBitWidth(mlir::Type type) = 0;

  virtual void AddTypeConversions(mlir::MLIRContext &context,
                                  LLVMTypeConverter &typeConverter) = 0;
  virtual void AddLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) = 0;

  // Caching
  virtual void LowerCacheTreeOp(
      ConversionPatternRewriter &rewriter, mlir::Operation *op,
      ArrayRef<Value> operands,
      std::shared_ptr<decisionforest::IModelSerializer> m_serializer) = 0;

  virtual void LowerCacheRowsOp(ConversionPatternRewriter &rewriter,
                                mlir::Operation *op,
                                ArrayRef<Value> operands) = 0;
};

class ArrayBasedRepresentation : public IRepresentation {
protected:
  // TODO the names of the model and offset global should be generated so
  // they're unique for each ensemble constant
  // TODO the getter function names need to be persisted with the actual tree
  // values in the JSON so the runtime can call them.
  const std::string kModelMemrefName = "model";
  const std::string kOffsetMemrefName = "offsets";
  const std::string kLengthMemrefName = "lengths";
  const std::string kClassInfoMemrefName = "treeClassInfo";
  const std::string kThresholdsMemrefName = "thresholdValues";
  const std::string kFeatureIndexMemrefName = "featureIndexValues";
  const std::string kTileShapeMemrefName = "tileShapeValues";

  typedef struct Memrefs {
    mlir::Type model;
    mlir::Type offset;
    mlir::Type classInfo;
  } GlobalMemrefTypes;

  struct EnsembleConstantLoweringInfo {
    mlir::Value modelGlobal;
    mlir::Value offsetGlobal;
    mlir::Value lengthGlobal;
    mlir::Value classInfoGlobal;

    mlir::Type modelGlobalType;
    mlir::Type offsetGlobaltype;
    mlir::Type lengthGlobalType;
    mlir::Type classInfoType;
  };

  // Maps an ensemble constant operation to a model memref and an offsets memref
  std::map<mlir::Operation *, EnsembleConstantLoweringInfo>
      ensembleConstantToMemrefsMap;
  // Maps a GetTree operation to a memref that represents the tree once the
  // ensemble constant has been replaced
  std::map<mlir::Operation *, mlir::Value> getTreeOperationMap;

  int32_t m_tileSize = -1;
  mlir::Type m_thresholdType;
  mlir::Type m_featureIndexType;
  mlir::Type m_tileShapeType;

  void GenModelMemrefInitFunctionBody(MemRefType memrefType,
                                      Value getGlobalMemref,
                                      mlir::OpBuilder &builder,
                                      Location location, Value tileIndex,
                                      Value thresholdMemref, Value indexMemref,
                                      Value tileShapeIdMemref);
  GlobalMemrefTypes
  AddGlobalMemrefs(mlir::ModuleOp module,
                   mlir::decisionforest::EnsembleConstantOp &ensembleConstOp,
                   ConversionPatternRewriter &rewriter, Location location);

  void AddModelMemrefInitFunction(
      mlir::decisionforest::EnsembleConstantOp &ensembleConstOp,
      mlir::ModuleOp module, std::string globalName, MemRefType memrefType,
      ConversionPatternRewriter &rewriter, Location location);

  mlir::Value GetTreeMemref(mlir::Value treeValue);

public:
  virtual ~ArrayBasedRepresentation() {}
  void InitRepresentation() override;
  mlir::LogicalResult GenerateModelGlobals(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter,
      std::shared_ptr<decisionforest::IModelSerializer> m_serializer) override;
  virtual mlir::Value GetThresholdsMemref(mlir::Value treeValue) override {
    return GetTreeMemref(treeValue);
  }
  virtual mlir::Value GetFeatureIndexMemref(mlir::Value treeValue) override {
    return GetTreeMemref(treeValue);
  }
  virtual mlir::Value GetTileShapeMemref(mlir::Value treeValue) override {
    return GetTreeMemref(treeValue);
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
    return m_tileSize;
  }
  mlir::Type GetIndexElementType() override { return m_featureIndexType; }
  mlir::Type GetThresholdElementType() override { return m_thresholdType; }
  mlir::Type GetTileShapeType() override { return m_tileShapeType; }

  int32_t getTypeBitWidth(mlir::Type type) override;

  mlir::Value GetTreeIndex(Value tree) override;

  void AddTypeConversions(mlir::MLIRContext &context,
                          LLVMTypeConverter &typeConverter) override;
  void AddLLVMConversionPatterns(LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns) override;

  void LowerCacheTreeOp(
      ConversionPatternRewriter &rewriter, mlir::Operation *op,
      ArrayRef<Value> operands,
      std::shared_ptr<decisionforest::IModelSerializer> m_serializer) override {
  }

  void LowerCacheRowsOp(ConversionPatternRewriter &rewriter,
                        mlir::Operation *op, ArrayRef<Value> operands) override;
};

class SparseRepresentation : public IRepresentation {
protected:
  // TODO the names of the model and offset global should be generated so
  // they're unique for each ensemble constant
  // TODO the getter function names need to be persisted with the actual tree
  // values in the JSON so the runtime can call them.
  const std::string kModelMemrefName = "model";
  const std::string kOffsetMemrefName = "offsets";
  const std::string kLengthMemrefName = "lengths";
  const std::string kLeavesMemrefName = "leaves";
  const std::string kLeavesLengthMemrefName = "leavesLengths";
  const std::string kLeavesOffsetMemrefName = "leavesOffsets";
  const std::string kClassInfoMemrefName = "treeClassInfo";

  const std::string kThresholdsMemrefName = "thresholdValues";
  const std::string kFeatureIndexMemrefName = "featureIndexValues";
  const std::string kChildIndexMemrefName = "childIndexValues";
  const std::string kTileShapeMemrefName = "tileShapeValues";

  struct SparseEnsembleConstantLoweringInfo {
    mlir::Value modelGlobal;
    mlir::Value offsetGlobal;
    mlir::Value lengthGlobal;
    mlir::Value lutGlobal;
    mlir::Value leavesGlobal;
    mlir::Value leavesOffsetGlobal;
    mlir::Value leavesLengthGlobal;
    mlir::Value classInfoGlobal;

    mlir::Type modelGlobalType;
    mlir::Type offsetGlobaltype;
    mlir::Type lengthGlobalType;
    mlir::Type lutGlobalType;
    mlir::Type leavesGlobalType;
    mlir::Type classInfoType;
  };

  struct GetTreeLoweringInfo {
    mlir::Value treeMemref;
    mlir::Value leavesMemref;
  };

  // Maps an ensemble constant operation to a model memref and an offsets memref
  std::map<mlir::Operation *, SparseEnsembleConstantLoweringInfo>
      sparseEnsembleConstantToMemrefsMap;
  // Maps a GetTree operation to a memref that represents the tree once the
  // ensemble constant has been replaced
  std::map<mlir::Operation *, GetTreeLoweringInfo> sparseGetTreeOperationMap;

  int32_t m_tileSize = -1;
  mlir::Type m_thresholdType;
  mlir::Type m_featureIndexType;
  mlir::Type m_tileShapeType;
  mlir::Type m_childIndexType;

  void GenModelMemrefInitFunctionBody(
      MemRefType memrefType, Value getGlobalMemref, mlir::OpBuilder &builder,
      Location location, Value tileIndex, Value thresholdMemref,
      Value indexMemref, Value tileShapeIdMemref, Value childIndexMemref);

  mlir::Value GetTreeMemref(mlir::Value treeValue);

public:
  virtual ~SparseRepresentation() {}
  void InitRepresentation() override;
  mlir::LogicalResult GenerateModelGlobals(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter,
      std::shared_ptr<decisionforest::IModelSerializer> m_serializer) override;
  std::tuple<Type, Type, Type, Type>
  AddGlobalMemrefs(mlir::ModuleOp module,
                   mlir::decisionforest::EnsembleConstantOp &ensembleConstOp,
                   ConversionPatternRewriter &rewriter, Location location);

  void AddModelMemrefInitFunction(mlir::ModuleOp module, std::string globalName,
                                  MemRefType memrefType,
                                  ConversionPatternRewriter &rewriter,
                                  Location location);

  virtual mlir::Value GetThresholdsMemref(mlir::Value treeValue) override {
    return GetTreeMemref(treeValue);
  }
  virtual mlir::Value GetFeatureIndexMemref(mlir::Value treeValue) override {
    return GetTreeMemref(treeValue);
  }
  virtual mlir::Value GetTileShapeMemref(mlir::Value treeValue) override {
    return GetTreeMemref(treeValue);
  }

  mlir::Value GetLeafMemref(mlir::Value treeValue);
  std::vector<mlir::Value>
  GenerateExtraLoads(mlir::Location location,
                     ConversionPatternRewriter &rewriter, mlir::Value tree,
                     mlir::Value nodeIndex) override;
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
    return m_tileSize;
  }
  mlir::Type GetIndexElementType() override { return m_featureIndexType; }
  mlir::Type GetThresholdElementType() override { return m_thresholdType; }
  mlir::Type GetTileShapeType() override { return m_tileShapeType; }

  int32_t getTypeBitWidth(mlir::Type type) override;

  mlir::Value GetTreeIndex(Value tree) override;

  void AddTypeConversions(mlir::MLIRContext &context,
                          LLVMTypeConverter &typeConverter) override;
  void AddLLVMConversionPatterns(LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns) override;

  void LowerCacheTreeOp(
      ConversionPatternRewriter &rewriter, mlir::Operation *op,
      ArrayRef<Value> operands,
      std::shared_ptr<decisionforest::IModelSerializer> m_serializer) override {
  }

  void LowerCacheRowsOp(ConversionPatternRewriter &rewriter,
                        mlir::Operation *op, ArrayRef<Value> operands) override;
};

class RepresentationFactory {
  typedef std::shared_ptr<IRepresentation> (*RepresentationConstructor_t)();

private:
  std::map<std::string, RepresentationConstructor_t> m_constructionMap;

public:
  static RepresentationFactory &Get();
  bool RegisterRepresentation(const std::string &name,
                              RepresentationConstructor_t constructionFunc);

  std::shared_ptr<IRepresentation> GetRepresentation(const std::string &name);
};

#define REGISTER_REPRESENTATION(name, func)                                    \
  __attribute__((unused)) static bool UNIQUE_NAME(register_rep_) =             \
      RepresentationFactory::Get().RegisterRepresentation(#name, func);

// TODO This function needs to be removed
// Helper to construct the right representation to work around the
// global "UseSparseRepresentation"
std::shared_ptr<IRepresentation> ConstructRepresentation();
std::shared_ptr<IRepresentation> ConstructGPURepresentation();

template <typename T>
void createGlobalWithCorrectType(ConversionPatternRewriter &rewriter,
                                 Location location,
                                 const std::string &memrefName,
                                 mlir::MemRefType type, std::vector<T> &data);

template <typename T>
void createConstantGlobalOp(ConversionPatternRewriter &rewriter,
                            Location location, const std::string &memrefName,
                            mlir::MemRefType type, std::vector<T> &data) {
  if (type.getElementType().isIndex()) {
    if (sizeof(T) != sizeof(size_t)) {
      return createGlobalWithCorrectType(rewriter, location, memrefName, type,
                                         data);
    }
  } else if (type.getElementType().isIntOrFloat()) {
    if (std::is_floating_point<T>::value &&
        sizeof(T) * 8 != type.getElementTypeBitWidth()) {
      return createGlobalWithCorrectType(rewriter, location, memrefName, type,
                                         data);
    }
    if (std::is_integral<T>::value &&
        sizeof(T) * 8 != type.getElementTypeBitWidth()) {
      return createGlobalWithCorrectType(rewriter, location, memrefName, type,
                                         data);
    }
  } else {
    assert(false && "Unsupported type");
  }

  mlir::ArrayRef<T> dataArrayRef(data.data(), data.size());
  auto dataElementsAttribute = DenseElementsAttr::get(
      memref::getTensorTypeFromMemRefType(type), dataArrayRef);
  rewriter.create<memref::GlobalOp>(location, memrefName,
                                    rewriter.getStringAttr("private"), type,
                                    dataElementsAttribute, true, IntegerAttr());
}

template <typename T>
void createGlobalWithCorrectType(ConversionPatternRewriter &rewriter,
                                 Location location,
                                 const std::string &memrefName,
                                 mlir::MemRefType type, std::vector<T> &data) {
  if (type.getElementType().isInteger(sizeof(int64_t) * 8)) {
    std::vector<int64_t> int64Data(data.begin(), data.end());
    createConstantGlobalOp<int64_t>(rewriter, location, memrefName, type,
                                    int64Data);
  } else if (type.getElementType().isInteger(sizeof(int32_t) * 8)) {
    std::vector<int32_t> int32Data(data.begin(), data.end());
    createConstantGlobalOp<int32_t>(rewriter, location, memrefName, type,
                                    int32Data);
  } else if (type.getElementType().isInteger(sizeof(int16_t) * 8)) {
    std::vector<int16_t> int16Data(data.begin(), data.end());
    createConstantGlobalOp<int16_t>(rewriter, location, memrefName, type,
                                    int16Data);
  } else if (type.getElementType().isInteger(sizeof(int8_t) * 8)) {
    std::vector<int8_t> int8Data(data.begin(), data.end());
    createConstantGlobalOp<int8_t>(rewriter, location, memrefName, type,
                                   int8Data);
  } else if (type.getElementType().isIndex()) {
    std::vector<size_t> indexData(data.begin(), data.end());
    createConstantGlobalOp<size_t>(rewriter, location, memrefName, type,
                                   indexData);
  } else if (type.getElementType().isF64()) {
    std::vector<double> floatData(data.begin(), data.end());
    createConstantGlobalOp<double>(rewriter, location, memrefName, type,
                                   floatData);
  } else if (type.getElementType().isF32()) {
    std::vector<float> floatData(data.begin(), data.end());
    createConstantGlobalOp<float>(rewriter, location, memrefName, type,
                                  floatData);
  } else {
    assert(false && "Unsupported type");
  }
}

} // namespace decisionforest
} // namespace mlir

#endif