#ifndef _REPRESENTATIONS_H_
#define _REPRESENTATIONS_H_

#include <vector>

#include "mlir/Transforms/DialectConversion.h"
#include "TreebeardContext.h"

namespace mlir
{
namespace decisionforest
{

class IRepresentation {
public:
  virtual ~IRepresentation() { }
  virtual void InitRepresentation()=0;
  virtual mlir::LogicalResult GenerateModelGlobals(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter,
                                                   std::shared_ptr<decisionforest::IModelSerializer> m_serializer) = 0;
  virtual mlir::Value GetTreeMemref(mlir::Value treeValue) = 0;
  virtual std::vector<mlir::Value> GenerateExtraLoads(mlir::Location location, ConversionPatternRewriter &rewriter, mlir::Value treeMemref, mlir::Value nodeIndex, mlir::Type treeTileType)=0;
  virtual mlir::Value GenerateMoveToChild(mlir::Location location, ConversionPatternRewriter &rewriter, mlir::Value nodeIndex,
                                          mlir::Value childNumber, int32_t tileSize, std::vector<mlir::Value>& extraLoads)=0;
  virtual void GenerateTreeMemref(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex)=0;
  virtual mlir::Value GenerateGetTreeClassId(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex)=0;
  virtual mlir::Value GenerateGetLeafValueOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, 
                                             mlir::Value nodeIndex)=0;
  virtual mlir::Value GenerateIsLeafOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex)=0;
  virtual mlir::Value GenerateIsLeafTileOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex)=0;
};


class ArrayBasedRepresentation : public IRepresentation {
protected:
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
  std::map<mlir::Operation*, EnsembleConstantLoweringInfo> ensembleConstantToMemrefsMap;
  // Maps a GetTree operation to a memref that represents the tree once the ensemble constant has been replaced
  std::map<mlir::Operation*, mlir::Value> getTreeOperationMap;

  void GenModelMemrefInitFunctionBody(MemRefType memrefType, Value memrefValue,
                                      mlir::OpBuilder &rewriter, Location location, Value tileIndex,
                                      Value thresholdMemref, Value indexMemref, Value tileShapeIdMemref);
  GlobalMemrefTypes AddGlobalMemrefs(
    mlir::ModuleOp module,
    mlir::decisionforest::EnsembleConstantOp& ensembleConstOp,
    ConversionPatternRewriter &rewriter,
    Location location,
    const std::string& modelMemrefName,
    const std::string& offsetMemrefName,
    const std::string& lengthMemrefName,
    const std::string& treeInfo,
    std::shared_ptr<decisionforest::IModelSerializer> serializer);

  void AddModelMemrefInitFunction(mlir::ModuleOp module, std::string globalName, MemRefType memrefType, 
                                  ConversionPatternRewriter &rewriter, Location location);

public:
  virtual ~ArrayBasedRepresentation() { }
  void InitRepresentation() override;
  mlir::LogicalResult GenerateModelGlobals(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter,
                                           std::shared_ptr<decisionforest::IModelSerializer> m_serializer) override;

  mlir::Value GetTreeMemref(mlir::Value treeValue) override;
  std::vector<mlir::Value> GenerateExtraLoads(mlir::Location location, ConversionPatternRewriter &rewriter, mlir::Value treeMemref, 
                                              mlir::Value nodeIndex, mlir::Type treeTileType) override { return std::vector<mlir::Value>(); }
  mlir::Value GenerateMoveToChild(mlir::Location location, ConversionPatternRewriter &rewriter, mlir::Value nodeIndex, 
                                  mlir::Value childNumber, int32_t tileSize, std::vector<mlir::Value>& extraLoads) override;
  void GenerateTreeMemref(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) override;
  mlir::Value GenerateGetTreeClassId(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) override;
  mlir::Value GenerateGetLeafValueOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, 
                                     mlir::Value nodeIndex) override;
  mlir::Value GenerateIsLeafOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex) override;
  mlir::Value GenerateIsLeafTileOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex) override;
};

class SparseRepresentation : public IRepresentation {
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
  std::map<mlir::Operation*, SparseEnsembleConstantLoweringInfo> sparseEnsembleConstantToMemrefsMap;
  // Maps a GetTree operation to a memref that represents the tree once the ensemble constant has been replaced
  std::map<mlir::Operation*, GetTreeLoweringInfo> sparseGetTreeOperationMap;

public:
  virtual ~SparseRepresentation() { }
  void InitRepresentation() override;
  mlir::LogicalResult GenerateModelGlobals(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter,
                                           std::shared_ptr<decisionforest::IModelSerializer> m_serializer) override;
  std::tuple<Type, Type, Type, Type> AddGlobalMemrefs(mlir::ModuleOp module, mlir::decisionforest::EnsembleConstantOp& ensembleConstOp,
                                                      ConversionPatternRewriter &rewriter, Location location,
                                                      const std::string& modelMemrefName, const std::string& offsetMemrefName,
                                                      const std::string& lengthMemrefName,
                                                      const std::string& leavesMemrefName, const std::string& leavesLengthMemrefName,
                                                      const std::string& leavesOffsetMemrefName, const std::string& treeInfo,
                                                      std::shared_ptr<IModelSerializer> serializer);

  void AddModelMemrefInitFunction(mlir::ModuleOp module, std::string globalName, MemRefType memrefType, 
                                  ConversionPatternRewriter &rewriter, Location location);
  mlir::Value GetTreeMemref(mlir::Value treeValue) override;
  mlir::Value GetLeafMemref(mlir::Value treeValue);
  std::vector<mlir::Value> GenerateExtraLoads(mlir::Location location, ConversionPatternRewriter &rewriter, mlir::Value treeMemref, mlir::Value nodeIndex, mlir::Type treeTileType) override;
  mlir::Value GenerateMoveToChild(mlir::Location location, ConversionPatternRewriter &rewriter, mlir::Value nodeIndex, 
                                  mlir::Value childNumber, int32_t tileSize, std::vector<mlir::Value>& extraLoads) override;
  void GenerateTreeMemref(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) override;
  mlir::Value GenerateGetTreeClassId(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) override;
  mlir::Value GenerateGetLeafValueOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, 
                                     mlir::Value nodeIndex) override;
  mlir::Value GenerateIsLeafOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex) override;
  mlir::Value GenerateIsLeafTileOp(ConversionPatternRewriter &rewriter, mlir::Operation *op, mlir::Value treeValue, mlir::Value nodeIndex) override;
};

class RepresentationFactory {
public:
  static std::shared_ptr<IRepresentation> GetRepresentation(const std::string& name);
};

// TODO This function needs to be removed
// Helper to construct the right representation to work around the 
// global "UseSparseRepresentation"
std::shared_ptr<IRepresentation> ConstructRepresentation();


} // decisionforest
} // mlir

#endif