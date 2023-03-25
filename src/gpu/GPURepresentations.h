#ifndef _GPUREPRESENTATIONS_H_
#define _GPUREPRESENTATIONS_H_

#include "Representations.h"

namespace mlir
{
namespace decisionforest
{

class GPUArrayBasedRepresentation : public ArrayBasedRepresentation {
protected:
  Value m_modelMemref;
  int32_t m_modelMemrefArgIndex;  
  int32_t m_offsetMemrefArgIndex;
  int32_t m_lengthMemrefArgIndex;
  int32_t m_classInfoMemrefArgIndex;

  struct CacheTreesInfo {
    Value cachedModelBuffer;
  };

  std::map<mlir::Operation*, CacheTreesInfo> m_cacheTreesOpsMap;

  void GenerateModelMemrefInitializer(const std::string& funcName, ConversionPatternRewriter &rewriter, Location location, 
                                      ModuleOp module, MemRefType memrefType);

public:
  virtual ~GPUArrayBasedRepresentation() { }
  void InitRepresentation() override { }
  mlir::LogicalResult GenerateModelGlobals(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter,
                                           std::shared_ptr<decisionforest::IModelSerializer> m_serializer) override;

  void GenerateTreeMemref(mlir::ConversionPatternRewriter &rewriter, 
                          mlir::Operation *op, 
                          Value ensemble,
                          Value treeIndex) override;

  // mlir::Value GenerateGetTreeClassId(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) override;
  void LowerCacheTreeOp(ConversionPatternRewriter &rewriter, 
                        mlir::Operation *op,
                        ArrayRef<Value> operands,
                        std::shared_ptr<decisionforest::IModelSerializer> m_serializer) override;
};

class GPUSparseRepresentation : public SparseRepresentation {
protected:
  Value m_modelMemref;
  int32_t m_modelMemrefArgIndex;  
  int32_t m_offsetMemrefArgIndex;
  int32_t m_lengthMemrefArgIndex;
  int32_t m_classInfoMemrefArgIndex;

  void GenerateModelMemrefInitializer(const std::string& funcName, ConversionPatternRewriter &rewriter, Location location, 
                                      ModuleOp module, MemRefType memrefType);

public:
  virtual ~GPUSparseRepresentation() { }
  void InitRepresentation() override { }
  mlir::LogicalResult GenerateModelGlobals(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter,
                                           std::shared_ptr<decisionforest::IModelSerializer> m_serializer) override;

  // mlir::Value GenerateGetTreeClassId(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) override;
  void LowerCacheTreeOp(ConversionPatternRewriter &rewriter, 
                        mlir::Operation *op,
                        ArrayRef<Value> operands,
                        std::shared_ptr<decisionforest::IModelSerializer> m_serializer) override { assert(false); }
};

} // decisionforest
} // mlir

#endif // _GPUREPRESENTATIONS_H_