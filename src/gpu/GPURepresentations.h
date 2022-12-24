#ifndef _GPUREPRESENTATIONS_H_
#define _GPUREPRESENTATIONS_H_

#include "Representations.h"

namespace mlir
{
namespace decisionforest
{

// class GPUArrayBasedRepresentation : public IRepresentation {
// protected:

//   void AddModelMemrefInitFunction(mlir::ModuleOp module, std::string globalName, MemRefType memrefType, 
//                                   ConversionPatternRewriter &rewriter, Location location);

// public:
//   virtual ~GPUArrayBasedRepresentation() { }
//   void InitRepresentation() { }
//   mlir::LogicalResult GenerateModelGlobals(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter,
//                                            std::shared_ptr<decisionforest::IModelSerializer> m_serializer) override;

//   mlir::Value GetTreeMemref(mlir::Value treeValue) override;
//   void GenerateTreeMemref(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) override;
//   mlir::Value GenerateGetTreeClassId(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) override;
// };


} // decisionforest
} // mlir

#endif // _GPUREPRESENTATIONS_H_