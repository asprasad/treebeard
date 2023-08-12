#ifndef _LOWERREDUCEOPS_H_
#define _LOWERREDUCEOPS_H_

namespace mlir
{

class MLIRContext;
class ModuleOp;

namespace decisionforest
{

void LowerReduceOps(mlir::MLIRContext& context, mlir::ModuleOp module);

} // end namespace decisionforest
} // end namespace mlir

#endif // _LOWERREDUCEOPS_H_