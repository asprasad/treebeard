#ifndef _LOWERREDUCEOPS_H_
#define _LOWERREDUCEOPS_H_

namespace mlir {

class MLIRContext;
class ModuleOp;

namespace decisionforest {

void legalizeReductions(mlir::MLIRContext &context, mlir::ModuleOp module);
void legalizeReductionsAndCanonicalize(mlir::MLIRContext &context,
                                       mlir::ModuleOp module);
void lowerLinalgToLoops(mlir::MLIRContext &context, mlir::ModuleOp module);
void lowerReductionsAndCanonicalize(mlir::MLIRContext &context,
                                    mlir::ModuleOp module);
void runConvertToCooperativeReducePass(mlir::MLIRContext &context,
                                       mlir::ModuleOp module);
void convertToCooperativeReduceAndCanonicalize(mlir::MLIRContext &context,
                                               mlir::ModuleOp module);

} // end namespace decisionforest
} // end namespace mlir

#endif // _LOWERREDUCEOPS_H_