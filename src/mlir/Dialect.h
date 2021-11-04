#ifndef _DIALECT_H_
#define _DIALECT_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "DecisionTreeAttributes.h"
#include "DecisionTreeTypes.h"
#include "MemrefTypes.h"

#include "Dialect.h.inc"

#define GET_OP_CLASSES
#include "Ops.h.inc"

namespace mlir
{

class RewritePatternSet;
class LLVMTypeConverter;

namespace decisionforest
{
extern bool InsertDebugHelpers;
void populateDebugOpLoweringPatterns(RewritePatternSet& patterns, LLVMTypeConverter& typeConverter);

void LowerFromHighLevelToMidLevelIR(mlir::MLIRContext& context, mlir::ModuleOp module);
void LowerEnsembleToMemrefs(mlir::MLIRContext& context, mlir::ModuleOp module);
void ConvertNodeTypeToIndexType(mlir::MLIRContext& context, mlir::ModuleOp module);
void LowerToLLVM(mlir::MLIRContext& context, mlir::ModuleOp module);
int dumpLLVMIR(mlir::ModuleOp module);

// Optimizing passes
void DoUniformTiling(mlir::MLIRContext& context, mlir::ModuleOp module, int32_t tileSize);

}
}

#endif // _DIALECT_H_