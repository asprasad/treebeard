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
#include "ScheduleAttribute.h"

#include "Dialect.h.inc"

#define GET_OP_CLASSES
#include "Ops.h.inc"

#define OMP_SUPPORT

namespace mlir
{

class RewritePatternSet;
class LLVMTypeConverter;

namespace decisionforest
{
class IModelSerializer;
class IRepresentation;

// Debugging and profiling configuration
extern bool InsertDebugHelpers;
extern bool PrintVectors;
extern bool EnablePerfNotificationListener;

// Compiler configuration
extern bool UseBitcastForComparisonOutcome;
extern bool UseSparseTreeRepresentation;
extern bool PeeledCodeGenForProbabiltyBasedTiling;

void populateDebugOpLoweringPatterns(RewritePatternSet& patterns, LLVMTypeConverter& typeConverter);

void LowerFromHighLevelToMidLevelIR(mlir::MLIRContext& context, mlir::ModuleOp module);
void LowerEnsembleToMemrefs(mlir::MLIRContext& context, mlir::ModuleOp module, std::shared_ptr<IModelSerializer> serializer, std::shared_ptr<IRepresentation> representation);
void ConvertNodeTypeToIndexType(mlir::MLIRContext& context, mlir::ModuleOp module);
void LowerToLLVM(mlir::MLIRContext& context, mlir::ModuleOp module);
int dumpLLVMIR(mlir::ModuleOp module, bool dumpAsm = false);
int dumpLLVMIRToFile(mlir::ModuleOp module, const std::string& filename);

// Optimizing passes
void DoUniformTiling(mlir::MLIRContext& context, mlir::ModuleOp module, int32_t tileSize, int32_t tileShapeBitWidth, bool makeAllLeavesSameDepth);
void DoProbabilityBasedTiling(mlir::MLIRContext& context, mlir::ModuleOp module, int32_t tileSize, int32_t tileShapeBitWidth);
void DoHybridTiling(mlir::MLIRContext& context, mlir::ModuleOp module, int32_t tileSize, int32_t tileShapeBitWidth);
void DoReorderTreesByDepth(mlir::MLIRContext& context, mlir::ModuleOp module, int32_t pipelineSize=-1, int32_t numCores=-1);

}
}

#endif // _DIALECT_H_