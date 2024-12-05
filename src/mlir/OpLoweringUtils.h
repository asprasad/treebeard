#ifndef OP_LOWERING_UTILS
#define OP_LOWERING_UTILS

#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"

template <typename T>
T AssertOpIsOfType(mlir::Operation *operation) {
    T typedOp = llvm::dyn_cast<T>(operation);
    if (!typedOp) {
        llvm::errs() << "Operation type mismatch: expected " << T::getOperationName() << "\n";
        operation->dump(); // Print information about the actual operation type
        assert(false && "Operation is not of the expected type.");
    }
    return typedOp;
}


inline bool isConstantInt(mlir::Value val, int64_t &value) {
  llvm::APInt intVal;
  if (mlir::matchPattern(val, mlir::m_ConstantInt(&intVal))) {
    value = intVal.getSExtValue();  // Extracts the int64_t value
    return true;
  }
  return false;
}


inline int64_t GetConstantIntValueFromMLIRValue(mlir::Value val) {
  int64_t value;
  auto match = isConstantInt(val, value);
  assert(match);
  return value;
}

int64_t getConstantStepBetweenValues(mlir::Value start, mlir::Value end);

inline bool isLoopRangeOne(mlir::Value start, mlir::Value end) {
  if (start.getDefiningOp() && end.getDefiningOp() &&
      start.getDefiningOp()->hasTrait<mlir::OpTrait::ConstantLike>() &&
      end.getDefiningOp()->hasTrait<mlir::OpTrait::ConstantLike>()) {

    auto startConst = GetConstantIntValueFromMLIRValue(start);
    auto endConst = GetConstantIntValueFromMLIRValue(end);
    return (endConst - startConst) == 1;
  }

  return false;
}

#endif // OP_LOWERING_UTILS