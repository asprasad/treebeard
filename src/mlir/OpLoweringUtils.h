#ifndef OP_LOWERING_UTILS
#define OP_LOWERING_UTILS

#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"

template <typename T> T AssertOpIsOfType(mlir::Operation *operation) {
  T typedOp = llvm::dyn_cast<T>(operation);
  assert(typedOp);
  return typedOp;
}

inline bool isContantInt(mlir::Value val, int64_t &value) {
  auto definingOp = val.getDefiningOp();
  llvm::APInt constIntVal;
  mlir::detail::constant_int_op_binder binder(&constIntVal);
  bool match = binder.match(definingOp);
  value = constIntVal.getLimitedValue();
  return match;
}

inline int64_t GetConstantIntValueFromMLIRValue(mlir::Value val) {
  int64_t value;
  auto match = isContantInt(val, value);
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