#ifndef OP_LOWERING_UTILS
#define OP_LOWERING_UTILS

#include "mlir/IR/Operation.h"
#include "llvm/Support/Casting.h"

template<typename T>
T AssertOpIsOfType(mlir::Operation* operation) {
  T typedOp = llvm::dyn_cast<T>(operation);
  assert(typedOp);
  return typedOp;
}

#endif // OP_LOWERING_UTILS