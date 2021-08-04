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

/// Include the auto-generated header file containing the declaration of the sparse tensor dialect.
#include "Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the sparse tensor operations.
#define GET_OP_CLASSES
#include "Ops.h.inc"


#endif // _DIALECT_H_