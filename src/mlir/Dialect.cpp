#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "Dialect.h"

using namespace mlir;
using namespace mlir::decisionforest;

#include "Dialect.cpp.inc"

// Initialize the dialect
void DecisionForestDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
  addTypes<TreeEnsembleType, TreeType>();
  addAttributes<DecisionTreeAttr, DecisionForestAttribute>();
}

/// Parse a type registered to this dialect.
::mlir::Type DecisionForestDialect::parseType(::mlir::DialectAsmParser &parser) const
{
    return ::mlir::Type();
}

/// Print a type registered to this dialect.
void DecisionForestDialect::printType(::mlir::Type type,
                                      ::mlir::DialectAsmPrinter &os) const
{
    // mlir::decisionforest::Decs

}

mlir::Attribute DecisionForestDialect::parseAttribute(DialectAsmParser &parser, 
                                                      Type type) const
{
    llvm_unreachable("DecisionForestDialect::parseAttribute Unimplement");
    return mlir::Attribute();
}

void DecisionForestDialect::printAttribute(::mlir::Attribute attr,
                                           ::mlir::DialectAsmPrinter &os) const
{
    DecisionForestAttribute decisionForrestAttr = attr.cast<DecisionForestAttribute>();
    os << decisionForrestAttr.PrintToString();
}

#define GET_OP_CLASSES
#include "Ops.cpp.inc"