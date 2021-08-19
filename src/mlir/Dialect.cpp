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
  addTypes<TreeEnsembleType, TreeType, NodeType, LeafNodeType, NumericalNodeType>();
  addAttributes<DecisionTreeAttr, DecisionForestAttribute>();
}

/// Parse a type registered to this dialect.
::mlir::Type DecisionForestDialect::parseType(::mlir::DialectAsmParser &parser) const
{
    return ::mlir::Type();
}

// TODO Can this somehow be made polymorphic? MLIR seems to pass everything by value!
/// Print a type registered to this dialect.
void DecisionForestDialect::printType(::mlir::Type type,
                                      ::mlir::DialectAsmPrinter &os) const
{
    
    // TODO how do you handle numerical and leaf node types here? They will just 
    // get printed as NodeType now
    if (type.isa<mlir::decisionforest::NodeType>()) {
        mlir::decisionforest::NodeType nodeType = type.cast<mlir::decisionforest::NodeType>();
        nodeType.print(os);
    }
    else if(type.isa<mlir::decisionforest::TreeEnsembleType>()) {
        mlir::decisionforest::TreeEnsembleType ensembleType = type.cast<mlir::decisionforest::TreeEnsembleType>();
        ensembleType.print(os);
    }
    else if(type.isa<mlir::decisionforest::TreeType>()) {
        mlir::decisionforest::TreeType treeType = type.cast<mlir::decisionforest::TreeType>();
        treeType.print(os);
    }
    else
        assert(false && "Invalid decisionforest dialect type");
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
    decisionForrestAttr.Print(os);
}

#define GET_OP_CLASSES
#include "Ops.cpp.inc"