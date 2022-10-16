#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "Dialect.h"

using namespace mlir;
using namespace mlir::decisionforest;

#include "Dialect.cpp.inc"

bool mlir::decisionforest::InsertDebugHelpers = false;
bool mlir::decisionforest::PrintVectors = false;
bool mlir::decisionforest::EnablePerfNotificationListener = false;

bool mlir::decisionforest::UseBitcastForComparisonOutcome = true;
bool mlir::decisionforest::UseSparseTreeRepresentation = false;
bool mlir::decisionforest::PeeledCodeGenForProbabiltyBasedTiling = false;

void TreeTypeStorage::print(mlir::DialectAsmPrinter &printer) {
    printer << "TreeType(returnType:" << m_resultType << ", tileSize:" << m_tileSize 
            << ", tileShapeType:" << m_tileShapeType << ", sparse:" << m_sparseRepresentation 
            << ", childIndexType:" << m_childIndexType << "))";
}

void TreeEnsembleTypeStorage::print(mlir::DialectAsmPrinter &printer) {
    printer << "TreeEnsembleType(#Trees:" << m_numTrees << ", rowType:" << m_rowType 
            << ", resultType:" << m_resultType << ", reductionType:" << (int32_t)m_reductionType << ")";
}

void NumericalNodeTypeStorage::print(mlir::DialectAsmPrinter &printer) {
    printer << "NumericalNodeType";
 }

void LeafNodeTypeStorage::print(mlir::DialectAsmPrinter &printer) {
    printer << "LeafNodeType";
 }

// Initialize the dialect
void DecisionForestDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
  addTypes<TreeEnsembleType, TreeType, NodeType, LeafNodeType, NumericalNodeType, TiledNumericalNodeType, ScheduleType>();
  addAttributes<DecisionTreeAttribute, DecisionForestAttribute, ScheduleAttribute, UnrollLoopAttribute>();
}

/// Parse a type registered to this dialect.
::mlir::Type DecisionForestDialect::parseType(::mlir::DialectAsmParser &parser) const
{
    llvm_unreachable("DecisionForestDialect::parseAttribute Unimplement");
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
    else if(type.isa<mlir::decisionforest::NumericalNodeType>()) {
        auto numericalNodeType = type.cast<mlir::decisionforest::NumericalNodeType>();
        numericalNodeType.print(os);
    }
    else if(type.isa<mlir::decisionforest::LeafNodeType>()) {
        auto leafNodeType = type.cast<mlir::decisionforest::LeafNodeType>();
        leafNodeType.print(os);
    }
    else if(type.isa<mlir::decisionforest::TiledNumericalNodeType>()) {
        auto tiledNodeType = type.cast<mlir::decisionforest::TiledNumericalNodeType>();
        tiledNodeType.print(os);
    }
    else if(type.isa<mlir::decisionforest::ScheduleType>()) {
        auto scheduleType = type.cast<mlir::decisionforest::ScheduleType>();
        scheduleType.print(os);
    }
    else {
        llvm_unreachable("Invalid decisionforest dialect type");
    }
}

mlir::Attribute DecisionForestDialect::parseAttribute(DialectAsmParser &parser, 
                                                      Type type) const
{
    llvm_unreachable("DecisionForestDialect::parseAttribute unimplemented");
    return mlir::Attribute();
}

void DecisionForestDialect::printAttribute(::mlir::Attribute attr,
                                           ::mlir::DialectAsmPrinter &os) const
{
    if (attr.isa<DecisionForestAttribute>()) 
    {
        DecisionForestAttribute decisionForrestAttr = attr.cast<DecisionForestAttribute>();
        decisionForrestAttr.Print(os);
    }
    else if (attr.isa<DecisionTreeAttribute>())
    {
        DecisionTreeAttribute decisionTreeAttr = attr.cast<DecisionTreeAttribute>();
        decisionTreeAttr.Print(os);
    }
    else if (attr.isa<ScheduleAttribute>())
    {
        auto scheduleAttr = attr.cast<ScheduleAttribute>();
        scheduleAttr.Print(os);
    }
    else if (attr.isa<UnrollLoopAttribute>())
    {
        auto unrollLoopAttr = attr.cast<UnrollLoopAttribute>();
        unrollLoopAttr.Print(os);
    }
    else
    {
        assert (false && "Unknow attribute");
    }
}

#define GET_OP_CLASSES
#include "Ops.cpp.inc"