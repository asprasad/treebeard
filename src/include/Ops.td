// -- Dialect specification for tree-heavy --
#ifndef TREE_HEAVY_OPS
#define TREE_HEAVY_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

def DecisionForestDialect : Dialect {
  let name = "decisionforest";
  let cppNamespace = "::mlir::decisionforest";
  let summary = "A high-level dialect to specify decision trees";
}

// def NodeType : 
//     DialectType<DecisionForestDialect, CPred<"$_self.isa<NodeType>()">,
//     "NodeType">;


// def NumericalNodeType :
//     DialectType<DecisionForestDialect, CPred<"$_self.isa<NumericalNodeType>()">,
//                 "NumericalNodeType">;

// def CategoricalNodeType :
//     DialectType<DecisionForestDialect, CPred<"$_self.isa<CategoricalNodeType>()">,
//                 "CategoricalNodeType">;

// def LeafNodeType :
//     DialectType<DecisionForestDialect, CPred<"$_self.isa<LeafNodeType>()">,
//                 "LeafNodeType">;

def TreeType : 
    DialectType<DecisionForestDialect, CPred<"$_self.isa<TreeType>()">,
    "TreeType">;

def TreeEnsembleType : 
    DialectType<DecisionForestDialect, CPred<"$_self.isa<TreeEnsembleType>()">,
    "TreeEnsembleType">;

class DecisionForest_Op<string mnemonic, list<OpTrait> traits = []> : 
  Op<DecisionForestDialect, mnemonic, traits>;

// class NodeConstantOp<string mnemonic, list<OpTrait> traits = []> : 
//   DecisionForest_Op<mnemonic, traits> {
//     string myStr = "";
// }

// def NumericalNodeConstantOp : NodeConstantOp<"numerical", [NoSideEffect]> {
//   let arguments = (ins F64ElementsAttr:$threshold, I32ElementsAttr:$featureIndex, NodeType:$left, NodeType:$right);
//   let results = (outs NumericalNodeType);
//   let myStr = "abc";
// }

// def CategoricalNodeConstantOp : NodeConstantOp<"numerical"> {
//   
// }

// def LeafNodeConstantOp : NodeConstantOp<"numerical", [NoSideEffect]> {
//   let arguments = (ins F64ElementsAttr:$value);
//   let results = (outs LeafNodeType);
// }

def DecisionTreeAttr : Attr<
  CPred<"$_self.isa<::mlir::decisionforest::DecisionTreeAttr>()">,
  "Decision tree attribute"> {

  let storageType = [{ ::mlir::decisionforest::DecisionTreeAttr }];
  let returnType = [{ ::mlir::decisionforest::DecisionTreeAttr }];

  // Note that this is only constructing scalar elements attribute.
  let constBuilderCall = "::mlir::decisionforest::DecisionTreeAttr::get()";
  let convertFromStorage = "$_self";
}

def TreeConstantOp : DecisionForest_Op<"tree", [NoSideEffect]> {
  // [TODO] We need to define some type that represents how we store the actual tree value
  let arguments = (ins DecisionTreeAttr:$value);

  let results = (outs TreeType);
  // [TODO] Need to define the following methods
  // 1. Add node (threshold, featureIndex) 2. Set parent/children on node 3. 
}

def DecisionForestAttr : Attr<
  CPred<"$_self.isa<::mlir::decisionforest::DecisionForestAttribute>()">,
  "Decision forest attribute"> {

  let storageType = [{ ::mlir::decisionforest::DecisionForestAttribute }];
  let returnType = [{ ::mlir::decisionforest::DecisionForestAttribute }];

  // Note that this is only constructing scalar elements attribute.
  let constBuilderCall = "::mlir::decisionforest::DecisionForestAttribute::get()";
  let convertFromStorage = "$_self";
}

def EnsembleConstantOp : DecisionForest_Op<"ensemble"> {

}

// TODO Need to parameterize 
// 1. The return type
// 2. The number of return values (may have multiple return probablities, class ID etc)
// 3. The type of the input tensor (need not be F64 always)
def PredictForestOp : DecisionForest_Op<"predict_ensemble"> {
    let arguments = (ins DecisionForestAttr:$ensemble, F64Tensor:$data);

    let results = (outs F64);

    // let builders = [
    //   // Build a constant with a given constant tensor value.
    //   OpBuilder<(ins "DecisionForestAttribute":$ensemble, "Value":$data)>
    // ];
}

def ReturnOp : DecisionForest_Op<"return", [NoSideEffect, HasParent<"FuncOp">,Terminator]> {
  let summary = "return operation";
  let description = "Represents function return operation. Takes optional tensor operand and produces no results.";
  let arguments = (ins F64:$input); 
  // let assemblyFormat = "($input^ `:` type($input)) ? attr-dict";

  // let builders = [
  //   OpBuilder<(ins "Value":$input)> // TODO: What's this?
  // ];

  // let extraClassDeclaration = [{
  //   bool hasOperand() { return getNumOperands() != 0; }
  // }];

  // let verifier = [{return ::verify(*this); }];
}

// function Predict(x)
// { 
//   %1 = predict_const_ensemble(x) // Attr = ensemble 
//   return %1
// }



#endif // TREE_HEAVY_OPS