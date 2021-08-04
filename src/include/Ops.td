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

// TODO Need to parameterize the return type and the number of return values
// (may have multiple return probablities, class ID etc)
class PredictForestOp : DecisionForest_Op<"predict_ensemble"> {
    let arguments = (ins DecisionForestAttr:$ensemble, F64ElementsAttr:$data);

    let results = (outs F64);
    // [TODO] Need to define the following methods
    // 1. Add node (threshold, featureIndex) 2. Set parent/children on node 3. 
}


// function Predict(x)
// { 
//   %1 = predict_const_ensemble(x) // Attr = ensemble 
//   return %1
// }

#endif // TREE_HEAVY_OPS