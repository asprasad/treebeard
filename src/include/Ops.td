// -- Dialect specification for tree-heavy --

// Function generated in the high-level dialect 
// function Predict(x)
// { 
//   %1 = predict_const_ensemble(x) // Attr = ensemble 
//   return %1
// }

#ifndef TREE_HEAVY_OPS
#define TREE_HEAVY_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

def DecisionForestDialect : Dialect {
  let name = "decisionforest";
  let cppNamespace = "::mlir::decisionforest";
  let summary = "A high-level dialect to specify and optimize decision trees";

  let extraClassDeclaration = [{
    Attribute parseAttribute(DialectAsmParser &parser,
                             Type type) const override;
    void printAttribute(Attribute attr,
                        DialectAsmPrinter &printer) const override;
  }];
}

//-------   Types   ----------------------
def NodeType : 
    DialectType<DecisionForestDialect, CPred<"$_self.isa<NodeType>()">,
    "NodeType">;

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

def TiledNumericalNodeType : 
    DialectType<DecisionForestDialect, CPred<"$_self.isa<TiledNumericalNodeType>()">,
    "TiledNumericalNodeType">;

//-------   Attributes   ----------------------

def DecisionTreeAttr : Attr<
  CPred<"$_self.isa<::mlir::decisionforest::DecisionTreeAttribute>()">,
  "Decision tree attribute"> {

  let storageType = [{ ::mlir::decisionforest::DecisionTreeAttribute }];
  let returnType = [{ ::mlir::decisionforest::DecisionTreeAttribute }];

  // Note that this is only constructing scalar elements attribute.
  let constBuilderCall = "::mlir::decisionforest::DecisionTreeAttribute::get()";
  let convertFromStorage = "$_self";
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

//-------   Operations   ----------------------

class DecisionForest_Op<string mnemonic, list<OpTrait> traits = []> : 
  Op<DecisionForestDialect, mnemonic, traits>;


//------- High level IR Ops ------------------
// TODO Need to parameterize 
// * The number of return values (may have multiple return probablities, class ID etc)
def ReturnType : AnyTypeOf<[F16, F32, F64, F16Tensor, F32Tensor, F64Tensor]>;
def LeafNodeValueType : AnyTypeOf<[F16, F32, F64]>;

def PredictForestOp : DecisionForest_Op<"predict_ensemble"> {
    let summary = "Predict forest operation";
    let description = "Represents the actual inference computation on the forest. Takes a tensor operand and produces a result."
                      "Contains the actual forest as an attribute.";

    let arguments = (ins DecisionForestAttr:$ensemble, F64Tensor:$data);

    let results = (outs ReturnType);

    let extraClassDeclaration = [{
      int32_t getBatchSize() { 
        auto tensorReturnType = data().getType().cast<mlir::RankedTensorType>();
        if (!tensorReturnType)
          return 1;
        return tensorReturnType.getShape()[0];
      }
    }];
}

def ReturnOp : DecisionForest_Op<"return", [NoSideEffect, HasParent<"FuncOp">,Terminator]> {
  let summary = "return operation";
  let description = "Represents function return operation. Takes optional tensor operand and produces no results.";
  let arguments = (ins ReturnType:$input); 
  // let verifier = [{return ::verify(*this); }];
}

// ----- Mid-level IR Ops ------

def ThresholdValueType : AnyTypeOf<[F16, F32, F64]>;
def FeatureIndexType : AnyTypeOf<[I8, I16, I32, I64]>;

def TreeConstantOp : DecisionForest_Op<"tree", [NoSideEffect]> {
  let summary = "Tree constant operation";
  let description = "Represents a constant whose value is a single tree in the forest. Takes no arguments and gives a tree value out.";

  let arguments = (ins DecisionTreeAttr:$tree);

  let results = (outs TreeType);
}

def EnsembleConstantOp : DecisionForest_Op<"ensemble", [NoSideEffect]> {
  let summary = "Ensemble constant operation";
  let description = "Represents a constant whose value is the full forest. Takes no arguments and gives an ensemble value out.";
  let arguments = (ins DecisionForestAttr:$forest);

  let results = (outs TreeEnsembleType);
}

def GetTreeFromEnsembleOp : DecisionForest_Op<"getTree", [NoSideEffect]> {
  let summary = "Get single tree from the forest.";
  let description = "Index the forest and get a single tree. Takes two arguments -- an ensemble and an integer and returns a tree value.";
  let arguments = (ins TreeEnsembleType:$forest, Index:$treeIndex);

  let results = (outs TreeType);
}

def GetRootOp : DecisionForest_Op<"getRoot", [NoSideEffect]> {
  let summary = "Get the root of a tree from the forest.";
  let description = "Get the root of the tree passed as an argument. Takes one argument -- a tree and returns a Node value.";
  let arguments = (ins TreeType:$tree);

  let results = (outs NodeType);
}

// TODO the argument node type needs to be parameterized
def IsLeafOp : DecisionForest_Op<"isLeaf", [NoSideEffect]> {
  let summary = "Check if the node passed as argument is a leaf.";
  let description = "Check if the node passed as argument is a leaf.";
  let arguments = (ins TreeType:$tree, NodeType:$node);

  let results = (outs I1);
}

def GetLeafValueOp : DecisionForest_Op<"getLeafValue", [NoSideEffect]> {
  let summary = "Get value of passed leaf node.";
  let description = "Get value of passed leaf node.";
  let arguments = (ins TreeType:$tree, NodeType:$node);

  let results = (outs LeafNodeValueType);
}

def TraverseTreeTileOp : DecisionForest_Op<"traverseTreeTile", [NoSideEffect]> {
  let summary = "Traverse one tile of tree.";
  let description = "Traverse one tile of the tree passed as argument starting from the node passed as argument. Returns the resultant node.";
  let arguments = (ins TreeType:$tree, NodeType:$node, F64Tensor:$data);

  let results = (outs NodeType);
}

def TraverseTileOfConstTreeOp : DecisionForest_Op<"traverseTileOfConstTree", [NoSideEffect]> {
  let summary = "Traverse one tile of a constant tree.";
  let description = "Traverse one tile of a constant tree stored as an attribute starting from the node passed as argument. Returns the resultant node.";
  let arguments = (ins DecisionTreeAttr:$tree, NodeType:$node, F64Tensor:$data);

  let results = (outs NodeType);
}

def TiledNumericalNodeMemRef  : MemRefOf<[TiledNumericalNodeType]>;
def TileOfThresholdTypes : AnyTypeOf<[ThresholdValueType, VectorOf<[ThresholdValueType]>]>;
def TileOfFeatureIndices : AnyTypeOf<[FeatureIndexType, VectorOf<[FeatureIndexType]>]>;

def LoadTileThresholdsOp : DecisionForest_Op<"loadTileThresholds", [NoSideEffect]> {
  let summary = "Load the thresholds from a tree tile.";
  let description = "Load all the thresholds from a tile in a tree memref. Takes a memref and an index into the memref and returns a vector of floating point values.";
  let arguments = (ins TiledNumericalNodeMemRef:$treeMemref, Index:$nodeIndex);

  let results = (outs TileOfThresholdTypes);
}

def LoadTileFeatureIndicesOp : DecisionForest_Op<"loadTileFeatureIndices", [NoSideEffect]> {
  let summary = "Load the feature indices from a tree tile.";
  let description = "Load all the feature indices from a tile in a tree memref. Takes a memref and an index into the memref and returns a vector of floating point values.";
  let arguments = (ins TiledNumericalNodeMemRef:$treeMemref, Index:$nodeIndex);

  let results = (outs TileOfFeatureIndices);
}

def IndexToNodeOp : DecisionForest_Op<"indexToNode", [NoSideEffect]> {
  let summary = "Casts an index into a tree memref into a node type.";
  let description = "Casts an index into a tree memref into a node type. This is a hack so that we don't have to change any block arguments while lowering.";
  let arguments = (ins TiledNumericalNodeMemRef:$treeMemref, Index:$nodeIndex);

  let results = (outs NodeType);
}

def NodeToIndexOp : DecisionForest_Op<"nodeToIndex", [NoSideEffect]> {
  let summary = "Casts a node into an index into a tree memref.";
  let description = "Casts a node into an index into a tree memref. This is a hack so that we don't have to change too any block arguments while lowering.";
  let arguments = (ins TiledNumericalNodeMemRef:$treeMemref, NodeType:$node);

  let results = (outs Index);
}

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

#endif // TREE_HEAVY_OPS