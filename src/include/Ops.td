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
  let useDefaultTypePrinterParser = 1;
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

def ScheduleAttr : Attr<
  CPred<"$_self.isa<::mlir::decisionforest::ScheduleAttribute>()">,
  "Schedule attribute"> {

  let storageType = [{ ::mlir::decisionforest::ScheduleAttribute }];
  let returnType = [{ ::mlir::decisionforest::ScheduleAttribute }];
  
  let constBuilderCall = "::mlir::decisionforest::ScheduleAttribute::get()";
  let convertFromStorage = "$_self";
}

def UnrollLoopAttr : Attr<
  CPred<"$_self.isa<::mlir::decisionforest::UnrollLoopAttribute>()">,
  "Unroll loop attribute"> {

  let storageType = [{ ::mlir::decisionforest::UnrollLoopAttribute }];
  let returnType = [{ ::mlir::decisionforest::UnrollLoopAttribute }];
  
  let constBuilderCall = "::mlir::decisionforest::UnrollLoopAttribute::get()";
  let convertFromStorage = "$_self";
}

//-------   Operations   ----------------------

class DecisionForest_Op<string mnemonic, list<Trait> traits = []> : 
  Op<DecisionForestDialect, mnemonic, traits>;


//------- High level IR Ops ------------------

def InputDataType : AnyTypeOf<[F16MemRef, F32MemRef, F64MemRef]>;
def ReturnType : AnyTypeOf<[I8, I16, I32, F16, F32, F64, I8MemRef, I16MemRef, I32MemRef, F16MemRef, F32MemRef, F64MemRef]>;
def LeafNodeValueType : AnyTypeOf<[F16, F32, F64]>;
def ClassIdValueType : AnyTypeOf<[I8, I16, I32]>;
def TreeClassInfoMemref : AnyTypeOf<[I8MemRef, I16MemRef, I32MemRef]>;

def PredictForestOp : DecisionForest_Op<"predict_ensemble"> {
    let summary = "Predict forest operation";
    let description = "Represents the actual inference computation on the forest. Takes a tensor operand and produces a result."
                      "Contains the actual forest as an attribute.";

    let arguments = (ins DecisionForestAttr:$ensemble, InputDataType:$data, ReturnType:$result, ScheduleAttr:$schedule);

    // Just returns its result argument after filling in the results
    let results = (outs ReturnType);

    // let extraClassDeclaration = [{
    //   int32_t getBatchSize() { 
       
    //     if (data().getType().isa<mlir::RankedTensorType>()) {
    //       auto tensorReturnType = data().getType().cast<mlir::RankedTensorType>();
    //       return tensorReturnType.getShape()[0];
    //     }
    //     else if (data().getType().isa<mlir::MemRefType>()) {
    //       auto memrefReturnType = data().getType().cast<mlir::MemRefType>();
    //       return memrefReturnType.getShape()[0];
    //     }
    //     return 1;
    //   }
    // }];
}

def WalkDecisionTreeOp : DecisionForest_Op<"walk_decision_tree"> {
  let summary = "Walk the decision tree.";
  let description = "Operation to walk the decision tree and generate a prediction."
                    "Takes input data and tree as input and produces prediction as output.";

  let arguments = (ins TreeType:$tree, InputDataType:$data);

  let results = (outs LeafNodeValueType);
}

def PipelinedWalkDecisionTreeOp : DecisionForest_Op<"pipelined_walk_decision_tree", [SameVariadicOperandSize]> {
  let summary = "Takes in equal number of data rows and trees and walks them as a element wise pair.";
  let description = "Operation to walk multiple decision trees and data rows as element wise pairs."
                    "Pass in trees <t1,t2, ... tn> and <d1, d2, ... dn>. The op walks <(t1, d1), (t2, d2) .... (tn, dn)>";

  let arguments = (ins UnrollLoopAttr:$UnrollLoopAttr, Variadic<TreeType>:$trees, Variadic<InputDataType>:$dataRows);

  let results = (outs Variadic<LeafNodeValueType>);
}

def WalkDecisionTreePeeledOp : DecisionForest_Op<"walkDecisionTreePeeled"> {
  let summary = "Walk the decision tree.";
  let description = "Operation to walk the decision tree and generate a prediction. Generates"
                    "code with the first few iterations peeled."
                    "Takes input data and tree as input and produces prediction as output.";

  let arguments = (ins TreeType:$tree, InputDataType:$data, I32Attr:$iterationsToPeel);

  let results = (outs LeafNodeValueType);
} 

// ----- Mid-level IR Ops ------

def ThresholdValueType : AnyTypeOf<[F16, F32, F64]>;
def FeatureIndexType : AnyTypeOf<[I8, I16, I32, I64]>;
def TileShapeType : AnyTypeOf<[I8, I16, I32, I64]>;
def ChildIndexType : AnyTypeOf<[I8, I16, I32, I64]>;

def TreeConstantOp : DecisionForest_Op<"tree", [Pure]> {
  let summary = "Tree constant operation";
  let description = "Represents a constant whose value is a single tree in the forest. Takes no arguments and gives a tree value out.";

  let arguments = (ins DecisionTreeAttr:$tree);

  let results = (outs TreeType);
}

def EnsembleConstantOp : DecisionForest_Op<"ensemble", [Pure]> {
  let summary = "Ensemble constant operation";
  let description = "Represents a constant whose value is the full forest. Takes no arguments and gives an ensemble value out.";
  let arguments = (ins DecisionForestAttr:$forest);

  let results = (outs TreeEnsembleType);
}

def GetTreeFromEnsembleOp : DecisionForest_Op<"getTree", [Pure]> {
  let summary = "Get single tree from the forest.";
  let description = "Index the forest and get a single tree. Takes two arguments -- an ensemble and an integer and returns a tree value.";
  let arguments = (ins TreeEnsembleType:$forest, Index:$treeIndex);

  let results = (outs TreeType);
}

def CacheTreesFromEnsembleOp : DecisionForest_Op<"cacheTrees", [Pure]> {
  let summary = "Cache trees from the forest.";
  let description = "Cache a subset of trees from the forest. Takes three arguments -- an ensemble, and two integers to represent the range of trees to cache. Returns an ensemble value.";
  let arguments = (ins TreeEnsembleType:$forest, Index:$startTreeIndex, Index:$endTreeIndex);

  let results = (outs TreeEnsembleType);
}

def GetTreeClassIdOp : DecisionForest_Op<"getTreeClassId", [Pure]> {
  let summary = "Get tree class from ensemble.";
  let description = "Index the forest and get the class Id of the tree at that index. Takes two arguments -- an ensemble and an integer and returns a tree value.";
  let arguments = (ins TreeEnsembleType:$forest, Index:$treeIndex);

  let results = (outs ClassIdValueType);
}

def GetRootOp : DecisionForest_Op<"getRoot", [Pure]> {
  let summary = "Get the root of a tree from the forest.";
  let description = "Get the root of the tree passed as an argument. Takes one argument -- a tree and returns a Node value.";
  let arguments = (ins TreeType:$tree);

  let results = (outs NodeType);
}

// TODO the argument node type needs to be parameterized
def IsLeafOp : DecisionForest_Op<"isLeaf", [Pure]> {
  let summary = "Check if the node passed as argument is a leaf.";
  let description = "Check if the node passed as argument is a leaf.";
  let arguments = (ins TreeType:$tree, NodeType:$node);

  let results = (outs I1);
}

def IsLeafTileOp : DecisionForest_Op<"isLeafTile", [Pure]> {
  let summary = "Check if the node passed as argument is a leaf tile.";
  let description = "Check if the node passed as argument is a leaf tile.";
  let arguments = (ins TreeType:$tree, NodeType:$node);

  let results = (outs I1);
}

def GetLeafValueOp : DecisionForest_Op<"getLeafValue", [Pure]> {
  let summary = "Get value of passed leaf node.";
  let description = "Get value of passed leaf node.";
  let arguments = (ins TreeType:$tree, NodeType:$node);

  let results = (outs LeafNodeValueType);
}

def GetLeafTileValueOp : DecisionForest_Op<"getLeafTileValue", [Pure]> {
  let summary = "Get value of passed leaf node.";
  let description = "Get value of passed leaf node.";
  let arguments = (ins TreeType:$tree, NodeType:$node);

  let results = (outs LeafNodeValueType);
}

def TraverseTreeTileOp : DecisionForest_Op<"traverseTreeTile", [Pure]> {
  let summary = "Traverse one tile of tree.";
  let description = "Traverse one tile of the tree passed as argument starting from the node passed as argument. Returns the resultant node.";
  let arguments = (ins TreeType:$tree, NodeType:$node, InputDataType:$data);

  let results = (outs NodeType);
}

def InterleavedTraverseTreeTileOp : DecisionForest_Op<"interleavedTraverseTileOp", [Pure, SameVariadicOperandSize]> {
  let summary = "Traverse each <node, data> pair for the list of pairs that are passed in.";
  let description = "Traverse each element wise <node, data> from individual input arguments that are passed in.";
  let arguments = (ins Variadic<TreeType>:$trees, Variadic<NodeType>:$nodes, Variadic<InputDataType>:$data);

  let results = (outs Variadic<NodeType>);
}

def TraverseTileOfConstTreeOp : DecisionForest_Op<"traverseTileOfConstTree", [Pure]> {
  let summary = "Traverse one tile of a constant tree.";
  let description = "Traverse one tile of a constant tree stored as an attribute starting from the node passed as argument. Returns the resultant node.";
  let arguments = (ins DecisionTreeAttr:$tree, NodeType:$node, InputDataType:$data);

  let results = (outs NodeType);
}

def TiledNumericalNodeMemRef  : MemRefOf<[TiledNumericalNodeType]>;
def TileOfThresholdTypes : AnyTypeOf<[ThresholdValueType, VectorOf<[ThresholdValueType]>]>;
def TileOfFeatureIndices : AnyTypeOf<[FeatureIndexType, VectorOf<[FeatureIndexType]>]>;

def ThresholdMemrefType : AnyTypeOf<[MemRefOf<[TiledNumericalNodeType]>, MemRefOf<[ThresholdValueType]>]>;
def FeatureIndexMemrefType : AnyTypeOf<[MemRefOf<[TiledNumericalNodeType]>, MemRefOf<[FeatureIndexType]>]>;
def TileShapeMemrefType : AnyTypeOf<[MemRefOf<[TiledNumericalNodeType]>, MemRefOf<[FeatureIndexType]>]>;

// ----- Low-level IR Ops ------

def LoadTileThresholdsOp : DecisionForest_Op<"loadTileThresholds", [Pure]> {
  let summary = "Load the thresholds from a tree tile.";
  let description = "Load all the thresholds from a tile in a tree memref. Takes a memref and an index into the memref and returns a vector of floating point values.";
  let arguments = (ins ThresholdMemrefType:$treeMemref, Index:$nodeIndex, Index:$treeIndex);

  let results = (outs TileOfThresholdTypes);
}

def LoadTileFeatureIndicesOp : DecisionForest_Op<"loadTileFeatureIndices", [Pure]> {
  let summary = "Load the feature indices from a tree tile.";
  let description = "Load all the feature indices from a tile in a tree memref. Takes a memref and an index into the memref and returns a vector of floating point values.";
  let arguments = (ins FeatureIndexMemrefType:$treeMemref, Index:$nodeIndex, Index:$treeIndex);

  let results = (outs TileOfFeatureIndices);
}

def LoadTileShapeOp : DecisionForest_Op<"loadTileShape", [Pure]> {
  let summary = "Load the shape of a tree tile.";
  let description = "Load the shape ID from a tile in a tree memref. Takes a memref and an index into the memref and returns an integer.";
  let arguments = (ins TileShapeMemrefType:$treeMemref, Index:$nodeIndex, Index:$treeIndex);

  let results = (outs TileShapeType);
}

def IndexToNodeOp : DecisionForest_Op<"indexToNode", [Pure]> {
  let summary = "Casts an index into a tree memref into a node type.";
  let description = "Casts an index into a tree memref into a node type. This is a hack so that we don't have to change any block arguments while lowering.";
  let arguments = (ins ThresholdMemrefType:$treeMemref, Index:$nodeIndex);

  let results = (outs NodeType);
}

def NodeToIndexOp : DecisionForest_Op<"nodeToIndex", [Pure]> {
  let summary = "Casts a node into an index into a tree memref.";
  let description = "Casts a node into an index into a tree memref. This is a hack so that we don't have to change too any block arguments while lowering.";
  let arguments = (ins ThresholdMemrefType:$treeMemref, NodeType:$node);

  let results = (outs Index);
}

// ----- Representation Specific Low-level IR Ops ------

// Array-based representation
def InitTileOp : DecisionForest_Op<"initTile"> {
  let summary = "Initialize a tree tile.";
  let description = "Initialize a single tile in a tree memref. Takes a memref and an index along with the values to write into the tile.";
  let arguments = (ins TiledNumericalNodeMemRef:$treeMemref, 
                       Index:$tileIndex,
                       TileOfThresholdTypes:$thresholds,
                       TileOfFeatureIndices:$featureIndices,
                       TileShapeType:$tileShapeID);
}

// Sparse representation
def LoadChildIndexOp : DecisionForest_Op<"loadChildIndex", [Pure]> {
  let summary = "Load the child index of a tree tile.";
  let description = "Load the child index from a tile in a tree memref. Takes a memref and an index into the memref and returns an integer.";
  let arguments = (ins TiledNumericalNodeMemRef:$treeMemref, Index:$nodeIndex);

  let results = (outs ChildIndexType);
}

def InitSparseTileOp : DecisionForest_Op<"initSparseTile"> {
  let summary = "Initialize a sparse tree tile.";
  let description = "Initialize a single tile in a tree memref. Takes a memref and an index along with the values to write into the tile.";
  let arguments = (ins TiledNumericalNodeMemRef:$treeMemref, 
                       Index:$tileIndex,
                       TileOfThresholdTypes:$thresholds,
                       TileOfFeatureIndices:$featureIndices,
                       TileShapeType:$tileShapeID,
                       ChildIndexType:$childIndex);
}

def GetModelMemrefSizeOp : DecisionForest_Op<"getModelMemrefSizeOp", [Pure]> {
  let summary = "Gets the size (in bytes) of the argument model memref.";
  let description = "Gets the size (in bytes) of the argument model memref. Takes into account the actual size of the tile on the target to which we are compiling.";
  let arguments = (ins TiledNumericalNodeMemRef:$modelMemref, Index:$length);

  let results = (outs I32);
}

// ----- Debugging IR Ops ------
def PrintTreePredictionOp : DecisionForest_Op<"printTreePrediction", []> {
  let arguments = (ins F64:$treePrediction, Index:$treeIndex);
}

def PrintTreeNodeOp : DecisionForest_Op<"printTreeNode", []> {
  let arguments = (ins Index:$nodeIndex);
}

def PrintTreeToDOTFileOp : DecisionForest_Op<"printTreeToDOTFile", []> {
  let arguments = (ins TiledNumericalNodeMemRef:$treeMemref, Index:$treeIndex);
}

def PrintInputRowOp : DecisionForest_Op<"printInputRow", []> {
  let arguments = (ins InputDataType:$data, Index:$rowIndex);
}

def PrintComparisonOp : DecisionForest_Op<"printComparison", []> {
  let arguments = (ins TileOfThresholdTypes:$data, TileOfThresholdTypes:$threshold, TileOfFeatureIndices:$nodeIndex);
}

def PrintIsLeafOp : DecisionForest_Op<"printIsLeaf", []> {
  let arguments = (ins Index:$nodeIndex, I32:$featureIndex, I32:$outcome);
}

def PrintVectorOp : DecisionForest_Op<"printVector", []> {
  let arguments = (ins I32:$kind, I32:$bitWidth, I32:$vectorSize, Variadic<AnyType>:$elements);
}

#endif // TREE_HEAVY_OPS