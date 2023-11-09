// -- Dialect specification for tree-heavy --

// Function generated in the high-level dialect 
// function Predict(x)
// { 
//   %1 = predict_const_ensemble(x) // Attr = ensemble 
//   return %1
// }

#ifndef TREE_HEAVY_OPS
#define TREE_HEAVY_OPS

include "mlir/Dialect/Arith/IR/ArithBase.td"
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

def ReorgMemrefElementType : 
    DialectType<DecisionForestDialect, CPred<"$_self.isa<ReorgMemrefElementType>()">,
    "ReorgMemrefElementType">;

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

def ReductionTypeAttr: Attr<
  CPred<"$_self.isa<::mlir::decisionforest::ReductionTypeAttribute>()">,
  "Reduction type attribute"> {

  let storageType = [{ ::mlir::decisionforest::ReductionTypeAttribute }];
  let returnType = [{ ::mlir::decisionforest::ReductionTypeAttribute }];
  
  let constBuilderCall = "::mlir::decisionforest::ReductionTypeAttribute::get()";
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

    let arguments = (ins DecisionForestAttr:$ensemble, 
                         Arith_CmpFPredicateAttr:$predicate,
                         InputDataType:$data,
                         ReturnType:$result,
                         ScheduleAttr:$schedule);

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

  let arguments = (ins Arith_CmpFPredicateAttr:$predicate,
                       I64Attr:$unrollSteps, 
                       TreeType:$tree,
                       InputDataType:$data);

  let results = (outs LeafNodeValueType);
}

def PipelinedWalkDecisionTreeOp : DecisionForest_Op<"pipelined_walk_decision_tree", [SameVariadicOperandSize]> {
  let summary = "Takes in equal number of data rows and trees and walks them as a element wise pair.";
  let description = "Operation to walk multiple decision trees and data rows as element wise pairs."
                    "Pass in trees <t1,t2, ... tn> and <d1, d2, ... dn>. The op walks <(t1, d1), (t2, d2) .... (tn, dn)>";

  let arguments = (ins UnrollLoopAttr:$UnrollLoopAttr,
                       Arith_CmpFPredicateAttr:$predicate,
                       Variadic<TreeType>:$trees,
                       Variadic<InputDataType>:$dataRows);

  let results = (outs Variadic<LeafNodeValueType>);
}

def WalkDecisionTreePeeledOp : DecisionForest_Op<"walkDecisionTreePeeled"> {
  let summary = "Walk the decision tree.";
  let description = "Operation to walk the decision tree and generate a prediction. Generates"
                    "code with the first few iterations peeled."
                    "Takes input data and tree as input and produces prediction as output.";

  let arguments = (ins Arith_CmpFPredicateAttr:$predicate,
                       TreeType:$tree,
                       InputDataType:$data,
                       I32Attr:$iterationsToPeel);

  let results = (outs LeafNodeValueType);
} 

// ----- Reduction Ops ------
def ReductionValueType : AnyTypeOf<[I8, F16, F32, F64]>;
def TargetMemrefType : AnyTypeOf<[I8MemRef, F16MemRef, F32MemRef, F64MemRef]>;

def ReduceOp : DecisionForest_Op<"reduce"> {
  let summary = "Accumulate a value into a memref.";
  let description = "Accumulate a floating point value into an element of a memref"
                    "specified by the indices.";

  let arguments = (ins ReductionTypeAttr:$reductionType,
                       TargetMemrefType:$targetMemref,
                       Variadic<Index>:$indices,
                       ReductionValueType:$value,
                       F64Attr:$initialValue);
}

def AtomicReduceOp : DecisionForest_Op<"atomic_reduce"> {
  let summary = "Atomically accumulate a value into a memref.";
  let description = "Atomically accumulate a floating point value into an element of a memref"
                    "specified by the indices.";

  let arguments = (ins ReductionTypeAttr:$reductionType,
                       TargetMemrefType:$targetMemref,
                       Variadic<Index>:$indices,
                       ReductionValueType:$value,
                       F64Attr:$initialValue);
}

def ReduceDimensionInplaceOp : DecisionForest_Op<"reduce_dimension_inplace", [AttrSizedOperandSegments]> {
  let summary = "Reduce the specified dimension of a memref inplace.";
  let description = "Reduce the specified dimension of a memref inplace."
                    "The reduced value is stored at index 0 of the specified dimension."
                    "The (pre, post)ReductionDimension* indices are used to determine which subset of the memref to reduce.";

  let arguments = (ins ReductionTypeAttr:$reductionType,
                       TargetMemrefType:$targetMemref,
                       Variadic<Index>:$preReductionDimensionStart,
                       Variadic<Index>:$preReductionDimensionEnd,
                       Index:$reductionDimension,
                       Variadic<Index>:$postReductionDimensionStart,
                       Variadic<Index>:$postReductionDimensionEnd);
}

def ReduceDimensionOp : DecisionForest_Op<"reduce_dimension", [AttrSizedOperandSegments]> {
  let summary = "Reduce the specified dimension of a memref and store result in target.";
  let description = "Reduce the specified dimension of a memref and store result in the target memref."
                    "The op copies reduced values to the target memref. Reduced dimesions is the list" 
                    "The (pre, post)ReductionDimension* indices are used to determine which subset of the memref to reduce."
                    "The mappedDimensions are used to the indices of the target memref."
                    "For eached mapped dimension, the corresponding source memref is mapped 1:1.";

  let arguments = (ins ReductionTypeAttr:$reductionType,
                       TargetMemrefType:$targetMemref,
                       TargetMemrefType:$sourceMemref,
                       Variadic<Index>:$targetMemrefOffsets,
                       Variadic<Index>:$mappedDimensions,
                       Variadic<Index>:$preReductionDimensionStart,
                       Variadic<Index>:$preReductionDimensionEnd,
                       Index:$reductionDimension,
                       Variadic<Index>:$postReductionDimensionStart,
                       Variadic<Index>:$postReductionDimensionEnd,
                       F64Attr:$initialValue);
}

def CooperativeReduceInplaceOp : DecisionForest_Op<"cooperative_reduce_dimension_inplace", [AttrSizedOperandSegments]> {
  let summary = "Reduce the specified dimension of a memref inplace.";
  let description = "Reduce the specified dimension of a memref inplace."
                    "The reduced value is stored at index 0 of the specified dimension."
                    "Use the threads specified in the arguments to cooperatively reduce.";

  let arguments = (ins TargetMemrefType:$targetMemref,
                       Variadic<Index>:$preReductionDimensionStart,
                       Variadic<Index>:$preReductionDimensionEnd,
                       Index:$reductionDimension,
                       Variadic<Index>:$postReductionDimensionStart,
                       Variadic<Index>:$postReductionDimensionEnd,
                       Index:$blockXStart, Index:$blockYStart, Index:$blockZStart,
                       Index:$blockXEnd, Index:$blockYEnd, Index:$blockZEnd);
}

def CooperativeReduceDimensionOp : DecisionForest_Op<"cooperative_reduce_dimension", [AttrSizedOperandSegments]> {
  let summary = "Reduce the specified dimension of a memref and store result in target.";
  let description = "Reduce the specified dimension of a memref and store result in the target memref."
                    "The op copies reduced values to the target memre. Reduced dimesions is the list" 
                    "of dimensions that have been previously reduced (i.e. the set of dimensions where"
                    "we should index with a 0).";

  let arguments = (ins ReductionTypeAttr:$reductionType,
                       TargetMemrefType:$targetMemref,
                       TargetMemrefType:$sourceMemref,
                       Variadic<Index>:$targetMemrefOffsets,
                       Index:$reductionDimension,
                       Variadic<Index>:$reducedDimensions,
                       Variadic<Index>:$rangeStart,
                       Variadic<Index>:$rangeEnd,
                       Index:$blockXStart, Index:$blockYStart, Index:$blockZStart,
                       Index:$blockXEnd, Index:$blockYEnd, Index:$blockZEnd,
                       F64Attr:$initialValue);
}

def CooperativeReduceArgMaxOp : DecisionForest_Op<"coopertivate_argmax", [AttrSizedOperandSegments]> {
  let summary = "Reduce the specified dimension of a memref and store result in target.";
  let description = "Reduce the specified dimension of a memref and store result in the target memref."
                    "The op copies reduced values to the target memref. Reduced dimesions is the list" 
                    "The (pre, post)ReductionDimension* indices are used to determine which subset of the memref to reduce."
                    "The mappedDimensions are used to the indices of the target memref."
                    "For eached mapped dimension, the corresponding source memref is mapped 1:1.";

  let arguments = (ins ReductionTypeAttr:$reductionType,
                       TargetMemrefType:$targetMemref,
                       TargetMemrefType:$sourceMemref,
                       Variadic<Index>:$targetMemrefOffsets,
                       Variadic<Index>:$mappedDimensions,
                       Variadic<Index>:$preReductionDimensionStart,
                       Variadic<Index>:$preReductionDimensionEnd,
                       Index:$reductionDimension,
                       Variadic<Index>:$postReductionDimensionStart,
                       Variadic<Index>:$postReductionDimensionEnd,
                       Index:$blockXStart, Index:$blockYStart, Index:$blockZStart,
                       Index:$blockXEnd, Index:$blockYEnd, Index:$blockZEnd);
}

def AtomicReduceDimensionOp : DecisionForest_Op<"atomic_reduce_dimension", [AttrSizedOperandSegments]> {
  let summary = "Reduce the specified dimension of a memref and store result in target.";
  let description = "Reduce the specified dimension of a memref and store result in the target memref."
                    "The op copies reduced values to the target memref. Reduced dimesions is the list" 
                    "The (pre, post)ReductionDimension* indices are used to determine which subset of the memref to reduce."
                    "The mappedDimensions are used to the indices of the target memref."
                    "For eached mapped dimension, the corresponding source memref is mapped 1:1.";

  let arguments = (ins ReductionTypeAttr:$reductionType,
                       TargetMemrefType:$targetMemref,
                       TargetMemrefType:$sourceMemref,
                       Variadic<Index>:$mappedDimensions,
                       Variadic<Index>:$preReductionDimensionStart,
                       Variadic<Index>:$preReductionDimensionEnd,
                       Index:$reductionDimension,
                       Index:$reductionDimensionIndex,
                       Variadic<Index>:$postReductionDimensionStart,
                       Variadic<Index>:$postReductionDimensionEnd,
                       F64Attr:$initialValue);
}

def InitializeMemrefOp : DecisionForest_Op<"initialize_memref"> {
  let summary = "Initialize all elements of a memref with the specified initial value";
  let description = "Initialize all elements of a memref with the specified initial value";

  let arguments = (ins TargetMemrefType:$targetMemref,
                       F64Attr:$initialValue);

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

def EnsembleConstantOp : DecisionForest_Op<"ensemble"> {
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
  let arguments = (ins TreeEnsembleType:$forest, Index:$startTreeIndex, Index:$endTreeIndex, I64Attr:$cacheID);

  let results = (outs TreeEnsembleType);
}

def CacheInputRowsOp : DecisionForest_Op<"cacheRows", [Pure]> {
  let summary = "Cache rows from the input.";
  let description = "Cache a subset of rows from the input. Takes three arguments -- input memref, and two integers to represent the range of rows to cache. Returns a memref value.";
  let arguments = (ins InputDataType:$data, Index:$startIndex, Index:$endIndex);

  let results = (outs InputDataType);
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
  let arguments = (ins Arith_CmpFPredicateAttr:$predicate, 
                       TreeType:$tree,
                       NodeType:$node,
                       InputDataType:$data);

  let results = (outs NodeType);
}

def InterleavedTraverseTreeTileOp : DecisionForest_Op<"interleavedTraverseTileOp", [Pure, SameVariadicOperandSize]> {
  let summary = "Traverse each <node, data> pair for the list of pairs that are passed in.";
  let description = "Traverse each element wise <node, data> from individual input arguments that are passed in.";
  let arguments = (ins Arith_CmpFPredicateAttr:$predicate,
                       Variadic<TreeType>:$trees,
                       Variadic<NodeType>:$nodes,
                       Variadic<InputDataType>:$data);

  let results = (outs Variadic<NodeType>);
}

def CooperativeTraverseTreeTileOp : DecisionForest_Op<"cooperativeTraverseTileOp", [Pure, SameVariadicOperandSize]> {
  let summary = "Multiple threads cooperate to traverse a set of tiles";
  let description = "Traverse different (tile, row) pairs cooperatively across GPU threads";
  let arguments = (ins Arith_CmpFPredicateAttr:$predicate,
                       TreeType:$tree,
                       NodeType:$node,
                       Variadic<InputDataType>:$data);

  let results = (outs NodeType);
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

def ThresholdMemrefType : AnyTypeOf<[MemRefOf<[TiledNumericalNodeType]>, MemRefOf<[ThresholdValueType]>, MemRefOf<[ReorgMemrefElementType]>]>;
def FeatureIndexMemrefType : AnyTypeOf<[MemRefOf<[TiledNumericalNodeType]>, MemRefOf<[FeatureIndexType]>, MemRefOf<[ReorgMemrefElementType]>]>;
def TileShapeMemrefType : AnyTypeOf<[MemRefOf<[TiledNumericalNodeType]>, MemRefOf<[FeatureIndexType]>]>;

// ----- Low-level IR Ops ------

def LoadTileThresholdsOp : DecisionForest_Op<"loadTileThresholds", [Pure]> {
  let summary = "Load the thresholds from a tree tile.";
  let description = "Load all the thresholds from a tile in a tree memref. Takes a memref and an index into the memref and returns a vector of floating point values.";
  let arguments = (ins ThresholdMemrefType:$treeMemref, Index:$nodeIndex, Index:$treeIndex, Optional<Index>:$elementIndex);

  let results = (outs TileOfThresholdTypes);
  let builders = [OpBuilder<(ins "Type":$resultType, "Value":$treeMemref, "Value":$nodeIndex, "Value":$treeIndex), [{
      build($_builder, $_state, resultType, treeMemref, nodeIndex, treeIndex, Value());
    }]>];
}

def LoadTileFeatureIndicesOp : DecisionForest_Op<"loadTileFeatureIndices", [Pure]> {
  let summary = "Load the feature indices from a tree tile.";
  let description = "Load all the feature indices from a tile in a tree memref. Takes a memref and an index into the memref and returns a vector of floating point values.";
  let arguments = (ins FeatureIndexMemrefType:$treeMemref, Index:$nodeIndex, Index:$treeIndex, Optional<Index>:$elementIndex);

  let results = (outs TileOfFeatureIndices);
  let builders = [OpBuilder<(ins "Type":$resultType, "Value":$treeMemref, "Value":$nodeIndex, "Value":$treeIndex), [{
      build($_builder, $_state, resultType, treeMemref, nodeIndex, treeIndex, Value());
    }]>];
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

def CacheOpBeginOp : DecisionForest_Op<"cacheOpBegin"> {
  let summary = "Sync at start of a cache op.";
  let description = "Sync at start of a cache op.";
}

def CacheOpEndOp : DecisionForest_Op<"cacheOpEnd"> {
  let summary = "Sync at end of a cache op.";
  let description = "Sync at end of a cache op.";
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

def PrintfOp : DecisionForest_Op<"printf", [MemoryEffects<[MemWrite]>]> {
  let summary = "printf for debugging";
  let description = [{
    `printf` takes a literal format string `format` and an arbitrary number of
    scalar arguments that should be printed.

    The format string is a C-style printf string, subject to any restrictions
    imposed by one's target platform.
  }];
  let arguments = (ins StrAttr:$format,
                   Variadic<AnyTypeOf<[AnyInteger, Index, AnyFloat]>>:$args);
  let assemblyFormat = [{
    $format attr-dict ($args^ `:` type($args))?
  }];
}


#endif // TREE_HEAVY_OPS