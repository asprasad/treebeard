# Reductions : Representation, Optimization and Lowering 

Currently, existing reduction support in MLIR is insufficient to easily code generate and optimize the reductions Treebeard needs
to perform while performing inference (sum up individual tree predictions to compute the prediction of the model). This document
proposes a possible solution to this problem.

We aim to design a mechanism to specify accumulating values into a tensor where the accumulation is performed inside an arbitrary 
loop nest where several surrounding loops maybe parallel and the ultimate target machine maybe a CPU or a GPU. Because this problem 
is of general interest (for example, we need to perform such reductions even in the context of generating data-processing code), we 
will design this as an MLIR dialect. The current focus is however limited to what is needed in Treebeard.

The main op we will intro is the `reduce` op which models accumulating values into an element of a tensor (represented by a MLIR tensor).
For example, to sum up all the elements of 1D memref `arr`, one would write. 
```C++
builtin.module @MyModule  {
  builtin.func @ReduceVector(%arr: memref<num_elemsxf64>, %result: memref<1xf64>) -> void {
    par.for i = range(0 : num_elems) {
      reduce(%result, 0, arr[i])
    }
  }
}
```
The main differences between our `reduce` and the existing reductions in MLIR are the following:
1. Existing reductions only support scalar reductions and not accumulating into memref elements.
2. Existing reductions have very strict rules about where they can be written. For example, `scf.reduce` needs to be an immediate child `scf.parallel`. However, our `reduce` op can be generated anywhere in the
loop nest. 

Having modeled the reductions with an abstract op, the aim now is to lower this to a correct and optimized implementation on both CPU and GPU. In order to do this, we make the following observations. 
1. The `reduce` op has a loop carried dependency on itself. A simple lowering to a load-add-store is 
incorrect if the self dependence is carried by any of the parallel loops. We call any such a surrounding
parallel loop a **conflicting loop** (TODO Change the name!) for the reduction.
2. The result memref needs to be **privatized** wrt each conflicting loop. We cannot do better than this in terms of memory usage (TODO Need a proof).
3. Each privatized dimension can be reduced at the end of the conflicting loop it was inserted for.

A **conflicting loop** for a `reduce` op is a surrounding parallel loop that has a non-zero dependence 
distance for the self dependence of the `reduce` op. In the context of Treebeard, this is exactly the set 
of surrounding parallel loops that are iterating over trees. The results can be privatized for each 
conflicting loop iteratively and reductions along each privatized dimension can be inserted at the 
exit of the loop the dimension was inserted due to. Consider the following code. 

```C++
builtin.module @MyModule  {
  builtin.func @ReduceVector(%arr: memref<num_elemsxf64>, %result: memref<1xf64>) -> void {
    par.for i0 = range(0 : num_elems/2 : num_elems) {
      for i1 = range(0 : num_elems/2) 
        reduce(%result, 0, arr[i0 + i1])
    }
  }
}
```

Here, the `i0` loop is a conflicting loop. We would therefore privatize the result memref for each 
iteration of the `i0` loop. 
```C++
builtin.module @MyModule  {
  builtin.func @ReduceVector(%arr: memref<num_elemsxf64>, %result: memref<1xf64>) -> void {
    results_1 = memref<2x1xf64>
    par.for i0 = range(0 : num_elems/2 : num_elems) {
      for i1 = range(0 : num_elems/2) 
        reduce(%result_1, i0, 0, arr[i0 + i1])
    }
    results = reduce_dimension(results_1, 0)
  }
}
```
The op `reduce_dimension` reduces values across the specified dimension of an n dimensional memref.
To minimize the amount of memory allocated, we also add an inplace version of the dimension reduce 
op, `reduce_dimension_inplace`. Only the final dimension reduction uses the `reduce_dimension` op.

Consider the following code with nested parallel loops. (A situation where trees are split across both
threads and thread blocks could result in such generated code in Treebeard.)
```C++
builtin.module @MyModule  {
  builtin.func @ReduceVector(%arr: memref<num_elemsxf64>, %result: memref<1xf64>) -> void {
    par.for i0 = range(0 : num_elems/2 : num_elems) {
      par.for i1 = range(0 : num_elems/4 : num_elems/2) {
        for i2 = range(0 : num_elems/4) 
          reduce(%result, 0, arr[i0 + i1 + i2])
      }
    }
  }
}
```

Here, the `i0` and `i1` loops are conflicting loops wrt the `reduce` op. We **legalize** the reduction 
by privatizing the result array wrt the `i0` and `i1` loops. 
```C++
builtin.module @MyModule  {
  builtin.func @ReduceVector(%arr: memref<num_elemsxf64>, %result: memref<1xf64>) -> void {
    results_1 = memref<2x2x1xf64>
    par.for i0 = range(0 : num_elems/2 : num_elems) {
      par.for i1 = range(0 : num_elems/4 : num_elems/2) {
        for i2 = range(0 : num_elems/4) 
          reduce(%result_1, i1, 0, arr[i0 + i1 + i2])
      }
      reduce_dimension_inplace(%result_1, 1, i0) // reduce elements result[i0, [0, 2), 0]
    }
    %result = reduce_dimension(%result_1, 0) // reduce elements result[[0, 2), 0, 0]
  }
}
```
## Use in Treebeard
We now present an example specific to Treebeard. The schedule with which code is generated 
is as below. `N_t` is the number of trees and `batch_size` is the batch size.
```C++
IndexVar i0, i1, t0, t1;
auto& batch = schedule.GetBatchIndex();
auto& tree = schedule.GetTreeIndex();
schedule.Tile(batch, i0, i1, batch_size/2);
schedule.Tile(tree, t0, t1, N_t/2);
schedule.Reorder({t0, i0, t1, i1});
schedule.Parallel(t0);
schedule.Parallel(i0);
```
The loop-nest generated by Treebeard for the above schedule is as follows. 
```C++
builtin.module @MyModule  {
  builtin.func @Prediction_Function(%arg0: memref<batch_sizexnum_featuresxf64>) -> memref<batch_sizexf64> {
    %result = memref.alloc <batch_sizexf64>
    %0 = #decisionforest<ReductionType = 0, #Trees = N_t, resultType = memref<batch_sizexf64>> //The forest we're performing inference on
    par.for t0 = range(0 : N_t/2 : N_t) {
      par.for i0 = range(0 : batch_size/2 : batch_size) {
        for t1 = range(0 : N_t/2) {
          for i1 = range(0 : batch_size/2) {
            %2 = GetTree(%0, t0 + t1) // Get the j th tree from the ensemble
            // 3.1 and 3.2 are used as short hand to simulate SSA, %3 refers to either assignment
            %3.1 = Root(%2) // Get the root of the tree
            while (!IsLeaf(%3)) {
                %3.2 = TraverseTreeTile(%2, %3, x[i]) // All trees are assumed to have the same tiling since no
                                                      // optimization passes have changed the tiling of trees.
            }
            %1 = GetValue(%3)
            reduce(%result, i0+i1, %1)
          }
        }
      }
    }
  }
}
```
Analysis reveals that the `t0` loop is the only conflicting loop for the reduction op. We therefore 
only need to privatize the result wrt the `t0` loop.
```C++
builtin.module @MyModule  {
  builtin.func @Prediction_Function(%arg0: memref<batch_sizexnum_featuresxf64>) -> memref<batch_sizexf64> {
    %result = memref.alloc <batch_sizexf64>
    %result_1 = memref.alloc <2xbatch_sizexf64>
    %0 = #decisionforest<ReductionType = 0, #Trees = N_t, resultType = memref<batch_sizexf64>> //The forest we're performing inference on
    par.for t0 = range(0 : N_t/2 : N_t) {
      par.for i0 = range(0 : batch_size/2 : batch_size) {
        for t1 = range(0 : N_t/2) {
          for i1 = range(0 : batch_size/2) {
            %2 = GetTree(%0, t0 + t1) // Get the j th tree from the ensemble
            // 3.1 and 3.2 are used as short hand to simulate SSA, %3 refers to either assignment
            %3.1 = Root(%2) // Get the root of the tree
            while (!IsLeaf(%3)) {
                %3.2 = TraverseTreeTile(%2, %3, x[i]) // All trees are assumed to have the same tiling since no
                                                      // optimization passes have changed the tiling of trees.
            }
            %1 = GetValue(%3)
            reduce(%result_1, t0, i0+i1, %1)
          }
        }
      }
    }
    %result = reduce_dimension(%result_1, 0)
  }
}
```
Now the `reduce` op can be lowered into a simple load-add-store. 

## Multi-class Models
```C++
func.func @Prediction_Function(%arg0: memref<200x54xf32>, %arg1: memref<200xi8>) -> memref<200xi8> {
  %0 = "decisionforest.ensemble"() { forest = #decisionforest<Forest = ( ReductionType = 0, #Trees = 800, InitialValue=0.5 ) }
  %c200 = arith.constant 200 : index // number of rows
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index // number of classes
  %cst = arith.constant 5.000000e-01 : f32
  %result = memref.alloca() : memref<200x8xf32> // -> classes output <n_rows x n_classes>, this fill be further reduced to <n_rows x 1>
  
  %c800 = arith.constant 800 : index // num trees
  %c0_3 = arith.constant 0 : index
  %c1_4 = arith.constant 1 : index
  %cst_5 = arith.constant 0.000000e+00 : f32
  scf.parallel (%arg2) = (%c0_3) to (%c800) step (%c1_4) {
    %subview = memref.subview %arg0[%arg2, 0] [1, 54] [1, 1] : memref<200x54xf32> to memref<1x54xf32, strided<[54, 1], offset: ?>>
    %c800 = arith.constant 800 : index
    %c0_3 = arith.constant 0 : index
    %c1_4 = arith.constant 1 : index
    %cst_5 = arith.constant 0.000000e+00 : f32
    %1 = scf.for %arg3 = %c0 to %c200 step %c1 iter_args(%arg4 = %cst_5) -> (f32) {
      %2 = "decisionforest.getTree"(%0, %arg2) : (!decisionforest<TreeEnsembleType(#Trees:800, rowType:memref<54xf32>, resultType:i8, reductionType:0)>, index) -> !decisionforest<TreeType(returnType:i8, tileSize:8, tileShapeType:i16, childIndexType:i1))>
      %3 = "decisionforest.walk_decision_tree"(%2, %subview) {predicate = 11 : i64} : (!decisionforest<TreeType(returnType:i8, tileSize:8, tileShapeType:i16, childIndexType:i1))>, memref<1x54xf32, strided<[54, 1], offset: ?>>) -> f32
      %4 = "decisionforest.getTreeClassId"(%0, %arg2) : (!decisionforest<TreeEnsembleType(#Trees:800, rowType:memref<54xf32>, resultType:i8, reductionType:0)>, index) -> i8
      %5 = arith.index_cast %4 : i8 to index
      reduce(%result, (%arg3, %5), %3) // reduce tree-walk result into the corresponding class.
      scf.yield %arg4 : f32
    }
    scf.yield
  }
  reduce_dimension(%result, 0, %arg1) : <reduction_type = max_index> // this reduces into %arg1. If %arg1 is not passed, new allocation is created.
  return %arg1 : memref<200xi8>
}
```

```C++
func.func @Prediction_Function(%arg0: memref<200x54xf32>, %arg1: memref<200xi8>) -> memref<200xi8> {
  %0 = "decisionforest.ensemble"() { forest = #decisionforest<Forest = ( ReductionType = 0, #Trees = 800, InitialValue=0.5 ) }
  %c200 = arith.constant 200 : index // number of rows
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index // number of classes
  %cst = arith.constant 5.000000e-01 : f32
  %result_1 = memref.alloca() : memref<200x8x800xf32> // -> classes output <n_rows x n_classes x n_trees>, this fill be further reduced to <n_rows x n_classes> and then to <n_rows x 1>

  %c800 = arith.constant 800 : index // num trees
  %c0_3 = arith.constant 0 : index
  %c1_4 = arith.constant 1 : index
  %cst_5 = arith.constant 0.000000e+00 : f32
  scf.parallel (%arg2) = (%c0_3) to (%c800) step (%c1_4) {
    %subview = memref.subview %arg0[%arg2, 0] [1, 54] [1, 1] : memref<200x54xf32> to memref<1x54xf32, strided<[54, 1], offset: ?>>
    %c800 = arith.constant 800 : index
    %c0_3 = arith.constant 0 : index
    %c1_4 = arith.constant 1 : index
    %cst_5 = arith.constant 0.000000e+00 : f32
    %1 = scf.for %arg3 = %c0 to %c200 step %c1 iter_args(%arg4 = %cst_5) -> (f32) {
      %2 = "decisionforest.getTree"(%0, %arg2) : (!decisionforest<TreeEnsembleType(#Trees:800, rowType:memref<54xf32>, resultType:i8, reductionType:0)>, index) -> !decisionforest<TreeType(returnType:i8, tileSize:8, tileShapeType:i16, childIndexType:i1))>
      %3 = "decisionforest.walk_decision_tree"(%2, %subview) {predicate = 11 : i64} : (!decisionforest<TreeType(returnType:i8, tileSize:8, tileShapeType:i16, childIndexType:i1))>, memref<1x54xf32, strided<[54, 1], offset: ?>>) -> f32
      %4 = "decisionforest.getTreeClassId"(%0, %arg2) : (!decisionforest<TreeEnsembleType(#Trees:800, rowType:memref<54xf32>, resultType:i8, reductionType:0)>, index) -> i8
      %5 = arith.index_cast %4 : i8 to index
      reduce(%result_1, (%arg3, %5, %arg3), %3) // reduce tree-walk result into the corresponding class.
      scf.yield %arg4 : f32
    }
    scf.yield
  }
  reduce_dimension_inplace(%result_1, 2) : <reduction_type = add>
  %subview = memref.subview %result_1[0, 0, 0] [200, 8, 1] [1, 1, 1] : memref<200x8x800xf32> to memref<200x8xf32, strided<[8, 1], offset: ?>>
  reduce_dimension(%subview, 0, %arg1) : <reduction_type = max_index> // this reduces into %arg1. If %arg1 is not passed, new allocation is created.
  return %arg1 : memref<200xi8>
}
```
## TODOs
* How would we generate atomic ops for the accumulations on CPU and GPU when possible?
* Is this set of ops and lowerings sufficient for multi-class models as well?
* How do we handle multiple `reduce` ops in the same loop nest?
  * When there are multiple `reduce` ops that are accumulating into the same memref in 
  the same loop nest, it seems to be sufficient to make the set of conflicting loops 
  the union of the conflicting loops of each of the `reduce` ops. We also need to add any 
  loops that conflict with the inter-op dependencies to the list of conflicting loops. 
  The memref will be privatized based on each conflicting loop as before.
* Do these ops provide any benefit while compiling database queries?