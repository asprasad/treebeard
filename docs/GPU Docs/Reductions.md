# Reductions : Representation, Optimization and Lowering 

Currently, existing reduction support in MLIR is insufficient 
to code generate and optimize the reductions Treebeard needs
to perform while performing inference (sum up individual 
tree predictions to compute the prediction of the model). 
MLIR only supports reductions of value types and cannot 
directly represent and optimize inplace reductions of 
several elements of a memory buffer. 

We design a mechanism to specify accumulating values into
an element of a multi-dimensional array inplace. 
The accumulation is performed inside an arbitrary 
loop nest where several surrounding loops maybe parallel and 
the ultimate target machine maybe a CPU or a GPU. Because this problem 
is of general interest, we design this as an MLIR dialect. 

The main abstraction we introduce is the `reduce` op. It models 
atomically accumulating values into an element of a 
multi-dimensional array (represented by a MLIR `memref`).
The following example sums up the elements of the 1D memref `arr`
into the first element of the memref `result`. It does this
using in two concurrent iterations of a surrounding 
parallel loop.

```C++
  builtin.func @ReduceVector(%arr: memref<num_elemsxf64>, %result: memref<1xf64>) -> void {
    par.for i0 in range(0 : num_elems/2 : num_elems) {
      for i1 in range(0 : num_elems/2) 
        reduce(%result, 0, arr[i0 + i1]) <"+", 0.0>
    }
  }
```

The semantics of the `reduce` op guarantee that all elements are 
correctly added and that there is no race between the parallel 
iterations of the loop.

The `reduce` op is defined for all associative and commutative
reduction operations with a well defined initial value. The 
reduction operator and the initial value are attributes applied
on the `reduce` op. 

The main differences between our `reduce` and the existing 
reductions in MLIR are the following:
1. Existing reductions only support scalar, by value reductions. 
It does not support accumulating inplace into memref elements.
2. Existing reduction support in MLIR does not provide a unified
way to handle reductions in GPUs and CPUs. To the best of our 
knowledge, there is currently no way to model reductions on GPUs
in MLIR.
2. Existing reductions have strict rules about where they can be 
written. For example, `scf.reduce` needs to be an immediate 
child `scf.parallel`. However, our `reduce` op can be
generated anywhere in the loop nest. 

Having modeled the reductions with an abstract op, the 
aim now is to lower this to a correct and optimized 
implementation on both CPU and GPU. In order to do this,
we make the following observations. 
1. The `reduce` op has a loop carried dependency on itself
and loop carried dependences on other `reduce` ops that
accumulate into the same taret array. 
A simple lowering to a sequence of load-add-store instructions is 
incorrect if any of these dependences are carried by a parallel 
loop. We call any such surrounding parallel loop a 
**conflicting loop** (TODO Change the name!) for the reduction.
2. There is a race between the parallel iterations of such a loop 
when naively accumulating values into target memref elements. 
To avoid this race, the result memref can be **privatized** wrt each 
surrounding conflicting loop. Subsequently, each privatized dimension 
can be reduced at the end of the conflicting loop it was inserted for. 
TODO: We cannot do better than this in terms of memory usage (TODO Need a proof).


_**Definition:**_
A parallel loop surrounding one or more `reduce` ops is 
a **conflicting loop** for a target multi-dimensional array if this 
loop has a non-zero dependence distance for the dependence between
any of the contained `reduce` ops. 

In the context of Treebeard, this set of loops is exactly the set 
of surrounding parallel loops that are iterating over trees. The 
results can be privatized for each conflicting loop iteratively 
and reductions along each privatized dimension can be inserted  
immediately following the loop the dimension was inserted due to.

We illustrate this process through the example above. The `i0` loop is 
a conflicting loop for the reduction into the `result` array.
We would therefore privatize the `result` memref for each iteration of 
the `i0` loop. 
```C++
  builtin.func @ReduceVector(%arr: memref<num_elemsxf64>, %result: memref<1xf64>) -> void {
    results_1 = memref<2x1xf64>
    par.for i0 = range(0 : num_elems/2 : num_elems) {
      for i1 = range(0 : num_elems/2) 
        reduce(%result_1[i0/(num_elems/2), 0], arr[i0 + i1]) <"+", 0.0>
    }
    results = reduce_dimension(results_1, 0) <"+", 0.0>
  }
```
The op `reduce_dimension` reduces values across the specified
dimension of an n-dimensional memref. In the above example, 
the `reduce_dimension` op is reducing across all elements 
of the first dimension (index 0). Therefore, in this case, it 
produces a result memref with a single element (the first dimension
with size 2 is collapsed). 

_**Definition:**_
**`reduce_dimension(targetMemref, memref, dim, [indices], [rangeStart], [rangeEnd])`:**
   Computes the reduction over the dimension specified by `dimension` and stores the 
   result in `targetMemref`. `[indices]` must be a vector of `dim` elements
   (or empty if the dimension being reduced is the first dimension). `[rangeStart]` 
   and `[rangeEnd]` represent the range of indices following the reduction dimension and 
   must have the same number of elements. If both are `null` (not passed), 
   all elements of these dimensions are reduced. The computation performed by the op is as follows.

  $targetMemref[\vec{\boldsymbol{indices}}, \vec{\boldsymbol{k}}] = \sum_{i=0}^{shape[dim]} memref[\vec{\boldsymbol{indices}}, i, \vec{\boldsymbol{k}}]\quad   \forall \vec{\boldsymbol{k}} \in \left[[rangeStart_0, rangeEnd_0), ... , [rangeStart_n, rangeEnd_n)\right]$


Consider the following code with nested parallel loops. 
(A situation where trees are split across both
threads and thread blocks could result in such generated 
code in Treebeard.)
```C++
  builtin.func @ReduceVector(%arr: memref<num_elemsxf64>, %result: memref<1xf64>) -> void {
    par.for i0 = range(0 : num_elems/2 : num_elems) {
      par.for i1 = range(0 : num_elems/4 : num_elems/2) {
        for i2 = range(0 : num_elems/4) 
          reduce(%result, 0, arr[i0 + i1 + i2]) <"+", 0.0>
      }
    }
  }
```

Here, the `i0` and `i1` loops are conflicting loops wrt 
the `result` memref. We **legalize** the reduction 
by privatizing the `result` array wrt the `i0` and `i1` loops.
However, there are now two privatized dimensions and therefore, 
two dimensions need to be reduced to compute the final result.
This multi-stage reduction is what enables us to model 
hierarchical reductions.

The following code shows how the reduction above is legalized. 
We introduce a new op, `reduce_dimension_inplace` which 
reduces a dimension of the input memref and stores results 
in the same array. This helps saves memory by removing the 
need to create multiple intermediate arrays to store results.
Only the final dimension reduction uses the `reduce_dimension` op.

```C++
  builtin.func @ReduceVector(%arr: memref<num_elemsxf64>, %result: memref<1xf64>) -> void {
    results_1 = memref<2x2x1xf64>
    par.for i0 = range(0 : num_elems/2 : num_elems) {
      par.for i1 = range(0 : num_elems/4 : num_elems/2) {
        for i2 = range(0 : num_elems/4) 
          index0 = i0/(num_elems/2)
          index1 = i1/(num_elems/4)
          reduce(%result_1[index0, index1, 0], arr[i0 + i1 + i2]) <"+", 0.0>
      }
      // result_1[i0/(num_elems/2), 0] = sum(result_1[i0/(num_elems/2), :])
      reduce_dimension_inplace(%result_1, 1, i0/(num_elems/2)) 
    }
    // result = sum(result[:, 0])
    %result = reduce_dimension(%result_1, 0)
  }
```
The behavior of the `reduce_dimension_inplace` op is similar to the 
`reduce_dimension` op except that it updates the input array inplace
rather than writing results to a target array. The definition of the 
op is as follows. 

_**Definition:**_
**`reduce_dimension_inplace(memref, dim, [indices], [rangeStart], [rangeEnd])`**
  Computes the reduction over the dimension specified by `dimension` and stores the 
  result at index 0 of that dimension. `[indices]` must be a vector of `dim` elements
   (or empty if the dimension being reduced is the first dimension). `[rangeStart]` 
   and `[rangeEnd]` must have the same number of elements. If both are `null` (not passed), 
   all elements of the corresponding dimension are reduced. 
  
  The computation performed by the op is defined by the following equation.
  
  $memref[\vec{\boldsymbol{indices}}, 0, \vec{\boldsymbol{k}}] = \sum_{i=0}^{shape[dim]} memref[\vec{\boldsymbol{indices}}, i, \vec{\boldsymbol{k}}]\quad   \forall \vec{\boldsymbol{k}} \in \left[[rangeStart_0, rangeEnd_0), ... , [rangeStart_n, rangeEnd_n)\right]$  

## Lowering Reduction Operations
We implement lowering of the operations defined above to both the CPU and GPU.
Since the comilation pipeline diverges after the reductions are legalized, 
we can implement lowering and optimization of our reduction dialect to CPUs and
GPUs simply using different MLIR rewrite patterns. We now briefly describe 
how these operations are lowered to the CPU and GPU. 

### Lowering to CPU
The lowering of the reduction operations to CPU is fairly straighforward. We lower the 
two operations listed above, `reduce_dimension_inplace` and 
`reduce_dimension` to a simple loop nest that goes over the specified
subset of the input array, performs the reduction and writes 
the result into the appropriate location of the target array. 
If the schedule specifies that the reduction is to be vectorized,
then as many elements as specified by the vector width are read 
from the input array as a vector, accumulated as a vector, and 
finally written back to the target array. In general, this works 
well because reductions are typically being performed on dimensions
other than the inner-most dimension and therefore, this strategy
loads successive elements from memory maximizing memory bandwidth 
utitlization. 

**TODO: explain atomic reduction** 

### Lowering to GPU
The lowering on GPU is slightly more involved than the lowering on CPUs.
However, we can lower the same abstractions to efficient implementations
and therefore simplify high-level code generation. The lowering for 
the inplace and non-inplace operations are essentially the same, except 
for the target array and we do not distinguish between them except 
for finally storing the result. 

The lowering of the `reduce_dimension_*` ops is distinct from existing 
work on implementing reductions efficiently on GPUs \cite{NVIDIAReductions}
because our abstractions potentially represent several independent reductions
(independent for different output elements).
Therefore, we can either exploit parallelism across the independent reductions or 
the inherent parallelism in the reduction by performing a divide and conquer 
reduction.

The reduction pass for GPU can follow one of two paths. If the lowering pass
determines that there are enough independent reductions to keep all threads
in a thread block busy, then it simply generates code that performs one (or 
multiple) reductions completely in a thread. If however there are not 
enough independent reductions, then the lowering pass generates a tree 
style reduction where multiple threads cooperate to perform a single reduction
using inter-thread shuffles.

Another feature specific to GPU reductions is the use of shared memory. 
If the schedule specifies that the reduction needs to be performed 
using shared memory, the privatized buffer is allocated in shared memory. 
Also, the compiler ensures that only as much shared memory is allocated 
as needed to hold values processed by a single thread-block and 
index offsets are appropriately rewritten to handle the differences between 
the indexing of the target memref and the shared memory array.
Our abstractions allow our lowering passes to be written completely 
independent of whether or not we use shared memory and therefore allow 
us to enable or disable shared memory use independently from the other 
parts of the compiler. 


## Use in Treebeard
We now present an example specific to Treebeard. The schedule with
which code is generated is as below. `N_t` is the number of trees 
and `batch_size` is the batch size. The schedule tiles both the 
batch loop and the tree loop and parallelizes the outer batch 
and tree loops.

```C++
IndexVar i0, i1, t0, t1;
auto& batch = schedule.GetBatchIndex();
auto& tree = schedule.GetTreeIndex();
schedule.Tile(batch, i0, i1, batch_size/2);
schedule.Tile(tree, t0, t1, N_t/2);
schedule.Reorder({i0, t0, t1, i1});
schedule.Parallel(t0);
schedule.Parallel(i0);
```
The loop-nest generated by Treebeard for the above schedule is as follows. 
```C++
  builtin.func @Prediction_Function(%arg0: memref<batch_sizexnum_featuresxf64>) -> memref<batch_sizexf64> {
    %result = memref.alloc <batch_sizexf64>
    %0 = #decisionforest<ReductionType = 0, #Trees = N_t, resultType = memref<batch_sizexf64>> 
    par.for i0 = range(0 : batch_size/2 : batch_size) {
      par.for t0 = range(0 : N_t/2 : N_t) {
        for t1 = range(0 : N_t/2) {
          for i1 = range(0 : batch_size/2) {
            %2 = GetTree(%0, t0 + t1) 
            %3 = WalkDecisionTree(%2, %arg0[i0+i1])
            reduce(%result, i0+i1, %3)
          }
        }
      }
    }
  }
```

Treebeard determines that the `t0` loop is a conflicting loop for the `result` 
array and therefore legalizes the reduction by inserting a privatized array 
`result_1`. The privatized dimension of this array is reduced at the end 
of the `t0` loop.

```C++
  builtin.func @Prediction_Function(%arg0: memref<batch_sizexnum_featuresxf64>) -> memref<batch_sizexf64> {
    %result = memref.alloc <batch_sizexf64>
    %result_1 = memref.alloc <2xbatch_sizexf64>
    %0 = #decisionforest<ReductionType = 0, #Trees = N_t, resultType = memref<batch_sizexf64>> 
    par.for i0 = range(0 : batch_size/2 : batch_size) {
      par.for t0 = range(0 : N_t/2 : N_t) {
        for t1 = range(0 : N_t/2) {
          for i1 = range(0 : batch_size/2) {
            %2 = GetTree(%0, t0 + t1) 
            %3 = WalkDecisionTree(%2, %arg0[i0+i1])
            reduce(%result, i0+i1, %3)
          }
        }
      }
      %result[i0 : i0+step] = reduce_dimension(%result_1, 0, i0 : i0+step)
    }
  }
```
While legalizing the reduction, the compiler determines that the 
`reduce_dimension` operation must only process a subset of the final 
result that is computed within the current parallel iteration of the 
`i0` loop. Once this process is complete, the `reduce` ops in 
the result IR can be lowered to a simple "read-accumulate-write" 
sequence of instructions

Finally, we note that in our experiments, we found that our 
current implementation of lowering the reduction operations 
was sufficient and reduction is not the bottleneck in our 
generated code. However, we believe this approach to enabling 
higher level code generators to easily generate reductions 
through simple abstractions and then having the compiler 
automatically lower them to efficient implementation is an
important area for future work with applicability in several 
different domains. 

**TODO Should we mention how we handle multi-class models?**