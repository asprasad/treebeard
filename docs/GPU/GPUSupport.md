# Treebeard Model Representations

This document gives an overview of Treebeard's GPU support.

## Implementation Options in Generated GPU Code

* Rows processed fully in a thread block vs Rows processed across thread blocks
  * [Single TB] Row processed entirely by one thread vs across threads
    * In latter case, reduction in shared memory is needed 
  * [Multiple TB] Reduction in global memory
    * Partial reductions in shared memory vs local memory (or registers)
* What is stored in shared memory?
  * Rows (Subsets of rows? How many rows?)
  * Trees (How many trees?)
* ILP vs SIMT
  * Should multiple (row, tree) pairs be walked in an interleaved fashion in a single thread vs split across multiple threads? 
* Tree tiling vs scalar code 
  * Tile size
  * Processing several tiles simultaneously across threads in a warp (or thread block)
* Loop structure (tiling) to expose various levels of parallelism
  * Loops that represent the iteration space over thread-blocks
  * Loops that represent the iteration space over threads in a thread block
* Reduction strategies
  * Reductions in global memory vs shared memory vs registers
* In-memory representation of the model

# Implementation Plan

* Implement end-to-end code generation for the simple strategy of one thread processing a single row completely reading everything from global memory
  * Correctly tile loop nests so that loops can be mapped to thread blocks and threads
  * In-memory representation of model
    * Should we implement FIL/Tahoe's representations?
  * Model reductions correctly (?)
  * ```C++
    for (r0=0 ; r0<batchSize ; r0+=threadBlockSize) { // Thread blocks
      for (r1=0 ; r1<threadBlockSize ; r1++) { // Threads
        // Body of this loop is the code executed by each thread
        pred = 0.0;
        for (int t=0 ; t<numberOfTrees ; ++t) {
          pred += WalkDecisionTree(trees[t], rows[r0+r1]);
        }
        predictions[r0+r1] = pred;
      }
    }
    ```
* Support tree tiling for simple case above
  * Multiple tiles in the same warp
* Promote rows to shared memory
  * Model cooperative load correctly
  * Model which features need to be loaded (?)
* Promote trees to shared memory
  * Reuse same cooperative loading mechanisms as above(?)
* Support different orders of walking through iteration space covered by a single thread block
  * Reduction needs to be modeled when single row is processed across threads
* Interleaving within a thread
* Strategies that need global reductions