# Notes on Treebeard GPU Support

This document contains notes about Treebeard's GPU support.

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

## Implementation Plan

* Implement end-to-end code generation for the simple strategy of one thread processing a single row completely reading everything from global memory
  * Correctly tile loop nests so that loops can be mapped to thread blocks and threads
  * In-memory representation of model
    * Should we implement FIL/Tahoe's representations?
  * Model reductions correctly (?)
  ```C++
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
  ```C++
  for (r0=0 ; r0<batchSize ; r0+=threadBlockSize) { // Thread blocks
    for (r1=0 ; r1<threadBlockSize ; r1++) { // Threads
      // 1. Body of this loop is the code executed by each thread.
      // 2. Each set of N_tile threads cooperate to walk as many (row, tree) 
      // pairs and each thread accumulates its corresponding row's prediction.
      pred = 0.0;
      for (int t=0 ; t<numberOfTrees ; ++t) {
        // One pointer for every thread. We're walking N_tile 
        // (row, tree) pairs in parallel
        __shared__ int nodePtrs[threadBlock.size]; 
        // Comparison outcomes
        __shared__ bool outcomes[N_tile][N_tile];
        for (int th=0 ; th<N_tile ; ++th) {
          // N_tile threads cooperate to compute predicates for
          // N_tile rows.
          auto nodePtr = nodePtrs[th];
          // Leaf checks can be avoided if we unroll the walk
          // Since all threads are walking the same tree, code can be specialized
          // by fissing the loop over the trees
          if (nodePtr == LEAF_PTR) continue;
          auto threshold = LoadThreshold(t, threadIdx, nodePtr);
          auto featureIdx = LoadFeatureIndex(t, threadIdx, nodePtr);
          auto feature = rows[r0+r1][featureIdx];
          auto result = feature < threshold;
          outcomes[th][threadIdx] = result;
          // TODO Do we really need to synchronize here since there is no divergence?
          __syncthreads();
        }
        auto nodePtr = nodePtrs[threadIdx];
        if (nodePtr == LEAF_PTR)
          pred += GetLeafValue(nodePtr, t);
        else
          // Use LUT to move to next tile
          nodePtrs[threadIdx] = MoveToNextTile(nodePtr, outcomes, t);
      }
      predictions[r0+r1] = pred;
    }
  }
  ```
* Promote rows to shared memory
  * Model cooperative load correctly
  * Model which features need to be loaded (?)
* Promote trees to shared memory
  * Reuse same cooperative loading mechanisms as above(?)
* Support different orders of walking through iteration space covered by a single thread block
  * Reduction needs to be modeled when single row is processed across threads
* Interleaving within a thread
* Strategies that need global reductions

## Tahoe Implementation Strategies

* __Direct Method__ 
  * Each thread processes one row
  * Reduction is done in registers
  * Everything is read from global memory
* __Shared Data__
  * Row is in shared memory and tree in global memory
  * Each thread block walks over all trees for a single row (each thread walks a different tree for the same row)
  * Threads accumulate partial results
  * Finally, there is a thread block wide reduction in shared memory (in the same kernel)
* __Shared Forest__
  * Entire model is loaded into shared memory
  * Rows are loaded from global memory
  * One thread performs full inference for one row (so reduction is in a register)
    ```C++
      auto& batchIndex = schedule.GetBatchIndex();
      auto& treeIndex = schedule.GetTreeIndex();
      
      auto& b0 = schedule.NewIndexVariable("b0");
      auto& b1 = schedule.NewIndexVariable("b1");

      auto& t0 = schedule.NewIndexVariable("t0");
      auto& t1 = schedule.NewIndexVariable("t1");
      
      schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);
      b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid,
                         decisionforest::IndexVariable::Dimension::X);
      b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
                         decisionforest::IndexVariable::Dimension::X);
      
      schedule.Tile(treeIndex, t0, t1, schedule.GetForestSize());
      schedule.Cache(t0);
    ```
* __Shared Partial Forest__
  * Part of model is loaded into shared memory
  * Row is read from global memory
  * One thread walks all trees in shared memory for a single row (partial results are reduced in a register)
  * Reduction across partial results is performed in global memory (as separate kernels)
    ```C++
      auto& batchIndex = schedule.GetBatchIndex();
      auto& treeIndex = schedule.GetTreeIndex();
      
      auto& b0 = schedule.NewIndexVariable("b0");
      auto& b1 = schedule.NewIndexVariable("b1");

      auto& t0 = schedule.NewIndexVariable("t0");
      auto& t0Inner = schedule.NewIndexVariable("t0Inner");
      auto& t1 = schedule.NewIndexVariable("t1");
      auto& t2 = schedule.NewIndexVariable("t2");
      
      schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);
      
      schedule.Tile(treeIndex, t0, t1, treesPerThreadBlock);
      schedule.Tile(t0Inner, t1, t2, treesPerThreadBlock);
      schedule.Cache(t1);
      schedule.Reorder(std::vector<decisionforest::IndexVariable*>{&b0, &t0, &b1, &t1, &t2});

      b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid, 
                        decisionforest::IndexVariable::Dimension::X);
      t0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid,
                        decisionforest::IndexVariable::Dimension::Y);
      b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock,
                        decisionforest::IndexVariable::Dimension::X);
    ```

## Implementation Details
* __High-level IR__
  * No changes should be needed
* __Mid-level IR__ 
  * We do not currently model reduction explicitly and just directly generate direct accumulation either into a memref or a value. This needs to change.
    * Non-issue for first stage of implementation where we will only have reduction within a thread.
  * Map outer loops to thread blocks and threads. All other loops should stay the same. 
  * Lowering of WalkDecisionTree and InterleavedWalkDecisionTree may need to change for GPU execution. (Mostly to handle shared memory. Anything else?)
* __Low-level IR__
  * Probably need to rewrite the representations and lowering to LLVM (may not be needed for first stage)
* __Representation__
  * How would we support something like the interleaved format used by FIL?
    * Not needed for first stage of implementation
* __Initialization for Prediction__
  * Should we be generating the functions to initialize model buffers in GPU memory? This would be better than handwriting in CUDA so we can support AMD GPUs automatically.

## Question
* Can the implementation options listed above cover the variants that Tahoe implements?
* How would each of these be represented in our dialect? What extensions are needed and what can be reused?
* Are all these extremely fixed strategies? Why do we need a compiler? What do we expect to change across GPUs or models?
  * Depths of trees
  * What features to load
  * Loop tiling -- size of shared memory and what goes into shared memory
  * Tree tile size (?)