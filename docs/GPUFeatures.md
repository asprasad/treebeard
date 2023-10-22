# Contributions
1. Scheduling language
    2. loop structure, caching, parallelism
    3. Works across targets
3. Unified abstractions and optimisations that allow compiler reuse across targets
    3. The same loop and tree-walk optimizations work on both CPU and GPU.
    4. Lowering is also shared until LIR lowering.
4. Reduction modelling and code generation
    1. The reduction dialect ops and how they model reductions.
    2. The process of "legalizing" reductions
    3. Optimization of reductions and scheduling (naturally allows hierarchical reductions and optimizations at different levels)
5. [Doubtful] Tiling on the GPU
    1. Doesn't seem like we'll get much performance from tiling on GPU  
6. Abstraction of representation and targets in the compiler
    1. Caching and it’s implementation

# Evaluation
1. Different models needing different schedules
2. Different batch sizes needing different schedules
    1. This is possible because the amount of parallelism available is different with different batch sizes
3. Different GPUs needing different schedules for the same model
4. Comparison with HB, Tahoe and RAPIDS
5. Can we also do more stuff for CPU? Paralleize across trees for small batch sizes for example?

# Holes
1. How do we unroll walks when we parallelize across trees?
  1. This is also something that is needed for GPUs
2. Shared memory usage. Currently don’t handle overflows.
3. Can we do anything with the branch probabilities for GPU compilation?
4. Overlap of transfers and computation
5. Caching implementations are not optimized (especially for coalescing)
6. Also see [TODOs](TODOs)
