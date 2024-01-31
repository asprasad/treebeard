# Motivations
0. Importance of accelerating decision tree inference has been motivated in previous paper.
1. Decision tree inference on GPUs
    * GPUs are present in most machines today and can significantly accelerate inference.
2. Compiler that targets GPUs
    * Best set of optimizations and configurations changes across models, input sizes and target GPU
    * Specialization of code gives performance gains (as established in MICRO paper)
    * Existing solutions are written in CUDA and cannot target AMD GPUs
3. Why won't just reusing the CPU-centric techniques from the MICRO paper work?
    * Much more parallelism on the GPU means that just parallelizing across rows is insufficient.
    * More complicated memory hierarchy needs to be carefully considered.

# Contributions
1. Present the design for a mulit-target decision tree compiler infrastructure and implement several optimizations within this framework.
2. Design and implement the first GPU compiler for decision tree inference.
3. We design a scheduling language that allows us to specify generated code structure at an abstract level. 
4. We design and implement a general framework for expressing and optimizing reductions within MLIR.
5. We evaluate our implementation by comparing it against RAPIDs and Tahoe, the state of the art decision tree inference frameworks for GPU and report significant speedups.

# Outline
1. Scheduling language
    * loop structure, caching, parallelism
    * Works across targets
    * Support for optimizations on reductions
2. Unified abstractions and optimisations that allow compiler reuse across targets
    * The same loop and tree-walk optimizations work on both CPU and GPU.
    * Lowering is also shared until LIR lowering.
    * Able to reuse the same walk interleaving optimization and reduction abstractions on both CPU and GPU
3. Reduction modelling and code generation
    * The reduction dialect ops and how they model reductions.
    * The process of "legalizing" reductions
    * Optimization of reductions and scheduling (naturally allows hierarchical reductions and optimizations at different levels)
    * Support for shared memory, atomics etc.
4. Abstraction of representation and targets in the compiler
    * Structure of the compiler (how adding representations is easy etc)
    * Caching and it’s implementation
5. Specializing traversals while parallelizing across trees
6. [Doubtful] Tiling on the GPU
    1. Doesn't seem like we'll get much performance from tiling on GPU  

# Evaluation
1. [Established] Different models needing different schedules
2. [Established] Different batch sizes needing different schedules
3. [Established] Different GPUs needing different schedules for the same model
4. [Established] Comparison with RAPIDS
    * About 2X faster on average
5. Comparison with Tahoe
6. Can we also do more stuff for CPU? Parallelize across trees for small batch sizes for example?

# Some Surprising Things
1. Reading trees into shared memory degrades performance
2. In a majority of cases, the reorg representation is slower than our simpler representations.

# Holes
1. Shared memory usage exceeds max allowed. Currently, compiler just returns an error.
2. Can we do anything with the branch probabilities for GPU compilation?
3. Overlap of transfers and computation.
4. No heuristic to narrow down on optimal schedules.

# Auto-Scheduling Variables
1. Number of rows Per thread block: Integer
    * Current values: {8, 16, 32, 64, 128, 256}
    * Nothing greater than 64 is needed
2. Number of threads to parallelize trees across: Integer
    * Current values: {1, 2, 10, 20, 50}
    * Smaller batch sizes mostly like 20 and 50, while larger batch sizes like 2 and 10. 
    * Models with more trees and large rows like airline-ohe like more tree parallelization throughout. Models like covtype and letters prefer parallelization across rows at large batch sizes. (Why would this be the case?)
3. Number of rows per thread: Integer -- should typically be 1
    * Current values: {1, 2}
    * Only one benchmark seems to need 2. Even there, performance with 1 is very close.
4. Cache rows in shared memory: Bool -- typically true (unless row doesn't fit)
    * Seems like there is not much difference between True and False even when fastest is currently False. Maybe always true?
5. Pad trees to make all leaves same depth: Bool
    * Can be picked separately after picking the rest of the parameters?
6. Unroll tree walks: Bool -- typically true when there is no divergence across threads (or if the number of parallel tree threads is small)
    * Easily determined from the model whether or not this is worth trying.
7. Tree interleaving depth: Integer -- Needs tree padding to be enabled (current implementation also needs unrolling to be enabled)
    * Current values: { -1, 2, 4 }
8. Reduction in shared memory: Bool
    * Pick as a post step? Seems like the first few (10?) schedules without shared memory should be sufficient.
9. Cache trees: Bool -- always false
10. Representation
    * {array, sparse, reorg}
    * Can't find an obvious way to narrow this down. reorg seems very infrequently picked. Maybe we can pick the representation after we pick the rest of the schedule?