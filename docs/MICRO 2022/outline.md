# Introduction
* Importance of decision trees and their performance
* Motivation for building a compiler
  * Compilers have been successful with ML models like DNNs \cite{TVM, Tiramisu, XLA}. However, compiler techniques for decision tree ensembles are less well studied.
  * Repeated effort to write libraries and several new architectures \cite{Halide?}
  * Model, batch size and architecture specific specialization and optimizations [cite : cache conscious ensemble ranking, vpred]
  * Existing libraries \cite{xgboost, Treelite, lightgbm, vpred} target specific optimizations and are hard to maintain/upgrade as hardware evolves. They cannot tailor the generated code according to the model being used. [TODO we also need to say the search through the potential optimization space maybe somewhat limited]
* Contributions
  * We build an extensible compiler infrastructure to compile decision tree models. The infrastructure is built to allow exploration of optimization and code generation techniques. [TODO is this claim too grand? Should we be saying something like MLIR?]
  * We develop a general infrastructure for the vectorization of decision tree walks based on grouping tree nodes into "tiles". This includes general support for code generation and the in-memory representation of tiled trees. The infrastructure can be used to tile trees based on different cost functions. 
  * We show that trees can be tiled using different cost functions. We present two novel tiling methods that are implemented using the general tiling infrastructure. [TODO this is not strictly true because the prob tiling also specializes the generated code]
  * We show how tree and loop transformations can easily be implemented in our framework to enable further optimizations. [TODO Not sure how to word this -- basically want to say the pipeling optimizations are a demonstration of how we can manipulate both the trees and the loops]

# Background
* Decision trees
* Decision tree probabilities and notation
* MLIR
* XGBoost

# Compiler Overview
* Overall compiler structure
  * Written as a dialect within MLIR
  * Optimizations are written as rewrites on a higher level representation and control how the IR is lowered
  * Initially start with a tree based IR where tree orders, tiling etc are decided. Additionally, loop structure is decided on this level of the IR. [TODO should we talk about the scheduling language here?]
  * Subsequent lowering specializes tree walks as needed (unrolled, peeled, pipelined etc)
  * Finally, the model is lowered to an explicit in memory representation and tree operations like IsLeaf, TraverseTreeTile, GetLeafValue are lowered to specific code that works with the specific tree representation.
  * The compiler ultimately produces LLVM IR which is JITted or compiled into a library. The result is a function that can be called to run inference on an input array of features.

* IR and lowering
  * Ops in the MLIR dialect
  * Tree transformations
  * Lower level optimizations

# Optimizations
* Vectorization Infrastructure
  * General tiling infrastructure that supports different types of tiling
    * High level algorithm for how a tile is walked (TraverseTile). Show loading values, doing the comparison and computing the child index (use abstract next index computation)
  * Some definitions and details of how this works
    * Tile definition
    * Valid tiling
    * How the infrastructure supports tiling -- computing tile orders, padding tiles, assigning tile shapes and LUT
* Representations
  * Array representation -- simple, but causes memory to blow up. 
  * Sparse representation -- Leaves need to be moved into a different array
* Uniform tiling
  * Algorithm
  * Tree padding 
  * Loop restructuring to sort by depth and walk unrolling 
* Probability based tiling
  * Motivation (one or two plots that show  ) 
  * Formulation
  * Algorithm (DP and greedy)
  * Proof of optimality?
  * Loop restructuring by depth and peel depth. Also generation of peeled walks 
* Unrolling, Pipelining and Root reuse
* Parallelization

# Results
* Intel machine
  * Comparison with our own baseline for vectorization, prob tiling, pipelining (for 2-3 batch sizes) (Currently ~2.25X total speedup for batch size 500)
  * Single core comparison with XGBoost over several batch sizes for the fastest single core variant for each benchmark (Currently ~2.8X total speedup at batch size 500)
  * Parallel comparison with XGBoost 
  * Comparison with Treelite over several batch sizes
* AMD machine
  * Comparison with our own baseline for vectorization, prob tiling and pipelining (for 2-3 batch sizes) (Currently ~1.8X total speedup for batch size 500)
  * Comparison with XGBoost, Treelite over several batch sizes 

# Related work
# Conclusion and Future work

# Concerns
* No pipelining for prob based tiling
* No comparison with daal (we can say that is hand tuned assembly and is not portable across architectures)
* All models are XGBoost
* Distinguish from prior works -- especially QuickScorer (the bit mask based algorithm that I mentioned a while ago. We can claim this is orthogonal as it is an strategy  that can be implemented in the compiler if needed?)
  * Also need to distinguish from Humming bird
* We still don't have Treelite style code gen (blow everything up into ifs) in our compiler
* We don't do any exploration or optimizations for tiling the iteration space. Several optimizations are left unexplored.