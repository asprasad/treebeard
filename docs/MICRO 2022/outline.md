# Introduction
* Importance of decision trees and their performance
* Motivation for building a compiler
  * Repeated effort to write libraries and several new architectures
  * Model,batch size and architecture specific specialization and optimizations 
* Contributions
  * Compiler infrastructure to compile decision tree models
  * 

# Background
* Decision trees
* Decision tree probabilities and notation
* XGBoost

# Compiler Overview
* Overall compiler structure?
* IR and lowering
  * Ops in the MLIR dialect
  * Tree transformations
  * Lower level optimizations

# Optimizations
* Vectorization Infrastructure
  * General tiling infrastructure that supports different types of tiling
  * Some definitions and details of how this works
* Representations
  * Array representation
  * Sparse representation
* Uniform tiling
  * Tree padding 
  * Walk unrolling 
* Probability based tiling
  * Motivation 
  * Formulation
  * Algorithm (DP and greedy)
  * Proof of optimality?
* Tree reordering, loop restructuring and walk peeling for tiling
* Walk unrolling
* Pipelining and root reuse
* Parallelization

# Results
# Related work
# Conclusion and Future work
