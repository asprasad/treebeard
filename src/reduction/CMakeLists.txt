target_sources(treebeard
PRIVATE 
LegalizeReduceOpPass.cpp
LowerReduceOps.cpp
CooperativeReduce.cpp)

target_sources(treebeard-runtime
PRIVATE
LegalizeReduceOpPass.cpp
LowerReduceOps.cpp
CooperativeReduce.cpp)
