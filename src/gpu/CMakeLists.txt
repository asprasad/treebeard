target_sources(treebeard 
PRIVATE 
GPUSupportUtils.cpp
GPURepresentations.cpp
LowerGPUToLLVM.cpp)

target_sources(treebeard-runtime 
PRIVATE
GPUSupportUtils.cpp
GPURepresentations.cpp
LowerGPUToLLVM.cpp)