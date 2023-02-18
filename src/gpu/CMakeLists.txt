target_sources(treebeard 
PRIVATE 
GPUSupportUtils.cpp
GPURepresentations.cpp
LowerGPUToLLVM.cpp
GPUExecutionHelper.cpp
ReorgForestRepresentation.cpp
GPUModelSerializers.cpp)

target_sources(treebeard-runtime 
PRIVATE
GPUSupportUtils.cpp
GPURepresentations.cpp
LowerGPUToLLVM.cpp
GPUExecutionHelper.cpp
ReorgForestRepresentation.cpp
GPUModelSerializers.cpp)