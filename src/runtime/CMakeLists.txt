add_llvm_library(treebeard-runtime SHARED
                 runtime.cpp)

add_dependencies(treebeard-runtime DecisionForestGen)

llvm_update_compile_flags(treebeard-runtime)
target_link_libraries(treebeard-runtime PRIVATE ${LIBS})

include_directories(../include)
include_directories(../json)
include_directories(../mlir)
include_directories(../test)
include_directories(../utils)
include_directories(../schedule)