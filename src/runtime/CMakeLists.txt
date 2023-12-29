add_llvm_library(treebeard-runtime SHARED
                 runtime.cpp tbruntime.h)

add_dependencies(treebeard-runtime DecisionForestGen)

llvm_update_compile_flags(treebeard-runtime)
target_link_libraries(treebeard-runtime PRIVATE ${TREEBEARD_DEPENDENCY_LIBS})
configure_file(tbruntime.h ${CMAKE_BINARY_DIR}/include/tbruntime.h COPYONLY)

include_directories(../include)
include_directories(../json)
include_directories(../mlir)
include_directories(../test)
include_directories(../utils)
include_directories(../schedule)

target_include_directories(treebeard-runtime PRIVATE ${Protobuf_INCLUDE_DIRS})
target_link_libraries(treebeard-runtime PRIVATE protobuf::libprotobuf)
