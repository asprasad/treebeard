message("Module path ${CMAKE_MODULE_PATH}")
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS onnx.proto)
# message("Protobuf headers in ${PROTO_HDRS} and sources in ${PROTO_SRCS}")

#cmake_path(GET PROTO_HDRS PARENT_PATH PROTO_DIR)

# set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)
# add_custom_target(generated_code DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})
# add_dependencies(treebeard generated_code)
# add_dependencies(treebeard-runtime generated_code)

# target_include_directories(treebeard PRIVATE ${PROTO_DIR})
# target_include_directories(treebeard-runtime PRIVATE ${PROTO_DIR})

target_link_libraries(treebeard PRIVATE ${Protobuf_LIBRARIES})
target_link_libraries(treebeard-runtime PRIVATE ${Protobuf_LIBRARIES})

# target_sources(treebeard PRIVATE ${PROTO_SRCS})
# target_sources(treebeard-runtime PRIVATE ${PROTO_SRCS})

target_sources(treebeard PRIVATE onnx.pb.cc)
target_sources(treebeard-runtime PRIVATE onnx.pb.cc)