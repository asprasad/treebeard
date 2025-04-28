message("Module path ${CMAKE_MODULE_PATH}")

if(ENABLE_ONNX_PARSER)
    if(NOT DEFINED PROTOBUF_LIB_DIR)
        message(FATAL_ERROR "Protobuf library directory not specified. Please set PROTOBUF_LIB_DIR that points to cmake compatible protobuf library installation.")
    endif()

    set(Protobuf_DIR ${PROTOBUF_LIB_DIR}/cmake)
    find_package(Protobuf REQUIRED)

    if(NOT Protobuf_FOUND)
        message(FATAL_ERROR "Protobuf not found. Please ensure it is installed and available.")
    endif()

    include_directories(${Protobuf_INCLUDE_DIRS})
    target_link_libraries(treebeard PRIVATE ${Protobuf_LIBRARIES})
    target_link_libraries(treebeard-runtime PRIVATE ${Protobuf_LIBRARIES})
endif()

target_sources(treebeard
    PRIVATE
    ForestCreatorFactory.cpp)

target_sources(treebeard-runtime
    PRIVATE
    ForestCreatorFactory.cpp)

if(ENABLE_ONNX_PARSER)
    target_sources(treebeard
        PRIVATE
        onnx.pb.cc)

    target_sources(treebeard-runtime
        PRIVATE
        onnx.pb.cc)
endif()