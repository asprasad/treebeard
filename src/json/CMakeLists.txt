message("Module path ${CMAKE_MODULE_PATH}")

if(ENABLE_ONNX_PARSER)
    if(DEFINED PROTOBUF_LIB_DIR)
        set(Protobuf_DIR ${PROTOBUF_LIB_DIR}/cmake)
    else()
        message(STATUS "Protobuf library directory not specified. Using default path.")
    endif()

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