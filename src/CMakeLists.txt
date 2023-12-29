option(GPU_SUPPORT "Turn on Treebeard GPU support" OFF)
cmake_minimum_required(VERSION 3.11)
include(FetchContent)

message(STATUS "---------------------------------------------")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "---------------------------------------------")


# download abseil and protobuf
set(AbseilURL https://github.com/abseil/abseil-cpp/archive/refs/tags/20230125.3.tar.gz)
set(AbseilSHA1 e21faa0de5afbbf8ee96398ef0ef812daf416ad8)
FetchContent_Declare(
  Abseil
  URL ${AbseilURL}
  URL_HASH SHA1=${AbseilSHA1}
)
set(ABSL_PROPAGATE_CXX_STD 1)
set(abseil_BUILD_TESTING 0)

message(STATUS "Download and build Abseil from ${AbseilURL}")
FetchContent_Populate(Abseil)
FetchContent_GetProperties(Abseil)

# ABSL_ROOT_DIR is required by Protobuf.
set(ABSL_ROOT_DIR ${abseil_SOURCE_DIR})
message(STATUS "Abseil source dir:" ${ABSL_ROOT_DIR})
set(ProtobufURL https://github.com/protocolbuffers/protobuf/releases/download/v22.3/protobuf-22.3.tar.gz)
set(ProtobufSHA1 310938afea334b98d7cf915b099ec5de5ae3b5c5)
FetchContent_Declare(
  Protobuf
  URL ${ProtobufURL}
  URL_HASH SHA1=${ProtobufSHA1}
)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build protobuf tests" FORCE)

message(STATUS "Download and build Protobuf from ${ProtobufURL}")
FetchContent_MakeAvailable(Protobuf Abseil)
set(ONNX_PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
set(Protobuf_VERSION "4.22.3")

message(STATUS "ONNX_PROTOC_EXECUTABLE: ${ONNX_PROTOC_EXECUTABLE}")
message(STATUS "Protobuf_VERSION: ${Protobuf_VERSION}")


execute_process(
  COMMAND ${ONNX_PROTOC_EXECUTABLE} --proto_path=json --cpp_out=json onnx.proto
  OUTPUT_VARIABLE PROTOC_OUTPUT
)
message(STATUS "${PROTOC_OUTPUT}")



get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(TREEBEARD_DEPENDENCY_LIBS
        #${dialect_libs}
        ${conversion_libs}
        #MLIROptLib
        #MLIRAnalysis
        #MLIRCallInterfaces
        #MLIRCastInterfaces
        MLIRExecutionEngine
        #MLIRIR
        #MLIRLLVMIR
        #MLIRLLVMToLLVMIRTranslation
        #MLIRParser
        #MLIRPass
        #MLIRSideEffectInterfaces
        #MLIRSupport
        #MLIRTargetLLVMIRExport
        #MLIRTransforms
        #MLIRTensorTransforms
        #MLIRSCFTransforms
        #MLIRLinalgTransforms
        MLIROpenMPToLLVMIRTranslation
        )
add_llvm_executable(treebeard main.cpp)
add_dependencies(treebeard DecisionForestGen)

llvm_update_compile_flags(treebeard)
add_definitions(-DTREEBEARD_SRC_DIR="${CMAKE_SOURCE_DIR}")
add_definitions(-DLLVM_LIB_DIR="${LLVM_BUILD_DIRECTORY}")
if(GPU_SUPPORT)
    add_definitions(-DTREEBEARD_GPU_SUPPORT=1)
endif()
mlir_check_all_link_libraries(treebeard)

add_subdirectory(runtime)
add_subdirectory(include)
add_subdirectory(json)
add_subdirectory(test)
add_subdirectory(mlir)
add_subdirectory(utils)
add_subdirectory(debug-helpers)
add_subdirectory(schedule)
add_subdirectory(gpu)

include_directories(include)
include_directories(json)
include_directories(mlir)
include_directories(test)
include_directories(utils)
include_directories(schedule)
include_directories(gpu)

target_link_libraries(treebeard PRIVATE ${TREEBEARD_DEPENDENCY_LIBS})

target_include_directories(treebeard PRIVATE ${PROTOBUF_INCLUDE_DIRS})
target_link_libraries(treebeard PRIVATE protobuf::libprotobuf)

