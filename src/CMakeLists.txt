option(GPU_SUPPORT "Turn on Treebeard GPU support" OFF)
option(ENABLE_ONNX_PARSER "Turn on Treebeard ONNX parser" OFF)

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
add_llvm_executable(treebeard
        main.cpp)
add_dependencies(treebeard DecisionForestGen)

llvm_update_compile_flags(treebeard)
add_definitions(-DTREEBEARD_SRC_DIR="${CMAKE_SOURCE_DIR}")
add_definitions(-DLLVM_LIB_DIR="${LLVM_BUILD_DIRECTORY}")
add_definitions(-DTREEBEARD_BUILD_DIR="${CMAKE_BINARY_DIR}")
if(GPU_SUPPORT)
        add_definitions(-DTREEBEARD_GPU_SUPPORT=1)
endif()
if(ENABLE_ONNX_PARSER)
        add_definitions(-DENABLE_ONNX_PARSER=1)
endif()
mlir_check_all_link_libraries(treebeard)

include_directories(include)
include_directories(json)
include_directories(mlir)
include_directories(test)
include_directories(utils)
include_directories(schedule)
include_directories(gpu)
include_directories(reduction)

add_subdirectory(runtime)
add_subdirectory(include)
add_subdirectory(json)
add_subdirectory(test)
add_subdirectory(mlir)
add_subdirectory(utils)
add_subdirectory(debug-helpers)
add_subdirectory(schedule)
add_subdirectory(gpu)
add_subdirectory(reduction)

target_link_libraries(treebeard PRIVATE ${TREEBEARD_DEPENDENCY_LIBS})
