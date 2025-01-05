option(AMD_GPU_SUPPORT "Turn on AMD GPU support" OFF)
option(NV_GPU_SUPPORT "Turn on NVIDIA GPU support" OFF)
option(ENABLE_ONNX_PARSER "Turn on Treebeard ONNX parser" OFF)
option(AMD_GPU_CHIPSET "AMD GPU architecture identifier" "")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(TREEBEARD_DEPENDENCY_LIBS

        # ${dialect_libs}
        ${conversion_libs}

        # MLIROptLib
        # MLIRAnalysis
        # MLIRCallInterfaces
        # MLIRCastInterfaces
        MLIRExecutionEngine

        # MLIRIR
        # MLIRLLVMIR
        MLIRLLVMToLLVMIRTranslation
        # MLIRParser
        # MLIRPass
        # MLIRSideEffectInterfaces
        # MLIRSupport
        # MLIRTargetLLVMIRExport
        # MLIRTransforms
        # MLIRTensorTransforms
        # MLIRSCFTransforms
        # MLIRLinalgTransforms
        MLIROpenMPToLLVMIRTranslation
        MLIRToLLVMIRTranslationRegistration
        MLIRBuiltinToLLVMIRTranslation
)
add_llvm_executable(treebeard
        main.cpp)
add_dependencies(treebeard DecisionForestGen)

llvm_update_compile_flags(treebeard)
add_definitions(-DTREEBEARD_SRC_DIR="${CMAKE_SOURCE_DIR}")
add_definitions(-DLLVM_LIB_DIR="${LLVM_BUILD_DIRECTORY}")
add_definitions(-DTREEBEARD_BUILD_DIR="${CMAKE_BINARY_DIR}")

if(NV_GPU_SUPPORT OR AMD_GPU_SUPPORT)
        add_definitions(-DTREEBEARD_GPU_SUPPORT=1)
endif()

if(NV_GPU_SUPPORT AND AMD_GPU_SUPPORT)
        message(FATAL_ERROR "Cannot enable both NV_GPU_SUPPORT and AMD_GPU_SUPPORT")
endif()

if(NV_GPU_SUPPORT)
        add_definitions(-DTREEBEARD_NV_GPU_SUPPORT)
elseif(AMD_GPU_SUPPORT)
        add_definitions(-DTREEBEARD_AMD_GPU_SUPPORT)

        if(AMD_GPU_CHIPSET STREQUAL "")
                message(FATAL_ERROR "AMD_GPU_CHIPSET must be set when AMD_GPU_SUPPORT is enabled")
        endif()

        add_definitions(-DTREEBEARD_AMD_GPU_CHIPSET="${AMD_GPU_CHIPSET}")
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
