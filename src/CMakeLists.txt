get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
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
target_link_libraries(treebeard PRIVATE ${LIBS})
add_definitions(-DTREEBEARD_SRC_DIR="${CMAKE_SOURCE_DIR}")
add_definitions(-DLLVM_LIB_DIR="${LLVM_BUILD_DIRECTORY}")
mlir_check_all_link_libraries(treebeard)

add_subdirectory(runtime)
add_subdirectory(include)
add_subdirectory(test)
add_subdirectory(mlir)
add_subdirectory(utils)
add_subdirectory(debug-helpers)
add_subdirectory(schedule)

include_directories(include)
include_directories(json)
include_directories(mlir)
include_directories(test)
include_directories(utils)
include_directories(schedule)