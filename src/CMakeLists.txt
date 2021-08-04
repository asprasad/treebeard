get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        # ${dialect_libs}
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
        )
add_llvm_executable(tree-heavy
        main.cpp)
add_dependencies(tree-heavy DecisionForestGen)

llvm_update_compile_flags(tree-heavy)
target_link_libraries(tree-heavy PRIVATE ${LIBS})

mlir_check_all_link_libraries(tree-heavy)

add_subdirectory(include)
#add_subdirectory(ast)
add_subdirectory(mlir)