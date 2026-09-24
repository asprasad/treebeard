add_llvm_library(treebeard-runtime SHARED
    runtime.cpp tbruntime.h)

add_dependencies(treebeard-runtime DecisionForestGen)

llvm_update_compile_flags(treebeard-runtime)
target_compile_options(treebeard-runtime PRIVATE -fexceptions)
target_link_libraries(treebeard-runtime PRIVATE
    ${TREEBEARD_DEPENDENCY_LIBS}
    ${CMAKE_DL_LIBS})
set_target_properties(treebeard-runtime PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH ON)
configure_file(tbruntime.h ${CMAKE_BINARY_DIR}/include/tbruntime.h COPYONLY)

install(FILES "$<TARGET_FILE:treebeard-runtime>"
    DESTINATION treebeard
    RENAME libtreebeard-runtime.so
    COMPONENT python)
install(FILES ${CMAKE_SOURCE_DIR}/src/python/__init__.py
    DESTINATION treebeard
    COMPONENT python)
install(FILES ${CMAKE_SOURCE_DIR}/src/python/treebeard.py
    DESTINATION treebeard
    RENAME _api.py
    COMPONENT python)
install(FILES ${CMAKE_SOURCE_DIR}/src/python/treebeard_runtime_api.py
    DESTINATION treebeard
    RENAME _runtime_api.py
    COMPONENT python)
install(FILES ${CMAKE_SOURCE_DIR}/THIRD_PARTY_NOTICES.md
    DESTINATION treebeard
    COMPONENT python)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(TREEBEARD_OPENMP_RUNTIME
        NAMES omp libomp
        HINTS "${LLVM_BUILD_DIRECTORY}/lib" ${LLVM_LIBRARY_DIRS}
        REQUIRED)
    get_filename_component(TREEBEARD_OPENMP_RUNTIME_REALPATH
        "${TREEBEARD_OPENMP_RUNTIME}" REALPATH)
    install(FILES "${TREEBEARD_OPENMP_RUNTIME_REALPATH}"
        DESTINATION treebeard
        RENAME libomp.so
        COMPONENT python)
endif()

if(NV_GPU_SUPPORT)
    find_library(TREEBEARD_MLIR_CUDA_RUNTIME
        NAMES mlir_cuda_runtime
        HINTS ${TREEBEARD_GPU_RUNTIME_DIR} "${LLVM_BUILD_DIRECTORY}/lib"
        ${LLVM_LIBRARY_DIRS}
        REQUIRED)
    get_filename_component(TREEBEARD_MLIR_CUDA_RUNTIME_REALPATH
        "${TREEBEARD_MLIR_CUDA_RUNTIME}" REALPATH)
    install(FILES "${TREEBEARD_MLIR_CUDA_RUNTIME_REALPATH}"
        DESTINATION treebeard
        RENAME libmlir_cuda_runtime.so
        COMPONENT python)

    find_library(TREEBEARD_MLIR_RUNNER_UTILS
        NAMES mlir_runner_utils
        HINTS "${LLVM_BUILD_DIRECTORY}/lib" ${LLVM_LIBRARY_DIRS}
        REQUIRED)
    get_filename_component(TREEBEARD_MLIR_RUNNER_UTILS_REALPATH
        "${TREEBEARD_MLIR_RUNNER_UTILS}" REALPATH)
    install(FILES "${TREEBEARD_MLIR_RUNNER_UTILS_REALPATH}"
        DESTINATION treebeard
        RENAME libmlir_runner_utils.so
        COMPONENT python)
endif()

include_directories(../include)
include_directories(../json)
include_directories(../mlir)
include_directories(../test)
include_directories(../utils)
include_directories(../schedule)