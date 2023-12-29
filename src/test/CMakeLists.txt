target_sources(treebeard 
PRIVATE 
${CMAKE_CURRENT_SOURCE_DIR}/TestMain.cpp
${CMAKE_CURRENT_SOURCE_DIR}/CodeGenTests.cpp
${CMAKE_CURRENT_SOURCE_DIR}/TestUtilsCommon.cpp
${CMAKE_CURRENT_SOURCE_DIR}/XGBoostTests.cpp
${CMAKE_CURRENT_SOURCE_DIR}/XGBoostBenchmarks.cpp
${CMAKE_CURRENT_SOURCE_DIR}/StatsTests.cpp
${CMAKE_CURRENT_SOURCE_DIR}/XGBoostProbTiling.cpp
${CMAKE_CURRENT_SOURCE_DIR}/ONNXTests.cpp
${CMAKE_CURRENT_SOURCE_DIR}/GPUTests.cpp)

target_sources(treebeard-runtime 
PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/TestUtilsCommon.cpp)

add_executable(runtime-tests ${CMAKE_CURRENT_SOURCE_DIR}/RuntimeTests.cpp)
target_link_libraries(runtime-tests ${CMAKE_DL_LIBS})