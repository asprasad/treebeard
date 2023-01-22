target_sources(treebeard 
PRIVATE 
TestMain.cpp
CodeGenTests.cpp
TestUtilsCommon.cpp
XGBoostTests.cpp
XGBoostBenchmarks.cpp
StatsTests.cpp
XGBoostProbTiling.cpp
GPUTests.cpp)

target_sources(treebeard-runtime 
PRIVATE
TestUtilsCommon.cpp)

add_executable(runtime-tests 
  RuntimeTests.cpp)
target_link_libraries(runtime-tests ${CMAKE_DL_LIBS})