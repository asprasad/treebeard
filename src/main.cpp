#include <iostream>
#include "json/xgboostparser.h"
#include "include/TreeTilingUtils.h"
#include "mlir/ExecutionHelpers.h"
#include "TestUtilsCommon.h"

using namespace std;

namespace TreeBeard
{
namespace test
{
void TestTileStringGen();
}
}

void SetInsertDebugHelpers(int argc, char *argv[]) {
  for (int32_t i=0 ; i<argc ; ++i)
    if (std::string(argv[i]).find(std::string("--debugJIT")) != std::string::npos) {
      mlir::decisionforest::InsertDebugHelpers = true;
      return;
    }
}

bool RunGenerationIfNeeded(int argc, char *argv[]) {
  for (int32_t i=0 ; i<argc ; ++i)
    if (std::string(argv[i]).find(std::string("--genJSONs")) != std::string::npos) {
      std::string modelsDir = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/xgb_models/test/Random_4Tree";
      std::cout << "Generating random models into directory : " << modelsDir << std::endl;
      TreeBeard::test::GenerateRandomModelJSONs(modelsDir, 25, 4, 20, -10.0, 10.0, 10);
      return true;
    }
  return false;
}

bool RunXGBoostBenchmarksIfNeeded(int argc, char *argv[]) {
  for (int32_t i=0 ; i<argc ; ++i)
    if (std::string(argv[i]).find(std::string("--xgboostBench")) != std::string::npos) {
      TreeBeard::test::RunXGBoostBenchmarks();
      return true;
    }
  return false;
}

int main(int argc, char *argv[]) {
  SetInsertDebugHelpers(argc, argv);
  if (RunGenerationIfNeeded(argc, argv))
    return 0;
  else if (RunXGBoostBenchmarksIfNeeded(argc, argv))
    return 0;
  else {  
    cout << "TreeBeard: A compiler for gradient boosting tree inference.\n";
    TreeBeard::test::RunTests();
  }
  return 0;
}