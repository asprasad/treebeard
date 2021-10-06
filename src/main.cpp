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
void TestRandomForestGeneration();
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
      std::string modelsDir = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/xgb_models/test/Random_2Tree";
      std::cout << "Generating random models into directory : " << modelsDir << std::endl;
      TreeBeard::test::GenerateRandomModelJSONs(modelsDir, 25, 2, 20, -10.0, 10.0, 10);
      return true;
    }
  return false;
}

void RunCompilerPasses(int argc, char *argv[]) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::decisionforest::DecisionForestDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();

  const int32_t batchSize = 16;
  TreeBeard::XGBoostJSONParser<> xgBoostParser(context, argv[1], batchSize);
  xgBoostParser.Parse();
  auto module = xgBoostParser.GetEvaluationFunction();
  // module->dump();

  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(context, module);
  module->dump();

  mlir::decisionforest::LowerEnsembleToMemrefs(context, module);
  // module->dump();

  mlir::decisionforest::ConvertNodeTypeToIndexType(context, module);
  // module->dump();

  mlir::decisionforest::LowerToLLVM(context, module);
  module->dump();

  mlir::decisionforest::dumpLLVMIR(module);

  // mlir::decisionforest::InferenceRunner inferenceRunner(module);

  std::vector<double> data(8);
  auto decisionForest = xgBoostParser.GetForest();
  cout << "Ensemble prediction: " << decisionForest->Predict(data) << endl;
}

int main(int argc, char *argv[]) {
  cout << "TreeBeard: A compiler for gradient boosting tree inference.\n";
  SetInsertDebugHelpers(argc, argv);
  if (RunGenerationIfNeeded(argc, argv))
    return 0;
  else  
    TreeBeard::test::RunTests();
  // RunCompilerPasses(argc, argv);
  // TreeBeard::test::TestRandomForestGeneration();
  return 0;
}