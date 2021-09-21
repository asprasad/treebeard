#include <iostream>
#include "json/xgboostparser.h"
#include "include/TreeTilingUtils.h"
#include "mlir/ExecutionHelpers.h"

using namespace std;

namespace mlir
{
namespace decisionforest
{
void LowerFromHighLevelToMidLevelIR(mlir::MLIRContext& context, mlir::ModuleOp module);
void LowerEnsembleToMemrefs(mlir::MLIRContext& context, mlir::ModuleOp module);
void ConvertNodeTypeToIndexType(mlir::MLIRContext& context, mlir::ModuleOp module);
void LowerToLLVM(mlir::MLIRContext& context, mlir::ModuleOp module);
int dumpLLVMIR(mlir::ModuleOp module);
}
}

namespace test
{
void RunTests();
}

void RunCompilerPasses(int argc, char *argv[]) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::decisionforest::DecisionForestDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();

  const int32_t batchSize = 16;
  TreeHeavy::XGBoostJSONParser<> xgBoostParser(context, argv[1], batchSize);
  xgBoostParser.Parse();
  auto module = xgBoostParser.GetEvaluationFunction();
  // module->dump();

  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(context, module);
  // module->dump();

  mlir::decisionforest::LowerEnsembleToMemrefs(context, module);
  // module->dump();

  mlir::decisionforest::ConvertNodeTypeToIndexType(context, module);
  // module->dump();

  mlir::decisionforest::LowerToLLVM(context, module);
  // module->dump();

  mlir::decisionforest::dumpLLVMIR(module);

  mlir::decisionforest::InferenceRunner inferenceRunner(module);

  std::vector<double> data(8);
  auto decisionForest = xgBoostParser.GetForest();
  cout << "Ensemble prediction: " << decisionForest->Predict(data) << endl;
}

int main(int argc, char *argv[]) {
  cout << "Tree-heavy: A compiler for gradient boosting tree inference.\n";
  
  test::RunTests();
  // RunCompilerPasses(argc, argv);

  return 0;
}