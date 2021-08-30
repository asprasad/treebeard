#include <iostream>
#include "json/xgboostparser.h"
// #include "mlir/Dialect.h"

// #include "mlir/DecisionTreeAttributes.h"
// #include "mlir/DecisionTreeTypes.h"

using namespace std;

mlir::decisionforest::DecisionTree<> decisionTree;

namespace mlir
{
namespace decisionforest
{
void LowerFromHighLevelToMidLevelIR(mlir::MLIRContext& context, mlir::ModuleOp module);
void LowerEnsembleToMemrefs(mlir::MLIRContext& context, mlir::ModuleOp module);
}
}

int main(int argc, char *argv[]) {
  cout << "Tree-heavy: A compiler for gradient boosting tree inference.\n";
  
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::decisionforest::DecisionForestDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  const int32_t batchSize = 16;
  TreeHeavy::XGBoostJSONParser<> xgBoostParser(context, argv[1], batchSize);
  xgBoostParser.Parse();
  auto module = xgBoostParser.GetEvaluationFunction();
  module->dump();

  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(context, module);

  module->dump();

  mlir::decisionforest::LowerEnsembleToMemrefs(context, module);
  module->dump();

  std::vector<double> data(8);
  auto decisionForest = xgBoostParser.GetForest();
  cout << "Ensemble prediction: " << decisionForest->Predict(data) << endl;
}