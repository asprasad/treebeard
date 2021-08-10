#include <iostream>
#include "json/xgboostparser.h"
// #include "mlir/Dialect.h"

// #include "mlir/DecisionTreeAttributes.h"
// #include "mlir/DecisionTreeTypes.h"

using namespace std;

mlir::decisionforest::DecisionTree<> decisionTree;

int main(int argc, char *argv[]) {
  cout << "Tree-heavy: A compiler for gradient boosting tree inference.\n";
  
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::decisionforest::DecisionForestDialect>();
  const int32_t batchSize = 16;
  TreeHeavy::XGBoostJSONParser<> xgBoostParser(context, argv[1], batchSize);
  xgBoostParser.Parse();
  auto module = xgBoostParser.GetEvaluationFunction();
  module->dump();
  std::vector<double> data(8);
  auto decisionForest = xgBoostParser.GetForest();
  cout << "Ensemble prediction: " << decisionForest->Predict(data) << endl;
}