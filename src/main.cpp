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

  TreeHeavy::XGBoostJSONParser<> xgBoostParser(context, argv[1]);
  xgBoostParser.Parse();
  auto module = xgBoostParser.GetEvaluationFunction();
  module->dump();
}