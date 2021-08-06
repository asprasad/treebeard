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
  // auto treeType = mlir::decisionforest::TreeType::get(::mlir::FloatType::getF64(&context));
  // auto treeAttribute = mlir::decisionforest::DecisionTreeAttr::get(treeType, 42);
  // auto treeAttribute = mlir::detail::AttributeUniquer::get<mlir::decisionforest::DecisionTree>(&context, treeType, 42);
}