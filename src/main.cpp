#include <iostream>
#include "json/xgboostparser.h"
// #include "mlir/DecisionTreeAttributes.h"
// #include "mlir/DecisionTreeTypes.h"
#include "mlir/Dialect.h"

using namespace std;

mlir::decisionforest::DecisionTree<> decisionTree;

int main(int argc, char *argv[]) {
  cout << "Tree-heavy: A compiler for gradient boosting tree inference.\n";
  TreeHeavy::XGBoostJSONParser<double, double, int, int> xgBoostParser(argv[1]);
  xgBoostParser.Parse();

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::decisionforest::DecisionForestDialect>();
  auto treeType = mlir::decisionforest::TreeType::get(::mlir::FloatType::getF64(&context));
  auto treeAttribute = mlir::decisionforest::DecisionTreeAttr::get(treeType, 42);
  // auto treeAttribute = mlir::detail::AttributeUniquer::get<mlir::decisionforest::DecisionTree>(&context, treeType, 42);
}