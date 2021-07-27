#include <iostream>
#include "json/xgboostparser.h"
using namespace std;

int main(int argc, char *argv[]) {
  cout << "Tree-heavy: A compiler for gradient boosting tree inference.\n";
  TreeHeavy::XGBoostJSONParser<double, int, int> xgBoostParser(argv[1]);
  xgBoostParser.Parse();
}