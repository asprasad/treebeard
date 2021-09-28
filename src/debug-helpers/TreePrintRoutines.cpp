#include <iostream>

extern "C" int64_t PrintTreePrediction(double prediction, int64_t treeIndex) {
  std::cout << "Prediction of tree " << treeIndex << " : " << prediction << std::endl;
  return 42;
}

extern "C" int64_t PrintNodeIndex(int64_t nodeIndex) {
  std::cout << "Node index : " << nodeIndex << std::endl;
  return 42;
}
