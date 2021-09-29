#include <iostream>
#include <fstream>
#include <string>

// TODO move these type definitions to a different file so they can be shared
#pragma pack(push, 1)
template<typename ThresholdType, typename FeatureIndexType, int32_t TileSize>
struct TileType {
  ThresholdType thresholds[TileSize];
  FeatureIndexType featureIndices[TileSize];
};
#pragma pack(pop)

template<typename T, int32_t Rank>
struct Memref {
  T *bufferPtr;
  T *alignedPtr;
  int64_t offset;
  int64_t lengths[Rank];
  int64_t strides[Rank];
};

extern "C" int64_t PrintTreePrediction(double prediction, int64_t treeIndex) {
  std::cout << "Prediction of tree " << treeIndex << " : " << prediction << std::endl;
  return 42;
}

extern "C" int64_t PrintNodeIndex(int64_t nodeIndex) {
  std::cout << "Node index : " << nodeIndex << std::endl;
  return 42;
}

using TreeTileType = TileType<double, int32_t, 1>;
extern "C" int64_t PrintTreeToDOTFile(TreeTileType *treeBuf, int64_t length, int64_t treeIndex, int64_t tileSize) {
  std::cout << "Fetching tree with index " << treeIndex << std::endl;
  std::string dotFileName = "TreeIdx_" + std::to_string(treeIndex) + "_length_" + std::to_string(length) + ".dot";
  std::ofstream fout(dotFileName);

  fout << "digraph {\n";
  for (int64_t i=0 ; i<length ; ++i) {
    int64_t parentIndex = (i-1)/2;
    bool isValidNode = i==0 || (treeBuf[parentIndex].featureIndices[0]!=-1);
    if (!isValidNode)
      continue;
    fout << "\t\"node" << i << "\" [ label = \"Id:" << i << ", Thres:" << treeBuf[i].thresholds[0] << ", FeatIdx:" << treeBuf[i].featureIndices[0] << "\"];\n";
    if (i != 0)
      fout << "\t\"node" << parentIndex << "\" -> \"node" << i << "\";\n";
  }

  fout << "}\n";
  return 42;
}

extern "C" int64_t PrintInputRow(double *treeBuf, int64_t length, int64_t rowIndex) {
  std::cout << "Fetching input row " << rowIndex << " : { ";
  for (int64_t i=0 ; i<length ; ++i) {
    std::cout << treeBuf[i] << " ";
  }
  std::cout << "}\n";
  return 42;
}

extern "C" int64_t PrintComparison(double data, double threshold, int64_t nodeIndex) {
  std::cout << "Comparison ( data:" << data << " threshold:" << threshold << " NodeIndex:" << nodeIndex << " )" << std::endl;
  return 42;
}