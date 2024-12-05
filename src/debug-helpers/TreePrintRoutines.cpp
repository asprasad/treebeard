#include <cassert>
#include <chrono>
#include <cstdarg>
#include <fstream>
#include <iostream>
#include <string>

// TODO move these type definitions to a different file so they can be shared
// TODO We're relying on the fact that clang and llvm will compute the same
// struct layout.
template <typename ThresholdType, typename FeatureIndexType, int32_t TileSize>
struct TileType {
  ThresholdType thresholds[TileSize];
  FeatureIndexType featureIndices[TileSize];
};

template <typename ThresholdType, typename FeatureIndexType, int32_t TileSize>
struct TileTypeWithTileID {
  ThresholdType thresholds[TileSize];
  FeatureIndexType featureIndices[TileSize];
  int32_t tileShapeID;
};

template <typename ThresholdType, typename FeatureIndexType, int32_t TileSize,
          typename TileShapeType, typename ChildIndexType>
struct SparseTileType {
  ThresholdType thresholds[TileSize];
  FeatureIndexType featureIndices[TileSize];
  TileShapeType tileShapeID;
  ChildIndexType childIndex;
};

template <typename T, int32_t Rank> struct Memref {
  T *bufferPtr;
  T *alignedPtr;
  int64_t offset;
  int64_t lengths[Rank];
  int64_t strides[Rank];
};

extern "C" int64_t PrintTreePrediction(double prediction, int64_t treeIndex) {
  std::cout << "Prediction of tree " << treeIndex << " : " << prediction
            << std::endl;
  return 42;
}

extern "C" int64_t PrintNodeIndex(int64_t nodeIndex) {
  std::cout << "Node index : " << nodeIndex << std::endl;
  return 42;
}

template <int32_t tileSize>
void PrintTiledTreeToDOTFile(
    TileTypeWithTileID<double, int32_t, tileSize> *treeBuf, int64_t length,
    std::ofstream &fout) {
  fout << "digraph {\n";
  for (int64_t i = 0; i < length; ++i) {
    int64_t numChildren = tileSize + 1;
    int64_t parentIndex = (i - 1) / numChildren;
    bool isValidNode = i == 0 || (treeBuf[parentIndex].featureIndices[0] != -1);
    if (!isValidNode)
      continue;
    fout << "\t\"node" << i << "\" [ label = \"Id:" << i << ", Thres:";
    for (int32_t j = 0; j < tileSize; ++j)
      fout << " " << treeBuf[i].thresholds[j];
    fout << ", FeatIdx:";
    for (int32_t j = 0; j < tileSize; ++j)
      fout << " " << treeBuf[i].featureIndices[j];
    fout << "TileShapeID:" << treeBuf[i].tileShapeID << "\"];\n";
    if (i != 0)
      fout << "\t\"node" << parentIndex << "\" -> \"node" << i << "\";\n";
  }
  fout << "}\n";
}

using TreeTileType = TileType<double, int32_t, 1>;
extern "C" int64_t PrintTreeToDOTFile(TreeTileType *treeBuf, int64_t length,
                                      int64_t treeIndex, int64_t tileSize) {
  std::cout << "Fetching tree with index " << treeIndex << std::endl;
  std::string dotFileName = "TreeIdx_" + std::to_string(treeIndex) +
                            "_length_" + std::to_string(length) + ".dot";
  std::ofstream fout(dotFileName);
  if (tileSize == 1) {
    fout << "digraph {\n";
    for (int64_t i = 0; i < length; ++i) {
      int64_t parentIndex = (i - 1) / 2;
      bool isValidNode =
          i == 0 || (treeBuf[parentIndex].featureIndices[0] != -1);
      if (!isValidNode)
        continue;
      fout << "\t\"node" << i << "\" [ label = \"Id:" << i
           << ", Thres:" << treeBuf[i].thresholds[0]
           << ", FeatIdx:" << treeBuf[i].featureIndices[0] << "\"];\n";
      if (i != 0)
        fout << "\t\"node" << parentIndex << "\" -> \"node" << i << "\";\n";
    }

    fout << "}\n";
  } else if (tileSize == 3) {
    PrintTiledTreeToDOTFile<3>(
        reinterpret_cast<TileTypeWithTileID<double, int32_t, 3> *>(treeBuf),
        length, fout);
  } else if (tileSize == 4) {
    PrintTiledTreeToDOTFile<4>(
        reinterpret_cast<TileTypeWithTileID<double, int32_t, 4> *>(treeBuf),
        length, fout);
  } else {
    assert(false && "Unimplemented");
  }
  return 42;
}

using SparseTreeTileType = SparseTileType<double, int32_t, 1, int32_t, int32_t>;
extern "C" int64_t PrintSparseTreeToDOTFile(SparseTreeTileType *treeBuf,
                                            int64_t length, int64_t treeIndex,
                                            int64_t tileSize) {
  return 42;
}

extern "C" int64_t PrintInputRow(double *treeBuf, int64_t length,
                                 int64_t rowIndex) {
  std::cout << "Fetching input row " << rowIndex << " : { ";
  for (int64_t i = 0; i < length; ++i) {
    std::cout << treeBuf[i] << " ";
  }
  std::cout << "}\n";
  return 42;
}

extern "C" int64_t PrintInputRow_Float(float *treeBuf, int64_t length,
                                       int64_t rowIndex) {
  std::cout << "Fetching input row " << rowIndex << " : { ";
  for (int64_t i = 0; i < length; ++i) {
    std::cout << treeBuf[i] << " ";
  }
  std::cout << "}\n";
  return 42;
}

extern "C" int64_t PrintComparison(double data, double threshold,
                                   int64_t nodeIndex) {
  std::cout << "Comparison ( data:" << data << " threshold:" << threshold
            << " NodeIndex:" << nodeIndex << " )" << std::endl;
  return 42;
}

extern "C" int64_t PrintIsLeaf(int64_t nodeIndex, int32_t featureIndex,
                               int32_t outcome) {
  std::cout << "IsLeaf ( NodeIndex:" << nodeIndex
            << " featureIndex:" << featureIndex << " Outcome:"
            << (outcome ? std::string("true") : std::string("false")) << " )"
            << std::endl;
  return 42;
}

extern "C" int64_t PrintVector(int32_t kind, int32_t elementSize,
                               int32_t vectorSize, ...) {
  const int32_t floatingPointKind = 0;
  const int32_t integerKind = 1;
  std::va_list args;
  if (kind == floatingPointKind) {
    if (elementSize == 64) {
      std::cout << "(";
      va_start(args, vectorSize);
      for (int i = 0; i < vectorSize; ++i)
        std::cout << " " << va_arg(args, double);
      std::cout << " )\n";
    } else {
      assert(false);
    }
  } else if (kind == integerKind) {
    if (elementSize == 32) {
      std::cout << "(";
      va_start(args, vectorSize);
      for (int i = 0; i < vectorSize; ++i)
        std::cout << " " << va_arg(args, int32_t);
      std::cout << " )\n";
    } else if (elementSize == 64) {
      std::cout << "(";
      va_start(args, vectorSize);
      for (int i = 0; i < vectorSize; ++i)
        std::cout << " " << va_arg(args, int64_t);
      std::cout << " )\n";
    } else {
      assert(false);
    }
  } else {
    assert(false && "Unknown element type");
  }
  return 42;
}

extern "C" int64_t PrintElementAddress(void *bufPtr, int64_t index,
                                       int64_t actualIndex,
                                       int32_t elementIndex, void *elemPtr) {
  std::cout << "Buffer:" << bufPtr << " Index:" << index
            << " ActualIndex:" << actualIndex
            << " ElementIndex:" << elementIndex << " ElementPtr:" << elemPtr
            << std::endl;
  return 42;
}

std::chrono::steady_clock::time_point begin;
int64_t totalKernelTimeInMicroseconds = 0;

extern "C" int64_t getTotalKernelTime() {
  return totalKernelTimeInMicroseconds;
}

extern "C" void resetTotalKernelTime() { totalKernelTimeInMicroseconds = 0; }

extern "C" void startKernelTimer() { begin = std::chrono::steady_clock::now(); }

extern "C" void endKernelTimer() {
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  int64_t timeTaken =
      std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
          .count();
  totalKernelTimeInMicroseconds += timeTaken;
}
