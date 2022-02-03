#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>
#include <limits.h>
#include <unistd.h>
#include <libgen.h>
#include <dlfcn.h>

namespace TreeBeard 
{
namespace test
{

template<typename FPType>
inline bool FPEqual(FPType a, FPType b) {
  const FPType scaledThreshold = std::max(std::fabs(a), std::fabs(b))/1e8;
  const FPType threshold = std::max(FPType(1e-6), scaledThreshold);
  auto sqDiff = (a-b) * (a-b);
  return sqDiff < threshold;
}

std::string GetRuntimeSOPath() {
  char exePath[PATH_MAX];
  memset(exePath, 0, sizeof(exePath)); 
  if (readlink("/proc/self/exe", exePath, PATH_MAX) == -1)
    return std::string("");
  // std::cout << "Calculated executable path : " << exePath << std::endl;
  char *execDir = dirname(exePath);
  char *srcDir = dirname(execDir);
  char* buildDir = dirname(srcDir);
  std::string soPath = std::string(buildDir) + "/lib/libtreebeard-runtime.so";
  return soPath;
}

class RuntimeWrapper {
public:
  typedef intptr_t (*InitFunc_t)(const char* soPath, const char* modelGlobalsJSONPath);
  typedef void (*InferenceFunc_t)(intptr_t inferenceRunnerInt, void *inputs, void *results);
  typedef void (*UnInitFunc_t)(intptr_t inferenceRunnerInt);
private:
  void *m_so;

public:
  InitFunc_t Init;
  InferenceFunc_t RunInference;
  UnInitFunc_t UnInit;

  RuntimeWrapper(const std::string& runtimeSOPath) {
    m_so = dlopen(runtimeSOPath.c_str(), RTLD_NOW);
    assert (m_so);
    Init = reinterpret_cast<InitFunc_t>(dlsym(m_so, "InitializeInferenceRunner"));
    assert(Init);
    RunInference = reinterpret_cast<InferenceFunc_t>(dlsym(m_so, "RunInference"));
    assert (RunInference);
    UnInit = reinterpret_cast<UnInitFunc_t>(dlsym(m_so, "DeleteInferenceRunner"));
    assert(UnInit);
  }

  ~RuntimeWrapper() {
    dlclose(m_so);
  }
};

class TestCSVReader {
  std::vector<std::vector<double>> m_data;
  std::string m_filename;
public:
  TestCSVReader(const std::string& filename);
  std::vector<double>& GetRow(size_t index) { return m_data[index]; }
  
  template<typename T>
  std::vector<T> GetRowOfType(size_t index) {
    auto& doubleRow = GetRow(index);
    std::vector<T> row(doubleRow.size());
    std::copy(std::begin(doubleRow), std::end(doubleRow), std::begin(row));
    return row;
  }

  size_t NumberOfRows() { return m_data.size(); }
};

std::vector<double> getNextLineAndSplitIntoTokens(std::istream& str) {
    std::vector<double> result;
    std::string line;
    std::getline(str, line);

    std::stringstream lineStream(line);
    std::string cell;

    while(std::getline(lineStream, cell, ',')) {
        result.push_back(std::stof(cell));
    }
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty()) {
        // If there was a trailing comma then add an empty element.
        result.push_back(NAN);
    }
    return result;
}

TestCSVReader::TestCSVReader(const std::string& filename) {
  std::ifstream fin(filename);
  assert(fin);
  while (!fin.eof()) {
    auto row = getNextLineAndSplitIntoTokens(fin);
    m_data.push_back(row);
  }
}

}
}

using namespace TreeBeard::test;

int main() {
  RuntimeWrapper runtimeWrapper(GetRuntimeSOPath());
  std::string soPath = "/home/ashwin/temp/treebeard-outputs/abalone_t8_b200_f_i16.so";
  std::string globalsJSONPath = "/home/ashwin/temp/treebeard-outputs/abalone_t8_b200_f_i16.so.treebeard-globals.json";
  auto inferenceRunner = runtimeWrapper.Init(soPath.c_str(), globalsJSONPath.c_str());
  
  int32_t batchSize = 200;
  using FloatType = float;
  std::string csvPath = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/xgb_models/abalone_xgb_model_save.json.csv";
  TreeBeard::test::TestCSVReader csvReader(csvPath);
  std::vector<std::vector<FloatType>> inputData;
  std::vector<std::vector<FloatType>> xgBoostPredictions;
  for (size_t i=batchSize  ; i<csvReader.NumberOfRows()-1 ; i += batchSize) {
    std::vector<FloatType> batch, preds;
    for (int32_t j=0 ; j<batchSize ; ++j) {
      auto rowIndex = (i-batchSize) + j;
      auto row = csvReader.GetRowOfType<FloatType>(rowIndex);
      auto xgBoostPrediction = row.back();
      row.pop_back();
      preds.push_back(xgBoostPrediction);
      batch.insert(batch.end(), row.begin(), row.end());
    }
    inputData.push_back(batch);
    xgBoostPredictions.push_back(preds);
  }

  // size_t rowSize = csvReader.GetRow(0).size() - 1; // The last entry is the xgboost prediction
  auto currentPredictionsIter = xgBoostPredictions.begin();
  for(auto& batch : inputData) {
    assert (batch.size() % batchSize == 0);
    std::vector<FloatType> result(batchSize, -1);
    runtimeWrapper.RunInference(inferenceRunner, batch.data(), result.data());
    for(int64_t rowIdx=0 ; rowIdx<batchSize ; ++rowIdx) {
      // This needs to be a vector of doubles because the type is hardcoded for Forest::Predict
      FloatType expectedResult = (*currentPredictionsIter)[rowIdx];
      assert(FPEqual<FloatType>(result[rowIdx], expectedResult));
      // std::cout << forestPrediction << "\t" << result[rowIdx] << "\t" << expectedResult << std::endl;
    }
    ++currentPredictionsIter;
  }

  return 0;
}