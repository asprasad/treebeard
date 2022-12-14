#ifndef _TESTUTILS_COMMON_H_
#define _TESTUTILS_COMMON_H_
#include <iostream>
#include <exception>
#include <stdexcept>
#include <cmath>
#include <random>
#include <functional>
#include "DecisionForest.h"
#include "schedule.h"

namespace mlir
{
class MLIRContext;
}

namespace TreeBeard
{
namespace test 
{

using TestException = std::runtime_error;

// TODO exceptions are disabled. Need to enable them on the tests.
// Replacing the more elegant C++ method below with a macro until then.

// inline void Test_ASSERT(bool predicate, std::string message = "") {
//   if (!predicate)
//     std::cout << "Test_ASSERT Failed : " << message << std::endl;
//   assert (predicate);
// }

#define Test_ASSERT(predicate) { \
  bool predicateVal = predicate; \
  if (!predicateVal) {\
    std::cout << "\nTest_ASSERT Failed : " << #predicate << std::endl; \
    assert(false); \
    return false; \
  } \
}

struct TestArgs_t {
  mlir::MLIRContext& context;
};

typedef bool(*TestFunc_t)(TestArgs_t& args);

struct TestDescriptor {
	std::string m_testName;
	TestFunc_t m_testFunc;
};

#define TEST_LIST_ENTRY(testName) { std::string(#testName), testName }

template<typename FPType>
inline bool FPEqual(FPType a, FPType b) {
  const FPType scaledThreshold = std::max(std::fabs(a), std::fabs(b))/1e8;
  const FPType threshold = std::max(FPType(1e-6), scaledThreshold);
  auto sqDiff = (a-b) * (a-b);
  return sqDiff < threshold;
}

template <>
inline bool FPEqual<float>(float a, float b) {
  using FPType = float;
  const FPType scaledThreshold = std::max(std::fabs(a), std::fabs(b))/1e8;
  const FPType threshold = std::max(FPType(1e-6), scaledThreshold);
  auto sqDiff = (a-b) * (a-b);
  return sqDiff < threshold;
}

template <>
inline bool FPEqual<int32_t>(int32_t a, int32_t b) {
  return a == b;
}

template <>
inline bool FPEqual<int8_t>(int8_t a, int8_t b) {
  return a == b;
}

using RandomIntGenerator = std::function<int32_t()>;
using RandomRealGenerator = std::function<double()>;

inline int32_t GetRandomInt(int32_t min, int32_t max) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<int32_t> dist(min, max);
    return dist(dev);
}

inline double GetRandomReal(double min, double max) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<> dist(min, max);
    return dist(dev);
}

inline bool IsFloatType(const double&) { return true; }
inline bool IsFloatType(const float&) { return true; }
inline bool IsFloatType(const int8_t&) { return false; }
// inline bool IsFloatType(const int16_t&) { return false; }
// inline bool IsFloatType(const int32_t&) { return false; }

class TestCSVReader {
  std::vector<std::vector<double>> m_data;
  std::string m_filename;
public:
  TestCSVReader(const std::string& filename, int32_t numLines=-1);
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

std::string GetTreeBeardRepoPath();
std::string GetTempFilePath();
std::string GetGlobalJSONNameForTests();

mlir::decisionforest::DecisionForest<> GenerateRandomDecisionForest(int32_t numTrees, int32_t numFeatures, double thresholdMin,
                                                                    double thresholdMax, int32_t maxDepth);
void SaveToXGBoostJSON(mlir::decisionforest::DecisionForest<>& forest, const std::string& filename);
void GenerateRandomModelJSONs(const std::string& dirname, int32_t numberOfModels, int32_t maxNumTrees, 
                              int32_t maxNumFeatures, double thresholdMin, double thresholdMax, int32_t maxDepth);

void RunTests();
void RunSanityTests();
void RunXGBoostBenchmarks();
void RunXGBoostParallelBenchmarks();

// ===---------------------------------------------=== //
// Configuration for tests
// ===---------------------------------------------=== //

// Defined in XGBoostTests.cpp
extern bool RunSingleBatchSizeForXGBoostTests;

} // test
} // TreeBeard

#endif // _TESTUTILS_COMMON_H_
