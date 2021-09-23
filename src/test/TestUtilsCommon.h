#ifndef _TESTUTILS_COMMON_H_
#define _TESTUTILS_COMMON_H_
#include <iostream>
#include <exception>
#include <stdexcept>
#include <cmath>

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
  const FPType threshold = 1e-9;
  return std::fabs(a - b) < threshold;
}

} // test


#endif // _TESTUTILS_COMMON_H_