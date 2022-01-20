#include <iostream>
#include <string>
#include "json/xgboostparser.h"
#include "include/TreeTilingUtils.h"
#include "mlir/ExecutionHelpers.h"
#include "TestUtilsCommon.h"
#include "CompileUtils.h"

namespace TreeBeard
{
namespace test
{
void TestTileStringGen();
}
}

bool ContainsString(char *arg, const std::string& str) {
  return (std::string(arg).find(str) != std::string::npos);
}

void SetInsertDebugHelpers(int argc, char *argv[]) {
  for (int32_t i=0 ; i<argc ; ++i)
    if (std::string(argv[i]).find(std::string("--debugJIT")) != std::string::npos) {
      mlir::decisionforest::InsertDebugHelpers = true;
      return;
    }
}

bool RunGenerationIfNeeded(int argc, char *argv[]) {
  for (int32_t i=0 ; i<argc ; ++i)
    if (std::string(argv[i]).find(std::string("--genJSONs")) != std::string::npos) {
      std::string modelsDir = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/xgb_models/test/Random_4Tree";
      std::cout << "Generating random models into directory : " << modelsDir << std::endl;
      TreeBeard::test::GenerateRandomModelJSONs(modelsDir, 25, 4, 20, -10.0, 10.0, 10);
      return true;
    }
  return false;
}

bool RunXGBoostBenchmarksIfNeeded(int argc, char *argv[]) {
  for (int32_t i=0 ; i<argc ; ++i)
    if (std::string(argv[i]).find(std::string("--xgboostBench")) != std::string::npos) {
      TreeBeard::test::RunXGBoostBenchmarks();
      return true;
    }
  return false;
}

void ReadIntegerFromCommandLineArgument(int argc, char *argv[], int32_t& i, int32_t& targetInt) {
  assert ((i+1) < argc);
  targetInt = std::stoi(argv[i+1]);
  i += 2;
}

bool DumpLLVMIfNeeded(int argc, char *argv[]) {
  // TODO need an additional switch here to specify whether the JSON is xgboost, lightgbm etc.
  // For now assuming xgboost
  bool dumpLLVMToFile = false;
  for (int32_t i=0 ; i<argc ; ++i)
    if (std::string(argv[i]).find(std::string("--dumpLLVM")) != std::string::npos) {
      dumpLLVMToFile = true;
      break;
    }
  if (!dumpLLVMToFile)
    return false;
  std::string jsonFile, llvmIRFile;
  int32_t thresholdTypeWidth=32, returnTypeWidth=32, featureIndexTypeWidth=16, tileShapeBitWidth=16, childIndexBitWidth=16;
  int32_t nodeIndexTypeWidth=32, inputElementTypeWidth=32, batchSize=4, tileSize=1;
  for (int32_t i=0 ; i<argc ; ) {
    if (ContainsString(argv[i], "-o")) {
      assert ((i+1) < argc);
      assert (llvmIRFile == "");
      llvmIRFile = argv[i+1];
      i += 2;
    }
    else if (ContainsString(argv[i], "-json")) {
      assert ((i+1) < argc);
      assert (jsonFile == "");
      jsonFile = argv[i+1];
      i += 2;
    }
    else if (ContainsString(argv[i], "--sparse")) {
      mlir::decisionforest::UseSparseTreeRepresentation = true;
      i += 1;
    }
    else if (ContainsString(argv[i], "-thresholdBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, thresholdTypeWidth);
    }
    else if (ContainsString(argv[i], "-returnBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, returnTypeWidth);
    }
    else if (ContainsString(argv[i], "-featIndexBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, featureIndexTypeWidth);
    }
    else if (ContainsString(argv[i], "-nodeIndexBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, nodeIndexTypeWidth);
    }
    else if (ContainsString(argv[i], "-inputBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, inputElementTypeWidth);
    }
    else if (ContainsString(argv[i], "-tileShapeBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, tileShapeBitWidth);
    }
    else if (ContainsString(argv[i], "-childIndexBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, childIndexBitWidth);
    }
    else if (ContainsString(argv[i], "-batchSize")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, batchSize);
    }
    else if (ContainsString(argv[i], "-tileSize")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, tileSize);
    }
    else
      ++i;
  }
  assert (jsonFile != "" && llvmIRFile != "");
  TreeBeard::CompilerOptions options(thresholdTypeWidth, returnTypeWidth, featureIndexTypeWidth,
                                     nodeIndexTypeWidth, inputElementTypeWidth, batchSize, tileSize, tileShapeBitWidth, childIndexBitWidth, nullptr);
  TreeBeard::ConvertXGBoostJSONToLLVMIR(jsonFile, llvmIRFile, options);
  return true;
}

bool RunInferenceFromSO(int argc, char *argv[]) {
  // TODO need an additional switch here to specify whether the JSON is xgboost, lightgbm etc.
  // For now assuming xgboost
  bool runInferenceFromSO = false;
  for (int32_t i=0 ; i<argc ; ++i)
    if (std::string(argv[i]).find(std::string("--loadSO")) != std::string::npos) {
      runInferenceFromSO = true;
      break;
    }
  if (!runInferenceFromSO)
    return false;
  std::string jsonFile, soPath, inputCSVFile;
  int32_t thresholdTypeWidth=32, returnTypeWidth=32, featureIndexTypeWidth=16, tileShapeBitWidth=16, childIndexBitWidth=16;
  int32_t nodeIndexTypeWidth=32, inputElementTypeWidth=32, batchSize=4, tileSize=1;
  for (int32_t i=0 ; i<argc ; ) {
    if (ContainsString(argv[i], "-so")) {
      assert ((i+1) < argc);
      assert (soPath == "");
      soPath = argv[i+1];
      i += 2;
    }
    else if (ContainsString(argv[i], "-json")) {
      assert ((i+1) < argc);
      assert (jsonFile == "");
      jsonFile = argv[i+1];
      i += 2;
    }
    else if (ContainsString(argv[i], "--sparse")) {
      mlir::decisionforest::UseSparseTreeRepresentation = true;
      i += 1;
    }
    else if (ContainsString(argv[i], "-i")) {
      assert ((i+1) < argc);
      assert (inputCSVFile == "");
      inputCSVFile = argv[i+1];
      i += 2;
    }
    else if (ContainsString(argv[i], "-thresholdBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, thresholdTypeWidth);
    }
    else if (ContainsString(argv[i], "-returnBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, returnTypeWidth);
    }
    else if (ContainsString(argv[i], "-featIndexBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, featureIndexTypeWidth);
    }
    else if (ContainsString(argv[i], "-nodeIndexBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, nodeIndexTypeWidth);
    }
    else if (ContainsString(argv[i], "-inputBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, inputElementTypeWidth);
    }
    else if (ContainsString(argv[i], "-tileShapeBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, tileShapeBitWidth);
    }
    else if (ContainsString(argv[i], "-childIndexBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, childIndexBitWidth);
    }
    else if (ContainsString(argv[i], "-batchSize")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, batchSize);
    }
    else if (ContainsString(argv[i], "-tileSize")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, tileSize);
    }
    else
      ++i;
  }
  assert (jsonFile != "" && soPath != "");
  TreeBeard::CompilerOptions options(thresholdTypeWidth, returnTypeWidth, featureIndexTypeWidth,
                                     nodeIndexTypeWidth, inputElementTypeWidth, batchSize, tileSize, tileShapeBitWidth, childIndexBitWidth, nullptr);
  TreeBeard::RunInferenceUsingSO(jsonFile, soPath, inputCSVFile, options);
  return true;
}

int main(int argc, char *argv[]) {
  SetInsertDebugHelpers(argc, argv);
  if (RunGenerationIfNeeded(argc, argv))
    return 0;
  else if (RunXGBoostBenchmarksIfNeeded(argc, argv))
    return 0;
  else if (DumpLLVMIfNeeded(argc, argv))
    return 0;
  else if (RunInferenceFromSO(argc, argv))
    return 0;
  else {  
    std::cout << "TreeBeard: A compiler for gradient boosting tree inference.\n";
    TreeBeard::test::RunTests();
  }
  return 0;
}