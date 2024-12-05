#include "CompileUtils.h"
#include "ModelSerializers.h"
#include "Representations.h"
#include "StatsUtils.h"
#include "TestUtilsCommon.h"
#include "TreebeardContext.h"
#include "include/TreeTilingUtils.h"
#include "mlir/ExecutionHelpers.h"
#include "json/xgboostparser.h"
#include <iostream>
#include <string>
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

namespace TreeBeard {
namespace test {
void TestTileStringGen();
void generateRandomXGBoostModels(const std::string &dirName);
} // namespace test
} // namespace TreeBeard

// Commmand line options

using namespace llvm;
static cl::opt<bool> printAfterAll("print-treebeard-ir-after-all",
    cl::desc("Print IR after each pass"), cl::init(false));

static cl::opt<bool> individual("individual", cl::desc("Enable individual mode"), cl::init(false));

static cl::opt<std::string> testName("testname", cl::desc("Test name"),
    cl::value_desc("name"), cl::init(""), cl::Hidden);



bool EqualsString(char *arg, const std::string &str) {
  return (std::string(arg) == str);
}

bool ContainsString(char *arg, const std::string &str) {
  return (std::string(arg).find(str) != std::string::npos);
}

void SetInsertDebugHelpers(int argc, char *argv[]) {
  for (int32_t i = 0; i < argc; ++i)
    if (std::string(argv[i]).find(std::string("--debugJIT")) !=
        std::string::npos) {
      mlir::decisionforest::InsertDebugHelpers = true;
      mlir::decisionforest::PrintVectors = true;
      return;
    }
}

void SetPerfNotificationListener(int argc, char *argv[]) {
  for (int32_t i = 0; i < argc; ++i)
    if (std::string(argv[i]).find(std::string(
            "--enablePerfNotificationListener")) != std::string::npos) {
      mlir::decisionforest::EnablePerfNotificationListener = true;
      return;
    }
}

void SetInsertPrintVectors(int argc, char *argv[]) {
  for (int32_t i = 0; i < argc; ++i)
    if (std::string(argv[i]).find(std::string("--printVec")) !=
        std::string::npos) {
      mlir::decisionforest::PrintVectors = true;
      return;
    }
}

bool RunXGBoostBenchmarksIfNeeded(int argc, char *argv[]) {
  for (int32_t i = 0; i < argc; ++i) {
    if (std::string(argv[i]).find(std::string("--xgboostBench")) !=
        std::string::npos) {
      TreeBeard::test::RunXGBoostBenchmarks();
      return true;
    }
    if (std::string(argv[i]).find(std::string("--xgboostGPUBench")) !=
        std::string::npos) {
      TreeBeard::test::RunXGBoostGPUBenchmarks();
      return true;
    }
  }
  return false;
}

bool RunXGBoostParallelBenchmarksIfNeeded(int argc, char *argv[]) {
  for (int32_t i = 0; i < argc; ++i)
    if (std::string(argv[i]).find(std::string("--xgboostParallelBench")) !=
        std::string::npos) {
      TreeBeard::test::RunXGBoostParallelBenchmarks();
      return true;
    }
  return false;
}

bool RunSanityTestsIfNeeded(int argc, char *argv[]) {
  for (int32_t i = 0; i < argc; ++i)
    if (std::string(argv[i]).find(std::string("--sanityTests")) !=
        std::string::npos) {
      TreeBeard::test::RunSanityTests();
      return true;
    }
  return false;
}

void ReadIntegerFromCommandLineArgument(int argc, char *argv[], int32_t &i,
                                        int32_t &targetInt) {
  assert((i + 1) < argc);
  targetInt = std::stoi(argv[i + 1]);
  i += 2;
}

bool DumpLLVMIfNeeded(int argc, char *argv[]) {
  // TODO need an additional switch here to specify whether the JSON is xgboost,
  // lightgbm etc. For now assuming xgboost
  bool dumpLLVMToFile = false;
  for (int32_t i = 0; i < argc; ++i)
    if (std::string(argv[i]).find(std::string("--dumpLLVM")) !=
        std::string::npos) {
      dumpLLVMToFile = true;
      break;
    }
  if (!dumpLLVMToFile)
    return false;
  std::string xgboostFile, llvmIRFile, modelGlobalsJSONFile,
      compilerConfigJSONFile, onnxModelFile;
  int32_t thresholdTypeWidth = 32, returnTypeWidth = 32,
          featureIndexTypeWidth = 16, tileShapeBitWidth = 16,
          childIndexBitWidth = 16;
  int32_t nodeIndexTypeWidth = 32, inputElementTypeWidth = 32, batchSize = 4,
          tileSize = 1;
  bool invertLoops = false, isReturnTypeFloat = true;
  for (int32_t i = 0; i < argc;) {
    if (EqualsString(argv[i], "-o")) {
      assert((i + 1) < argc);
      assert(llvmIRFile.empty());
      llvmIRFile = argv[i + 1];
      i += 2;
    } else if (ContainsString(argv[i], "-xgboost")) {
      assert((i + 1) < argc);
      assert(xgboostFile.empty());
      assert(onnxModelFile.empty());
      xgboostFile = argv[i + 1];
      i += 2;
    } else if (ContainsString(argv[i], "-onnx")) {
      assert((i + 1) < argc);
      assert(onnxModelFile.empty());
      assert(xgboostFile.empty());
      onnxModelFile = argv[i + 1];
      i += 2;
    } else if (ContainsString(argv[i], "-globalValuesJSON")) {
      assert((i + 1) < argc);
      assert(modelGlobalsJSONFile.empty());
      modelGlobalsJSONFile = argv[i + 1];
      i += 2;
    } else if (ContainsString(argv[i], "-compilerConfigJSON")) {
      assert((i + 1) < argc);
      compilerConfigJSONFile = argv[i + 1];
      i += 2;
    } else if (ContainsString(argv[i], "--sparse")) {
      mlir::decisionforest::UseSparseTreeRepresentation = true;
      i += 1;
    } else if (ContainsString(argv[i], "--invertLoops")) {
      invertLoops = true;
      i += 1;
    } else if (ContainsString(argv[i], "-thresholdBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, thresholdTypeWidth);
    } else if (ContainsString(argv[i], "-returnBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, returnTypeWidth);
    } else if (ContainsString(argv[i], "-returnIntBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, returnTypeWidth);
      isReturnTypeFloat = false;
    } else if (ContainsString(argv[i], "-featIndexBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, featureIndexTypeWidth);
    } else if (ContainsString(argv[i], "-nodeIndexBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, nodeIndexTypeWidth);
    } else if (ContainsString(argv[i], "-inputBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, inputElementTypeWidth);
    } else if (ContainsString(argv[i], "-tileShapeBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, tileShapeBitWidth);
    } else if (ContainsString(argv[i], "-childIndexBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, childIndexBitWidth);
    } else if (ContainsString(argv[i], "-batchSize")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, batchSize);
    } else if (ContainsString(argv[i], "-tileSize")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, tileSize);
    } else
      ++i;
  }
  assert((!xgboostFile.empty() || !onnxModelFile.empty()) &&
         !llvmIRFile.empty());
  mlir::decisionforest::ScheduleManipulationFunctionWrapper scheduleManipulator(
      mlir::decisionforest::OneTreeAtATimeSchedule);
  // TreeBeard::test::ScheduleManipulationFunctionWrapper
  // scheduleManipulator(TreeBeard::test::TileTreeDimensionSchedule<10>);

  TreeBeard::TreebeardContext tbContext;
  if (!compilerConfigJSONFile.empty()) {
    TreeBeard::CompilerOptions options(compilerConfigJSONFile);
    tbContext.options = options;
    tbContext.representation = mlir::decisionforest::ConstructRepresentation();
    tbContext.serializer =
        mlir::decisionforest::ConstructModelSerializer(modelGlobalsJSONFile);
    tbContext.modelGlobalsJSONPath = modelGlobalsJSONFile;
    tbContext.forestConstructor = nullptr; /*TODO_ForestCreator*/
  } else {
    TreeBeard::CompilerOptions options(
        thresholdTypeWidth, returnTypeWidth, isReturnTypeFloat,
        featureIndexTypeWidth, nodeIndexTypeWidth, inputElementTypeWidth,
        batchSize, tileSize, tileShapeBitWidth, childIndexBitWidth,
        TreeBeard::TilingType::kUniform, false, false,
        invertLoops ? &scheduleManipulator : nullptr);

    tbContext.options = options;
    tbContext.representation = mlir::decisionforest::ConstructRepresentation();
    tbContext.serializer =
        mlir::decisionforest::ConstructModelSerializer(modelGlobalsJSONFile);
    tbContext.modelGlobalsJSONPath = modelGlobalsJSONFile;
    tbContext.forestConstructor = nullptr; /*TODO_ForestCreator*/
  }

  if (!xgboostFile.empty()) {
    tbContext.modelPath = xgboostFile;
    TreeBeard::ConvertXGBoostJSONToLLVMIR(tbContext, llvmIRFile);
  } else {
    tbContext.modelPath = onnxModelFile;
    TreeBeard::ConvertONNXModelToLLVMIR(tbContext, llvmIRFile);
  }

  return true;
}

bool RunInferenceFromSO(int argc, char *argv[]) {
  // TODO need an additional switch here to specify whether the JSON is xgboost,
  // lightgbm etc. For now assuming xgboost
  bool runInferenceFromSO = false;
  for (int32_t i = 0; i < argc; ++i)
    if (std::string(argv[i]).find(std::string("--loadSO")) !=
        std::string::npos) {
      runInferenceFromSO = true;
      break;
    }
  if (!runInferenceFromSO)
    return false;
  std::string jsonFile, soPath, inputCSVFile, modelGlobalsJSONFile;
  int32_t thresholdTypeWidth = 32, returnTypeWidth = 32,
          featureIndexTypeWidth = 16, tileShapeBitWidth = 16,
          childIndexBitWidth = 16;
  int32_t nodeIndexTypeWidth = 32, inputElementTypeWidth = 32, batchSize = 4,
          tileSize = 1;
  bool isReturnTypeFloat = true;
  for (int32_t i = 0; i < argc;) {
    if (ContainsString(argv[i], "-so")) {
      assert((i + 1) < argc);
      assert(soPath.empty());
      soPath = argv[i + 1];
      i += 2;
    } else if (ContainsString(argv[i], "-globalValuesJSON")) {
      assert((i + 1) < argc);
      assert(modelGlobalsJSONFile.empty());
      modelGlobalsJSONFile = argv[i + 1];
      i += 2;
    } else if (ContainsString(argv[i], "--sparse")) {
      mlir::decisionforest::UseSparseTreeRepresentation = true;
      i += 1;
    } else if (ContainsString(argv[i], "-i")) {
      assert((i + 1) < argc);
      assert(inputCSVFile.empty());
      inputCSVFile = argv[i + 1];
      i += 2;
    } else if (ContainsString(argv[i], "-thresholdBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, thresholdTypeWidth);
    } else if (ContainsString(argv[i], "-returnBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, returnTypeWidth);
    } else if (ContainsString(argv[i], "-returnIntBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, returnTypeWidth);
      isReturnTypeFloat = false;
    } else if (ContainsString(argv[i], "-featIndexBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, featureIndexTypeWidth);
    } else if (ContainsString(argv[i], "-nodeIndexBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, nodeIndexTypeWidth);
    } else if (ContainsString(argv[i], "-inputBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, inputElementTypeWidth);
    } else if (ContainsString(argv[i], "-tileShapeBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, tileShapeBitWidth);
    } else if (ContainsString(argv[i], "-childIndexBitWidth")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, childIndexBitWidth);
    } else if (ContainsString(argv[i], "-batchSize")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, batchSize);
    } else if (ContainsString(argv[i], "-tileSize")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, tileSize);
    } else
      ++i;
  }
  assert(!soPath.empty());
  // TODO all these options are not needed here! We only need the information
  // that is required to infer types.
  TreeBeard::CompilerOptions options(
      thresholdTypeWidth, returnTypeWidth, isReturnTypeFloat,
      featureIndexTypeWidth, nodeIndexTypeWidth, inputElementTypeWidth,
      batchSize, tileSize, tileShapeBitWidth, childIndexBitWidth,
      TreeBeard::TilingType::kUniform, false, false, nullptr);
  TreeBeard::RunInferenceUsingSO(soPath, modelGlobalsJSONFile, inputCSVFile,
                                 options);
  return true;
}

bool ComputeInferenceStatsIfNeeded(int argc, char *argv[]) {
  bool computeInferenceStats = false;
  for (int32_t i = 0; i < argc; ++i)
    if (std::string(argv[i]).find(std::string("--computeInferenceStats")) !=
        std::string::npos) {
      computeInferenceStats = true;
      break;
    }
  if (!computeInferenceStats)
    return false;

  std::string modelName, csvPath;
  int32_t numRows = -1;
  for (int32_t i = 0; i < argc;) {
    if (ContainsString(argv[i], "-model")) {
      assert(modelName.empty());
      assert(i + 1 < argc);
      modelName = argv[i + 1];
      i += 2;
    } else if (ContainsString(argv[i], "-csv")) {
      assert(csvPath.empty());
      assert(i + 1 < argc);
      csvPath = argv[i + 1];
      i += 2;
    } else if (ContainsString(argv[i], "-n")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, numRows);
    } else
      ++i;
  }
  if (csvPath.empty()) {
    TreeBeard::Profile::ComputeForestInferenceStatsOnSampledTestInput(modelName,
                                                                      numRows);
  } else {
    TreeBeard::Profile::ComputeForestInferenceStatsOnModel(modelName, csvPath,
                                                           numRows);
  }
  return true;
}

bool ComputeProbabilityProfileIfNeeded(int argc, char *argv[]) {
  bool computeProbabilityProfile = false;
  for (int32_t i = 0; i < argc; ++i)
    if (std::string(argv[i]).find(std::string("--probabilityProfile")) !=
        std::string::npos) {
      computeProbabilityProfile = true;
      break;
    }
  if (!computeProbabilityProfile)
    return false;

  std::string modelName, csvPath, outputCSVPath;
  int32_t numRows = -1;
  for (int32_t i = 0; i < argc;) {
    if (ContainsString(argv[i], "-model")) {
      assert(modelName.empty());
      assert(i + 1 < argc);
      modelName = argv[i + 1];
      i += 2;
    } else if (ContainsString(argv[i], "-csv")) {
      assert(csvPath.empty());
      assert(i + 1 < argc);
      csvPath = argv[i + 1];
      i += 2;
    } else if (ContainsString(argv[i], "-o")) {
      assert(outputCSVPath.empty());
      assert(i + 1 < argc);
      outputCSVPath = argv[i + 1];
      i += 2;
    } else if (ContainsString(argv[i], "-n")) {
      ReadIntegerFromCommandLineArgument(argc, argv, i, numRows);
    } else
      ++i;
  }
  assert(!modelName.empty() && !csvPath.empty() && !outputCSVPath.empty());
  TreeBeard::Profile::ComputeForestProbabilityProfileForXGBoostModel(
      modelName, csvPath, outputCSVPath, numRows);
  return true;
}

bool GenerateRandomXGBoostBenchmarksIfNeeded(int argc, char *argv[]) {
  bool generateBenchmarks = false;
  for (int32_t i = 0; i < argc; ++i)
    if (std::string(argv[i]).find(std::string("--generateBenchmarks")) !=
        std::string::npos) {
      generateBenchmarks = true;
      break;
    }
  if (!generateBenchmarks)
    return false;
  std::string dirName = "/home/ashwin/mlir-build/llvm-project/mlir/examples/"
                        "treebeard/xgb_models/test/GPUBenchmarks";
  TreeBeard::test::generateRandomXGBoostModels(dirName);
  return true;
}

int main(int argc, char *argv[]) {
  cl::ParseCommandLineOptions(argc, argv, "TreeBread Runner");

  // Check if 'printAfterAll' is used without 'individual'
  if (printAfterAll && !individual) {
    llvm::errs() << "Error: 'print-treebeard-ir-after-all' can only be used with "
                    "'individual' flag.\n";
    return 1;
  }

  // Check if 'testName' is used without 'individual'
  if (!testName.empty() && !individual) {
    llvm::errs() << "Error: 'testname' can only be used with 'individual' flag.\n";
    return 1;
  }

  SetInsertDebugHelpers(argc, argv);
  SetInsertPrintVectors(argc, argv);
  SetPerfNotificationListener(argc, argv);
  if (RunSanityTestsIfNeeded(argc, argv))
    return 0;
  else if (RunXGBoostBenchmarksIfNeeded(argc, argv))
    return 0;
  else if (RunXGBoostParallelBenchmarksIfNeeded(argc, argv))
    return 0;
  else if (DumpLLVMIfNeeded(argc, argv))
    return 0;
  else if (RunInferenceFromSO(argc, argv))
    return 0;
  else if (ComputeInferenceStatsIfNeeded(argc, argv))
    return 0;
  else if (ComputeProbabilityProfileIfNeeded(argc, argv))
    return 0;
  else if (GenerateRandomXGBoostBenchmarksIfNeeded(argc, argv))
    return 0;
  else if (individual) {

    // If printAfterAll is true, set the environment variable
    if (printAfterAll) {
      setenv("PRINT_AFTER_ALL", "true", 1); // Set the environment variable
    }

    std::string individualTestName = testName;
    TreeBeard::test::RunIndividualTests(individualTestName);

    // Unset the environment variable after the tests are run
    if (printAfterAll) {
      unsetenv("PRINT_AFTER_ALL"); // Unset the environment variable
    }
  } else {
    std::cout
        << "TreeBeard: A compiler for gradient boosting tree inference.\n";
    TreeBeard::test::RunTests();
  }
  return 0;
}