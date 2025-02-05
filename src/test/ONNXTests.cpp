#include "Dialect.h"
#include "TestUtilsCommon.h"
#include <filesystem>
#include <sstream>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"

#include "CompileUtils.h"
#include "ExecutionHelpers.h"
#include "ForestTestUtils.h"
#include "ModelSerializers.h"
#include "Representations.h"
#include "TreeTilingDescriptor.h"
#include "TreeTilingUtils.h"
#include "onnxmodelparser.h"
#include "xgboostparser.h"
using namespace mlir;
using namespace mlir::decisionforest;

namespace TreeBeard {
namespace test {
bool Test_ONNX_TileSize8_Abalone(TestArgs_t &args) {
#if ENABLE_ONNX_PARSER
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/onnx_models";
  auto modelPath = testModelsDir + "/abalone_onnx_model_save.onnx";
  auto csvPath = modelPath + ".csv";
  int32_t tileSize = 8;

  TreeBeard::CompilerOptions options;
  options.tileSize = tileSize;
  options.thresholdTypeWidth = 32;
  options.featureIndexTypeWidth = 32;
  options.inputElementTypeWidth = 32;
  options.batchSize = 1024;
  options.returnTypeWidth = 32;
  options.pipelineSize = 8;
  options.makeAllLeavesSameDepth = true;
  options.tilingType = TilingType::kUniform;
  options.numberOfCores = 8;
  options.numberOfFeatures = 8;

  // #TODOSampath - Delete this temp file.
  auto modelGlobalsJSONPath =
      std::filesystem::temp_directory_path() / "modelGlobals.json";

  auto *inferenceRunner = CreateInferenceRunnerForONNXModel<float>(
      modelPath.c_str(), modelGlobalsJSONPath.string().c_str(), &options);
  return ValidateModuleOutputAgainstCSVdata<float, float>(*inferenceRunner,
                                                          csvPath, 1024);
#else
  return true;
#endif // ENABLE_ONNX_PARSER
}
} // namespace test
} // namespace TreeBeard
