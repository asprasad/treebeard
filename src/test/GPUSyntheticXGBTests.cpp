#ifdef TREEBEARD_GPU_SUPPORT

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <thread>
#include <vector>

#include "ExecutionHelpers.h"
#include "TiledTree.h"
#include "TreeTilingUtils.h"
#include "forestcreator.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "xgboostparser.h"
#include "llvm/ADT/STLExtras.h"

#include "CompileUtils.h"
#include "ForestTestUtils.h"
#include "GPUCompileUtils.h"
#include "GPUExecutionHelper.h"
#include "GPUModelSerializers.h"
#include "GPUSchedules.h"
#include "GPUSupportUtils.h"
#include "ModelSerializers.h"
#include "ReorgForestRepresentation.h"
#include "Representations.h"
#include "TestUtilsCommon.h"

using namespace mlir;
using namespace mlir::decisionforest;

namespace TreeBeard {
namespace test {
// ===---------------------------------------------------=== //
// XGBoost Scalar Inference Tests
// ===---------------------------------------------------=== //
template <typename FloatType, typename FeatureIndexType,
          typename ResultType = FloatType>
bool Test_GPUCodeGenForXGBJSON(
    TestArgs_t &args, int64_t batchSize, const std::string &modelJsonPath,
    const std::string &csvPath, int32_t tileSize, int32_t tileShapeBitWidth,
    int32_t childIndexBitWidth,
    std::function<void(decisionforest::Schedule &)> scheduleManipulatorFunc,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation) {
  using NodeIndexType = int32_t;
  int32_t floatTypeBitWidth = sizeof(FloatType) * 8;
  ScheduleManipulationFunctionWrapper scheduleManipulator(
      scheduleManipulatorFunc);
  TreeBeard::CompilerOptions options(
      floatTypeBitWidth, sizeof(ResultType) * 8, IsFloatType(ResultType()),
      sizeof(FeatureIndexType) * 8, sizeof(NodeIndexType) * 8,
      floatTypeBitWidth, batchSize, tileSize, tileShapeBitWidth,
      childIndexBitWidth, TreeBeard::TilingType::kUniform,
      true,  // makeAllLeavesSameDepth
      false, // reorderTrees
      &scheduleManipulator);

  auto modelGlobalsJSONFilePath = serializer->GetFilePath();

  TreeBeard::TreebeardContext tbContext(modelJsonPath, modelGlobalsJSONFilePath,
                                        options, representation, serializer,
                                        nullptr /*TODO_ForestCreator*/);
  tbContext.SetForestCreatorType("xgboost_json");
  auto module = TreeBeard::ConstructGPUModuleFromTreebeardContext(tbContext);

  GPUInferenceRunner inferenceRunner(serializer, module, tileSize,
                                     sizeof(FloatType) * 8,
                                     sizeof(FeatureIndexType) * 8);

  return ValidateModuleOutputAgainstCSVdata<FloatType, ResultType>(
      inferenceRunner, csvPath, batchSize);
}

template <typename FloatType, typename FeatureIndexType>
bool Test_RandomXGBoostJSONs_SingleFolder(
    TestArgs_t &args, int32_t batchSize,
    const std::string &modelDirRelativePath, int32_t tileSize,
    int32_t tileShapeBitWidth, int32_t childIndexBitWidth,
    std::function<void(decisionforest::Schedule &)> scheduleManipulatorFunc,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation) {
  auto repoPath = GetTreeBeardRepoPath();
  auto testModelsDir = repoPath + "/" + modelDirRelativePath;
  auto modelListFile = testModelsDir + "/ModelList.txt";
  std::ifstream fin(modelListFile);
  if (!fin)
    return false;
  while (!fin.eof()) {
    std::string modelJSONName;
    std::getline(fin, modelJSONName);
    auto modelJSONPath = testModelsDir + "/" + modelJSONName;
    // std::cout << "Model file : " << modelJSONPath << std::endl;
    auto testResult = (Test_GPUCodeGenForXGBJSON<FloatType, FeatureIndexType>(
        args, batchSize, modelJSONPath, modelJSONPath + ".csv", tileSize,
        tileShapeBitWidth, childIndexBitWidth, scheduleManipulatorFunc,
        serializer, representation));
    Test_ASSERT(testResult);
  }
  return true;
}

template <typename FloatType, typename FeatureIndexType = int32_t>
bool Test_RandomXGBoostJSONs_1Tree(
    TestArgs_t &args, int32_t batchSize, int32_t tileSize,
    int32_t tileShapeBitWidth, int32_t childIndexBitWidth,
    std::function<void(decisionforest::Schedule &)> scheduleManipulatorFunc,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation) {
  return Test_RandomXGBoostJSONs_SingleFolder<FloatType, FeatureIndexType>(
      args, batchSize, "xgb_models/test/Random_1Tree", tileSize,
      tileShapeBitWidth, childIndexBitWidth, scheduleManipulatorFunc,
      serializer, representation);
}

template <typename FloatType, typename FeatureIndexType = int32_t>
bool Test_RandomXGBoostJSONs_2Trees(
    TestArgs_t &args, int32_t batchSize, int32_t tileSize,
    int32_t tileShapeBitWidth, int32_t childIndexBitWidth,
    std::function<void(decisionforest::Schedule &)> scheduleManipulatorFunc,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation) {
  return Test_RandomXGBoostJSONs_SingleFolder<FloatType, FeatureIndexType>(
      args, batchSize, "xgb_models/test/Random_2Tree", tileSize,
      tileShapeBitWidth, childIndexBitWidth, scheduleManipulatorFunc,
      serializer, representation);
}

template <typename FloatType, typename FeatureIndexType = int32_t>
bool Test_RandomXGBoostJSONs_4Trees(
    TestArgs_t &args, int32_t batchSize, int32_t tileSize,
    int32_t tileShapeBitWidth, int32_t childIndexBitWidth,
    std::function<void(decisionforest::Schedule &)> scheduleManipulatorFunc,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation) {
  return Test_RandomXGBoostJSONs_SingleFolder<FloatType, FeatureIndexType>(
      args, batchSize, "xgb_models/test/Random_4Tree", tileSize,
      tileShapeBitWidth, childIndexBitWidth, scheduleManipulatorFunc,
      serializer, representation);
}

//------------------------------------------------------------//
// Array representation - scalar - Random XGBoost JSONs
//------------------------------------------------------------//

bool Test_GPU_1TreeXGB_Array_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<double>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_2TreeXGB_Array_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<double>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_4TreeXGB_Array_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_1TreeXGB_Array_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<float, int16_t>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_2TreeXGB_Array_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_4TreeXGB_Array_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}
//------------------------------------------------------------//
// Sparse representation - scalar - Random XGBoost JSONs
//------------------------------------------------------------//

bool Test_GPU_1TreeXGB_Sparse_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_2TreeXGB_Sparse_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_4TreeXGB_Sparse_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_1TreeXGB_Sparse_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_2TreeXGB_Sparse_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_4TreeXGB_Sparse_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

//------------------------------------------------------------//
// Reorg representation - scalar - Random XGBoost JSONs
//------------------------------------------------------------//

bool Test_GPU_1TreeXGB_Reorg_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_2TreeXGB_Reorg_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_4TreeXGB_Reorg_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_1TreeXGB_Reorg_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_2TreeXGB_Reorg_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_4TreeXGB_Reorg_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

//------------------------------------------------------------//
// Array representation - Tile Size 4 - Random XGBoost JSONs
//------------------------------------------------------------//

bool Test_GPU_1TreeXGB_Array_Tile4(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 4;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<double>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_2TreeXGB_Array_Tile4(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 4;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<double>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_4TreeXGB_Array_Tile4(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 4;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_1TreeXGB_Array_Tile4_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 4;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<float, int16_t>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_2TreeXGB_Array_Tile4_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 4;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_4TreeXGB_Array_Tile4_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 4;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

//------------------------------------------------------------//
// Sparse representation - Tile Size 4 - Random XGBoost JSONs
//------------------------------------------------------------//

bool Test_GPU_1TreeXGB_Sparse_Tile4(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 4;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_2TreeXGB_Sparse_Tile4(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 4;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_4TreeXGB_Sparse_Tile4(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 4;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_1TreeXGB_Sparse_Tile4_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 4;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_2TreeXGB_Sparse_Tile4_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 4;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_4TreeXGB_Sparse_Tile4_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 4;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

//------------------------------------------------------------//
// Cache full model - Array representation - scalar - Random XGBoost JSONs
//------------------------------------------------------------//

bool Test_GPU_SharedForest_1TreeXGB_Array_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::TahoeSharedForestStrategy,
                std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<double>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_SharedForest_2TreeXGB_Array_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::TahoeSharedForestStrategy,
                std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<double>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_SharedForest_4TreeXGB_Array_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::TahoeSharedForestStrategy,
                std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_SharedForest_1TreeXGB_Array_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::TahoeSharedForestStrategy,
                std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<float, int16_t>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_SharedForest_2TreeXGB_Array_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::TahoeSharedForestStrategy,
                std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_SharedForest_4TreeXGB_Array_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::TahoeSharedForestStrategy,
                std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 1, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

//------------------------------------------------------------//
// Cache full model - Sparse representation - scalar - Random XGBoost JSONs
//------------------------------------------------------------//

bool Test_GPU_SharedForest_1TreeXGB_Sparse_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::TahoeSharedForestStrategy,
                std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_SharedForest_2TreeXGB_Sparse_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::TahoeSharedForestStrategy,
                std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_SharedForest_4TreeXGB_Sparse_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::TahoeSharedForestStrategy,
                std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_SharedForest_1TreeXGB_Sparse_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::TahoeSharedForestStrategy,
                std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_SharedForest_2TreeXGB_Sparse_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::TahoeSharedForestStrategy,
                std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_SharedForest_4TreeXGB_Sparse_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::TahoeSharedForestStrategy,
                std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

//------------------------------------------------------------//
// Cache full model - Reorg representation - scalar - Random XGBoost JSONs
//------------------------------------------------------------//

bool Test_GPU_SharedForest_1TreeXGB_Reorg_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(TahoeSharedForestStrategy, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_SharedForest_2TreeXGB_Reorg_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(TahoeSharedForestStrategy, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_SharedForest_4TreeXGB_Reorg_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(TahoeSharedForestStrategy, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_SharedForest_1TreeXGB_Reorg_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(TahoeSharedForestStrategy, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_1Tree<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_SharedForest_2TreeXGB_Reorg_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(TahoeSharedForestStrategy, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_SharedForest_4TreeXGB_Reorg_Scalar_f32i16(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(TahoeSharedForestStrategy, std::placeholders::_1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

//------------------------------------------------------------//
// Partially cache model - Sparse - scalar - Random XGBoost JSONs
//------------------------------------------------------------//

// bool Test_GPU_SharedForest_1TreeXGB_Sparse_Scalar(TestArgs_t& args) {
//   int32_t batchSize = 64;
//   auto tileSize = 1;
//   std::function<void(decisionforest::Schedule&)> scheduleManipulator =
//   std::bind(decisionforest::TahoeSharedForestStrategy, std::placeholders::_1,
//   8); auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests(); return
//   Test_RandomXGBoostJSONs_1Tree<double>(args, batchSize, tileSize, 16, 16,
//                                                scheduleManipulator,
//                                                ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
//                                                modelGlobalsJSONPath),
//                                                RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
// }

bool Test_GPU_CachePartialForest1Tree_2TreeXGB_Sparse_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::CachePartialForestStrategy,
                std::placeholders::_1, 1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_CachePartialForest2Trees_4TreeXGB_Sparse_Scalar(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::CachePartialForestStrategy,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

// bool Test_GPU_SharedForest_1TreeXGB_Sparse_Scalar_f32i16(TestArgs_t& args) {
//   int32_t batchSize = 64;
//   auto tileSize = 1;
//   std::function<void(decisionforest::Schedule&)> scheduleManipulator =
//   std::bind(decisionforest::TahoeSharedForestStrategy, std::placeholders::_1,
//   8); auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests(); return
//   Test_RandomXGBoostJSONs_1Tree<float, int16_t>(args, batchSize, tileSize,
//   16, 16,
//                                                scheduleManipulator,
//                                                ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
//                                                modelGlobalsJSONPath),
//                                                RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
// }

bool Test_GPU_CachePartialForest2Trees_2TreeXGB_Sparse_Scalar_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::CachePartialForestStrategy,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_CachePartialForest2Trees_4TreeXGB_Sparse_Scalar_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::CachePartialForestStrategy,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

//------------------------------------------------------------//
// Partially cache model - Array - scalar - Random XGBoost JSONs
//------------------------------------------------------------//

bool Test_GPU_CachePartialForest1Tree_2TreeXGB_Array_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::CachePartialForestStrategy,
                std::placeholders::_1, 1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_CachePartialForest1Tree_4TreeXGB_Array_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::CachePartialForestStrategy,
                std::placeholders::_1, 1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_CachePartialForest2Trees_2TreeXGB_Array_Scalar_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::CachePartialForestStrategy,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_CachePartialForest2Trees_4TreeXGB_Array_Scalar_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::CachePartialForestStrategy,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

//------------------------------------------------------------//
// Partially cache model - Reorg - scalar - Random XGBoost JSONs
//------------------------------------------------------------//

bool Test_GPU_CachePartialForest1Tree_2TreeXGB_Reorg_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::CachePartialForestStrategy,
                std::placeholders::_1, 1, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_CachePartialForest2Trees_4TreeXGB_Reorg_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::CachePartialForestStrategy,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_CachePartialForest2Trees_2TreeXGB_Reorg_Scalar_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::CachePartialForestStrategy,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_CachePartialForest2Trees_4TreeXGB_Reorg_Scalar_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::CachePartialForestStrategy,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

//------------------------------------------------------------//
// TahoeSharedDataStrategy (one row per TB) - Sparse - scalar - Random XGBoost
// JSONs
//------------------------------------------------------------//
bool Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Sparse_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      TahoeSharedDataStrategy;
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Sparse_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      TahoeSharedDataStrategy;
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Sparse_Scalar_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      TahoeSharedDataStrategy;
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Sparse_Scalar_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      TahoeSharedDataStrategy;
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

//------------------------------------------------------------//
// TahoeSharedDataStrategy (one row per TB) - Array - scalar - Random XGBoost
// JSONs
//------------------------------------------------------------//
bool Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Array_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      TahoeSharedDataStrategy;
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Array_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      TahoeSharedDataStrategy;
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Array_Scalar_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      TahoeSharedDataStrategy;
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Array_Scalar_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      TahoeSharedDataStrategy;
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

//------------------------------------------------------------//
// TahoeSharedDataStrategy (one row per TB) - Reorg - scalar - Random XGBoost
// JSONs
//------------------------------------------------------------//
bool Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Reorg_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      TahoeSharedDataStrategy;
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Reorg_Scalar(TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      TahoeSharedDataStrategy;
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Reorg_Scalar_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      TahoeSharedDataStrategy;
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_2Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Reorg_Scalar_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      TahoeSharedDataStrategy;
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

//------------------------------------------------------------//
// iterativeCachedPartialForestStrategy (split trees and rows across threads)
// Sparse - scalar - Random XGBoost JSONs
//------------------------------------------------------------//
bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Scalar(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::iterativeCachedPartialForestStrategy,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Scalar_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::iterativeCachedPartialForestStrategy,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

//------------------------------------------------------------//
// iterativeCachedPartialForestStrategy (split trees and rows across threads)
// Array - scalar - Random XGBoost JSONs
//------------------------------------------------------------//
// TODO_Ashwin This test uses too much shared memory to run on holmes
bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Array_Scalar(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::iterativeCachedPartialForestStrategy,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Array_Scalar_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::iterativeCachedPartialForestStrategy,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_array",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_array"));
}

//------------------------------------------------------------//
// iterativeCachedPartialForestStrategy (split trees and rows across threads)
// Reorg - scalar - Random XGBoost JSONs
//------------------------------------------------------------//
bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Reorg_Scalar(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::iterativeCachedPartialForestStrategy,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Reorg_Scalar_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 1;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::iterativeCachedPartialForestStrategy,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_reorg"));
}

//------------------------------------------------------------//
// iterativeCachedPartialForestStrategy (split trees and rows across threads)
// Sparse - Tiled - Random XGBoost JSONs
//------------------------------------------------------------//
bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Tile4(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 4;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::iterativeCachedPartialForestStrategy_NoCache,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<double>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Tile4_f32i16(
    TestArgs_t &args) {
  int32_t batchSize = 64;
  auto tileSize = 4;
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::iterativeCachedPartialForestStrategy_NoCache,
                std::placeholders::_1, 2, 8);
  auto modelGlobalsJSONPath = test::GetGlobalJSONNameForTests();
  return Test_RandomXGBoostJSONs_4Trees<float, int16_t>(
      args, batchSize, tileSize, 16, 16, scheduleManipulator,
      ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse",
                                                       modelGlobalsJSONPath),
      RepresentationFactory::Get().GetRepresentation("gpu_sparse"));
}

} // namespace test
} // namespace TreeBeard

#endif // TREEBEARD_GPU_SUPPORT