#include "Dialect.h"
#include "TestUtilsCommon.h"
#include <sstream>
#include <vector>

#include "forestcreator.h"
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
#include "GPUSupportUtils.h"
#include "LowerReduceOps.h"
#include "ModelSerializers.h"
#include "Representations.h"
#include "TiledTree.h"
#include "TreeTilingDescriptor.h"
#include "TreeTilingUtils.h"
#include "xgboostparser.h"

using namespace mlir;
using namespace mlir::decisionforest;

namespace TreeBeard {
namespace test {

void ParallelizeTreesSchedule(decisionforest::Schedule &schedule) {
  auto &treeIndex = schedule.GetTreeIndex();
  auto &bathIndex = schedule.GetBatchIndex();
  schedule.Parallel(treeIndex);
  schedule.Reorder({&treeIndex, &bathIndex});
}

void ParallelizeTreesAndRows(decisionforest::Schedule &schedule) {
  const int32_t numSubBatches = 2;
  auto batchSize = schedule.GetBatchSize();
  auto tileSize = batchSize / numSubBatches;

  auto &treeIndex = schedule.GetTreeIndex();
  auto &batchIndex = schedule.GetBatchIndex();
  auto &b0 = schedule.NewIndexVariable("b0");
  auto &b1 = schedule.NewIndexVariable("b1");
  schedule.Tile(batchIndex, b0, b1, tileSize);

  schedule.Parallel(treeIndex);
  schedule.Parallel(b0);
  schedule.Reorder({&b0, &treeIndex, &b1});
}

void TileAndParallelizeTrees(decisionforest::Schedule &schedule) {
  int32_t numSubBatches = 2;
  auto numTrees = schedule.GetForestSize();
  auto tileSize = numTrees / numSubBatches;

  auto &treeIndex = schedule.GetTreeIndex();
  auto &batchIndex = schedule.GetBatchIndex();
  auto &t0 = schedule.NewIndexVariable("t0");
  auto &t1 = schedule.NewIndexVariable("t1");
  schedule.Tile(treeIndex, t0, t1, tileSize);

  schedule.Parallel(t0);
  schedule.Parallel(t1);
  schedule.Reorder({&t0, &t1, &batchIndex});
}

void tileAndParallelizeTrees_AtomicReduce(decisionforest::Schedule &schedule,
                                          int32_t numSubBatches) {
  auto numTrees = schedule.GetForestSize();
  auto tileSize = numTrees / numSubBatches;

  auto &treeIndex = schedule.GetTreeIndex();
  auto &batchIndex = schedule.GetBatchIndex();
  auto &t0 = schedule.NewIndexVariable("t0");
  auto &t1 = schedule.NewIndexVariable("t1");
  schedule.Tile(treeIndex, t0, t1, tileSize);

  schedule.Parallel(t0);
  schedule.AtomicReduce(t0);
  schedule.Reorder({&t0, &t1, &batchIndex});
}

struct NoOpDeleter {
  void operator()(ForestCreator *ptr) const {
    // Do nothing
  }
};

template <typename ThresholdType, typename IndexType>
bool VerifyCodeGeneration(
    TestArgs_t &args, const int32_t batchSize, ForestCreator &forestCreator,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation,
    int32_t tileSize, int32_t tileShapeBitwidth, int32_t childIndexBitWidth,
    std::function<void(decisionforest::Schedule &)> scheduleManipulator,
    const std::string &csvPath = "") {

  decisionforest::ScheduleManipulationFunctionWrapper
      scheduleManipulatorWrapper(scheduleManipulator);

  TreeBeard::CompilerOptions options(
      sizeof(FloatType) * 8, sizeof(FloatType) * 8, true, sizeof(IndexType) * 8,
      sizeof(IndexType) * 8, sizeof(FloatType) * 8, batchSize, tileSize,
      tileShapeBitwidth, childIndexBitWidth, TreeBeard::TilingType::kUniform,
      true, false, &scheduleManipulatorWrapper);

  // [HACK!] Create a shared pointer that points to forestCreator
  std::shared_ptr<ForestCreator> forestCreatorPtr(&forestCreator,
                                                  NoOpDeleter());

  TreeBeard::TreebeardContext tbContext(forestCreator.GetContext(), "", "",
                                        options, representation, serializer,
                                        forestCreatorPtr);

  auto module = TreeBeard::ConstructLLVMDialectModuleFromForestCreator(
      tbContext, *forestCreatorPtr);

  // module->dump();
  // return true;

  InferenceRunnerForTest inferenceRunner(serializer, module, tileSize,
                                         sizeof(ThresholdType) * 8,
                                         sizeof(IndexType) * 8);

  // return true;

  if (!csvPath.empty()) {
    return ValidateModuleOutputAgainstCSVdata<ThresholdType, int8_t>(
        inferenceRunner, csvPath, batchSize);
  }

  assert(batchSize % 2 == 0);
  std::vector<std::vector<ThresholdType>> inputData;
  inputData.emplace_back(std::vector<ThresholdType>());
  auto &firstVec = inputData.front();
  for (int32_t i = 0; i < batchSize / 2; ++i) {
    auto data = GetBatchSize2Data();
    firstVec.insert(firstVec.end(), data.front().begin(), data.front().end());
  }
  for (auto &batch : inputData) {
    assert(batch.size() % batchSize == 0);
    size_t rowSize = batch.size() / batchSize;
    std::vector<ThresholdType> result(batchSize, -1);
    inferenceRunner.RunInference<ThresholdType, ThresholdType>(batch.data(),
                                                               result.data());
    for (int64_t rowIdx = 0; rowIdx < batchSize; ++rowIdx) {
      std::vector<double> row(batch.begin() + rowIdx * rowSize,
                              batch.begin() + (rowIdx + 1) * rowSize);
      ThresholdType expectedResult =
          static_cast<ThresholdType>(forestCreator.GetForest()->Predict(row));
      Test_ASSERT(FPEqual(result[rowIdx], expectedResult));
    }
  }
  return true;
}

template <typename ThresholdType, typename IndexType>
bool Test_FixedConstructor_AnyRep(
    TestArgs_t &args, const int32_t batchSize,
    ForestConstructor_t forestConstructor,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation,
    int32_t tileSize, int32_t tileShapeBitWidth, int32_t childIndexBitWidth,
    std::function<void(decisionforest::Schedule &)> scheduleManipulator) {

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  return VerifyCodeGeneration<ThresholdType, IndexType>(
      args, batchSize, irConstructor, serializer, representation, tileSize,
      tileShapeBitWidth, childIndexBitWidth, scheduleManipulator);
}

bool Test_TreePar_LeftRightAndBalanced_DblI32(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  std::string repName = "array";
  return Test_FixedConstructor_AnyRep<double, int32_t>(
      args, 8, AddRightLeftAndBalancedTreesTwice<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          repName, modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(repName),
      1 /*tileSize*/, 32 /*tileShapeBitWidth*/, 1 /*childIndexBitWidth*/,
      ParallelizeTreesAndRows);
}

bool Test_NestedTreePar_LeftRightAndBalanced_DblI32(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  std::string repName = "array";
  return Test_FixedConstructor_AnyRep<double, int32_t>(
      args, 4, AddRightLeftAndBalancedTreesTwice<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          repName, modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(repName),
      1 /*tileSize*/, 32 /*tileShapeBitWidth*/, 1 /*childIndexBitWidth*/,
      TileAndParallelizeTrees);
}

bool Test_AtomicReduction_TwiceLeftRightAndBalanced_DblI32(TestArgs_t &args) {
  int32_t batchSize = 4;
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  std::string repName = "array";
  auto scheduleManipulator =
      std::bind(tileAndParallelizeTrees_AtomicReduce, std::placeholders::_1, 2);
  return Test_FixedConstructor_AnyRep<double, int32_t>(
      args, batchSize, AddRightLeftAndBalancedTreesTwice<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          repName, modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(repName),
      1 /*tileSize*/, 32 /*tileShapeBitWidth*/, 1 /*childIndexBitWidth*/,
      scheduleManipulator);
}

} // namespace test
} // namespace TreeBeard