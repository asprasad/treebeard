#ifdef TREEBEARD_GPU_SUPPORT

#include <chrono>
#include <cstdint>
#include <dlfcn.h>
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
#include "GPUTestUtils.h"
#include "LowerReduceOps.h"
#include "ModelSerializers.h"
#include "ReorgForestRepresentation.h"
#include "Representations.h"
#include "TestUtilsCommon.h"

using namespace mlir;

using mlir::decisionforest::GPUBasicSchedule;
using mlir::decisionforest::TahoeSharedDataStrategy;
using mlir::decisionforest::TahoeSharedDataStrategy_Modified;
using mlir::decisionforest::TahoeSharedForestStrategy;
using mlir::decisionforest::TahoeSharedPartialForestStrategy;

namespace TreeBeard {
namespace test {

// ===---------------------------------------------------=== //
// GPU Model Initialization Test Helpers
// ===---------------------------------------------------=== //

void AddGPUModelMemrefGetter_Scalar(mlir::ModuleOp module) {
  // Get the tiled node type from the model memref type by finding the
  // Init_Model method
  MemRefType modelMemrefType;
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getName() == "Init_Model")
      modelMemrefType = func.getResultTypes()[0].cast<mlir::MemRefType>();
  });
  decisionforest::TiledNumericalNodeType tileType =
      modelMemrefType.getElementType()
          .cast<decisionforest::TiledNumericalNodeType>();
  auto thresholdMemrefType = MemRefType::get(
      modelMemrefType.getShape()[0], tileType.getThresholdElementType());
  auto featureIndexMemrefType = MemRefType::get(modelMemrefType.getShape()[0],
                                                tileType.getIndexElementType());
  auto getModelFunctionType = FunctionType::get(
      tileType.getContext(),
      TypeRange{modelMemrefType, thresholdMemrefType, featureIndexMemrefType},
      TypeRange{IntegerType::get(tileType.getContext(), 32)});

  auto location = module.getLoc();
  auto getModelFunc =
      func::FuncOp::create(location, "GetModelValues", getModelFunctionType);
  getModelFunc.setPublic();

  auto entryBlock = getModelFunc.addEntryBlock();
  mlir::OpBuilder builder(getModelFunc.getContext());
  builder.setInsertionPointToStart(entryBlock);

  // 1. Allocate the threshold and feature index buffers on the GPU
  // 2. Launch kernel to copy model values to these buffers
  // 3. Copy results in to the argument memrefs

  auto waitOp = builder.create<gpu::WaitOp>(
      location, gpu::AsyncTokenType::get(module.getContext()), ValueRange{});
  auto allocThreshold = builder.create<gpu::AllocOp>(
      location, thresholdMemrefType, waitOp.getAsyncToken().getType(),
      ValueRange{waitOp.getAsyncToken()}, ValueRange{}, ValueRange{});
  auto allocIndices = builder.create<gpu::AllocOp>(
      location, featureIndexMemrefType, waitOp.getAsyncToken().getType(),
      ValueRange{allocThreshold.getAsyncToken()}, ValueRange{}, ValueRange{});

  auto oneIndexConst = builder.create<arith::ConstantIndexOp>(location, 1);
  int32_t numThreadsPerBlock = 32;
  int32_t numBlocks =
      std::ceil((double)modelMemrefType.getShape()[0] / numThreadsPerBlock);
  auto numThreadBlocksConst =
      builder.create<arith::ConstantIndexOp>(location, numBlocks);
  auto numThreadsPerBlockConst =
      builder.create<arith::ConstantIndexOp>(location, numThreadsPerBlock);
  auto gpuLaunch = builder.create<gpu::LaunchOp>(
      location, numThreadBlocksConst, oneIndexConst, oneIndexConst,
      numThreadsPerBlockConst, oneIndexConst, oneIndexConst, nullptr,
      waitOp.getAsyncToken().getType(), allocIndices.getAsyncToken());

  builder.setInsertionPointToStart(&gpuLaunch.getBody().front());

  // // Generate the body of the launch op
  auto memrefLengthConst = builder.create<arith::ConstantIndexOp>(
      location, modelMemrefType.getShape()[0]);
  auto firstThreadNum = builder.create<arith::MulIOp>(
      location, gpuLaunch.getBlockSizeX(), gpuLaunch.getBlockIds().x);
  auto elementIndex = builder.create<arith::AddIOp>(location, firstThreadNum,
                                                    gpuLaunch.getThreadIds().x);
  auto inBoundsCondition = builder.create<arith::CmpIOp>(
      location, arith::CmpIPredicate::slt, elementIndex, memrefLengthConst);
  auto ifInBounds =
      builder.create<scf::IfOp>(location, inBoundsCondition, false);
  {
    // Generate the initialization code
    auto thenBuilder = ifInBounds.getThenBodyBuilder();
    auto loadThreshold =
        thenBuilder.create<decisionforest::LoadTileThresholdsOp>(
            location, tileType.getThresholdElementType(),
            getModelFunc.getArgument(0), elementIndex, oneIndexConst);
    auto loadIndex =
        thenBuilder.create<decisionforest::LoadTileFeatureIndicesOp>(
            location, tileType.getIndexElementType(),
            getModelFunc.getArgument(0), elementIndex, oneIndexConst);
    /*auto writeThreshold =*/thenBuilder.create<memref::StoreOp>(
        location, static_cast<Value>(loadThreshold), allocThreshold.getMemref(),
        static_cast<Value>(elementIndex));
    /*auto writeIndex =*/thenBuilder.create<memref::StoreOp>(
        location, static_cast<Value>(loadIndex), allocIndices.getMemref(),
        static_cast<Value>(elementIndex));
  }
  builder.create<gpu::TerminatorOp>(location);
  builder.setInsertionPointAfter(gpuLaunch);

  // Transfer back the offsets and the thresholds
  auto transferThresholds = builder.create<gpu::MemcpyOp>(
      location, gpuLaunch.getAsyncToken().getType(),
      ValueRange{gpuLaunch.getAsyncToken()}, getModelFunc.getArgument(1),
      allocThreshold.getMemref());

  auto transferIndices = builder.create<gpu::MemcpyOp>(
      location, transferThresholds.getAsyncToken().getType(),
      ValueRange{transferThresholds.getAsyncToken()},
      getModelFunc.getArgument(2), allocIndices.getMemref());

  // Wait and return
  /*auto waitBeforeReturn =*/builder.create<gpu::WaitOp>(
      location, Type(), transferIndices.getAsyncToken());
  auto returnVal = builder.create<arith::ConstantIntOp>(location, 42 /*value*/,
                                                        32 /*width*/);
  builder.create<func::ReturnOp>(location, static_cast<Value>(returnVal));

  module.push_back(getModelFunc);
}

void AddGPUModelMemrefGetter_Reorg(mlir::ModuleOp module) {
  // Get the tiled node type from the model memref type by finding the
  // Init_Model method
  MemRefType modelMemrefType, featureIndexMemrefType;
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getName() == "Init_Thresholds")
      modelMemrefType = func.getResultTypes()[0].cast<mlir::MemRefType>();
    if (func.getName() == "Init_FeatureIndices")
      featureIndexMemrefType =
          func.getResultTypes()[0].cast<mlir::MemRefType>();
  });
  auto getModelFunctionType = FunctionType::get(
      featureIndexMemrefType.getContext(),
      TypeRange{modelMemrefType, featureIndexMemrefType, modelMemrefType,
                featureIndexMemrefType},
      TypeRange{IntegerType::get(featureIndexMemrefType.getContext(), 32)});

  auto location = module.getLoc();
  auto getModelFunc =
      func::FuncOp::create(location, "GetModelValues", getModelFunctionType);
  getModelFunc.setPublic();

  auto entryBlock = getModelFunc.addEntryBlock();
  mlir::OpBuilder builder(getModelFunc.getContext());
  builder.setInsertionPointToStart(entryBlock);

  auto waitOp = builder.create<gpu::WaitOp>(
      location, gpu::AsyncTokenType::get(module.getContext()), ValueRange{});

  // auto oneIndexConst = builder.create<arith::ConstantIndexOp>(location, 1);
  // auto numThreadBlocksConst =
  // builder.create<arith::ConstantIndexOp>(location, 1); auto
  // numThreadsPerBlockConst = builder.create<arith::ConstantIndexOp>(location,
  // 7); auto gpuLaunch = builder.create<gpu::LaunchOp>(location,
  // numThreadBlocksConst,
  //                                               oneIndexConst, oneIndexConst,
  //                                               numThreadsPerBlockConst,
  //                                               oneIndexConst, oneIndexConst,
  //                                               nullptr,
  //                                               waitOp.getAsyncToken().getType(),
  //                                               waitOp.getAsyncToken());

  // builder.setInsertionPointToStart(&gpuLaunch.getBody().front());
  // auto thresholdVal = builder.create<memref::LoadOp>(location,
  // getModelFunc.getArgument(0), ValueRange{gpuLaunch.getThreadIds().x}); auto
  // indexVal = builder.create<memref::LoadOp>(location,
  // getModelFunc.getArgument(1), ValueRange{gpuLaunch.getThreadIds().x});
  // // builder.create<gpu::PrintfOp>(location, "Threshold[%ld]:
  // %lf\tIndices[%ld]: %d\n", ValueRange{ gpuLaunch.getThreadIds().x,
  // static_cast<Value>(thresholdVal),
  // // gpuLaunch.getThreadIds().x, static_cast<Value>(indexVal)});
  // builder.create<gpu::TerminatorOp>(location);
  // // Wait and return
  // builder.setInsertionPointAfter(gpuLaunch);

  // Transfer back the offsets and the thresholds
  auto transferThresholds = builder.create<gpu::MemcpyOp>(
      location, waitOp.getAsyncToken().getType(),
      ValueRange{waitOp.getAsyncToken()}, getModelFunc.getArgument(2),
      getModelFunc.getArgument(0));

  auto transferIndices = builder.create<gpu::MemcpyOp>(
      location, transferThresholds.getAsyncToken().getType(),
      ValueRange{transferThresholds.getAsyncToken()},
      getModelFunc.getArgument(3), getModelFunc.getArgument(1));

  // Wait and return
  /*auto waitBeforeReturn =*/builder.create<gpu::WaitOp>(
      location, Type(), transferIndices.getAsyncToken());
  auto returnVal = builder.create<arith::ConstantIntOp>(location, 42 /*value*/,
                                                        32 /*width*/);
  builder.create<func::ReturnOp>(location, static_cast<Value>(returnVal));

  module.push_back(getModelFunc);
}

// ===---------------------------------------------------=== //
// GPU Model Initialization Tests
// ===---------------------------------------------------=== //

template <typename ThresholdType, typename IndexType>
bool CheckGPUModelInitialization_Scalar(TestArgs_t &args,
                                        ForestConstructor_t forestConstructor) {
  // Batch size and the exact number of inputs per thread do not affect the
  // model initialization. So just hard coding those.
  const int32_t batchSize = 32;

  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  auto serializer =
      decisionforest::ConstructGPUModelSerializer(modelGlobalsJSONPath);

  TreeBeard::GPUCompileInfo compileInfo;
  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);

  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  irConstructor.ConstructForest();
  irConstructor.SetChildIndexBitWidth(1);
  auto module = irConstructor.GetEvaluationFunction();

  auto schedule = irConstructor.GetSchedule();
  GPUBasicSchedule(*schedule, 4);

  // module->dump();
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(context, module);
  mlir::decisionforest::legalizeReductionsAndCanonicalize(context, module);
  // module->dump();

  mlir::decisionforest::ConvertParallelLoopsToGPU(context, module);
  // module->dump();

  mlir::decisionforest::lowerReductionsAndCanonicalize(context, module);
  auto representation = decisionforest::ConstructGPURepresentation();
  mlir::decisionforest::LowerGPUEnsembleToMemrefs(context, module, serializer,
                                                  representation);

  mlir::decisionforest::ConvertNodeTypeToIndexType(context, module);
  AddGPUModelMemrefGetter_Scalar(module);
  // module->dump();

  mlir::decisionforest::LowerGPUToLLVM(context, module, representation,
                                       compileInfo);
  // module->dump();

  GPUInferenceRunnerForTest inferenceRunner(
      serializer, module, 1, sizeof(ThresholdType) * 8, sizeof(IndexType) * 8);

  // TODO this is a hack. We are kind of breaking the abstraction to get hold of
  // information that would otherwise be hard to get.
  auto numModelElements =
      decisionforest::ForestJSONReader::GetInstance().GetTotalNumberOfTiles();
  std::vector<ThresholdType> actualThresholds;
  std::vector<IndexType> actualFeatureIndices, actualTileShapes;
  decisionforest::ForestJSONReader::GetInstance().GetModelValues(
      1 /*tileSize*/, sizeof(ThresholdType) * 8, sizeof(IndexType) * 8,
      actualThresholds, actualFeatureIndices, actualTileShapes);

  std::vector<ThresholdType> thresholds(numModelElements, -42);
  std::vector<IndexType> featureIndices(numModelElements, -42);

  auto thresholdMemref = VectorToMemref(thresholds);
  auto featureIndicesMemref = VectorToMemref(featureIndices);
  auto modelMemref = inferenceRunner.GetModelMemref();

  int32_t retVal = -1;
  // Get the threshold values from the model memref
  std::vector<void *> funcArgs = {&modelMemref.bufferPtr,
                                  &modelMemref.alignedPtr,
                                  &modelMemref.offset,
                                  &modelMemref.lengths[0],
                                  &modelMemref.strides[0],
                                  &thresholdMemref.bufferPtr,
                                  &thresholdMemref.alignedPtr,
                                  &thresholdMemref.offset,
                                  &thresholdMemref.lengths[0],
                                  &thresholdMemref.strides[0],
                                  &featureIndicesMemref.bufferPtr,
                                  &featureIndicesMemref.alignedPtr,
                                  &featureIndicesMemref.offset,
                                  &featureIndicesMemref.lengths[0],
                                  &featureIndicesMemref.strides[0],
                                  &retVal};

  inferenceRunner.ExecuteFunction("GetModelValues", funcArgs);

  auto thresholdZip = llvm::zip(thresholds, actualThresholds);
  for (auto thresholdTuple : thresholdZip) {
    // Test_ASSERT(FPEqual(std::get<0>(thresholdTuple),
    // std::get<1>(thresholdTuple)));
    Test_ASSERT(std::get<0>(thresholdTuple) == std::get<1>(thresholdTuple));
  }
  auto featureIndexZip = llvm::zip(featureIndices, actualFeatureIndices);
  for (auto indexTuple : featureIndexZip) {
    Test_ASSERT(std::get<0>(indexTuple) == std::get<1>(indexTuple));
  }
  return true;
}

bool Test_GPUModelInit_LeftHeavy_Scalar_DoubleInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_Scalar<double, int32_t>(
      args, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_RightHeavy_Scalar_DoubleInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_Scalar<double, int32_t>(
      args, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_Balanced_Scalar_DoubleInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_Scalar<double, int32_t>(
      args, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftAndRightHeavy_Scalar_DoubleInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_Scalar<double, int32_t>(
      args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftHeavy_Scalar_FloatInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_Scalar<float, int32_t>(
      args, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_RightHeavy_Scalar_FloatInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_Scalar<float, int32_t>(
      args, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_Balanced_Scalar_FloatInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_Scalar<float, int32_t>(
      args, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftAndRightHeavy_Scalar_FloatInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_Scalar<float, int32_t>(
      args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftHeavy_Scalar_FloatInt16(TestArgs_t &args) {
  return CheckGPUModelInitialization_Scalar<float, int16_t>(
      args, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_RightHeavy_Scalar_FloatInt16(TestArgs_t &args) {
  return CheckGPUModelInitialization_Scalar<float, int16_t>(
      args, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_Balanced_Scalar_FloatInt16(TestArgs_t &args) {
  return CheckGPUModelInitialization_Scalar<float, int16_t>(
      args, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftAndRightHeavy_Scalar_FloatInt16(TestArgs_t &args) {
  return CheckGPUModelInitialization_Scalar<float, int16_t>(
      args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

// ===---------------------------------------------------=== //
// GPU Code Generation Tests -- Helpers
// ===---------------------------------------------------=== //

template <typename ThresholdType, typename IndexType,
          typename ReturnType = ThresholdType>
bool VerifyGPUCodeGeneration(
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

  auto module = ConstructGPUModuleFromTreebeardContext(tbContext);
  // return true;

  GPUInferenceRunnerForTest inferenceRunner(serializer, module, tileSize,
                                            sizeof(ThresholdType) * 8,
                                            sizeof(IndexType) * 8);

  if (!csvPath.empty() && std::filesystem::exists(csvPath)) {
    return ValidateModuleOutputAgainstCSVdata<ThresholdType, ReturnType>(
        inferenceRunner, csvPath, batchSize);
  }

  assert(batchSize % 2 == 0);
  std::vector<ThresholdType> inputData;
  for (int32_t i = 0; i < batchSize; ++i) {
    size_t numFeatures = forestCreator.GetForest()->GetFeatures().size();
    std::vector<ThresholdType> curVec;
    // generate 'numFeatures' random numbers.
    for (size_t j = 0; j < numFeatures; ++j) {
      inputData.push_back(static_cast<ThresholdType>(rand()) /
                          static_cast<ThresholdType>(RAND_MAX));
    }
  }

  size_t rowSize = inputData.size() / batchSize;
  std::vector<ReturnType> result(batchSize, -1);
  inferenceRunner.RunInference<ThresholdType, ReturnType>(inputData.data(),
                                                          result.data());
  for (int64_t rowIdx = 0; rowIdx < batchSize; ++rowIdx) {
    std::vector<double> row(inputData.begin() + rowIdx * rowSize,
                            inputData.begin() + (rowIdx + 1) * rowSize);
    ReturnType expectedResult =
        static_cast<ReturnType>(forestCreator.GetForest()->Predict(row));
    Test_ASSERT(FPEqual(result[rowIdx], expectedResult));
  }

  return true;
}

template <typename ThresholdType, typename IndexType, typename ReturnType>
bool VerifyGPUAutoScheduleCodeGeneration(
    TestArgs_t &args, const int32_t batchSize, ForestCreator &forestCreator,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation,
    int32_t tileSize, int32_t tileShapeBitwidth, int32_t childIndexBitWidth,
    TreeBeard::GPUAutoScheduleOptions gpuAutoScheduleOptions,
    const std::string &csvPath = "") {

  TreeBeard::CompilerOptions options(
      sizeof(FloatType) * 8, sizeof(FloatType) * 8, true, sizeof(IndexType) * 8,
      sizeof(IndexType) * 8, sizeof(FloatType) * 8, batchSize, tileSize,
      tileShapeBitwidth, childIndexBitWidth, TreeBeard::TilingType::kUniform,
      true, false, nullptr);

  // [HACK!] Create a shared pointer that points to forestCreator
  std::shared_ptr<ForestCreator> forestCreatorPtr(&forestCreator,
                                                  NoOpDeleter());

  TreeBeard::TreebeardContext tbContext(
      forestCreator.GetContext(), "", "", options, representation, serializer,
      forestCreatorPtr, &gpuAutoScheduleOptions);

  auto module = ConstructGPUModuleFromTreebeardContext(tbContext);
  // return true;

  GPUInferenceRunnerForTest inferenceRunner(serializer, module, tileSize,
                                            sizeof(ThresholdType) * 8,
                                            sizeof(IndexType) * 8);

  if (!csvPath.empty()) {
    bool res = ValidateModuleOutputAgainstCSVdata<ThresholdType, ReturnType>(
        inferenceRunner, csvPath, batchSize);
    // std::cout << "Kernel execution time: "
    //           << inferenceRunner.GetKernelExecutionTime() << std::endl;
    return res;
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
    std::vector<ReturnType> result(batchSize, -1);
    inferenceRunner.RunInference<ThresholdType, ReturnType>(batch.data(),
                                                            result.data());
    for (int64_t rowIdx = 0; rowIdx < batchSize; ++rowIdx) {
      std::vector<double> row(batch.begin() + rowIdx * rowSize,
                              batch.begin() + (rowIdx + 1) * rowSize);
      ReturnType expectedResult =
          static_cast<ReturnType>(forestCreator.GetForest()->Predict(row));
      Test_ASSERT(FPEqual(result[rowIdx], expectedResult));
    }
  }
  return true;
}

template <typename ThresholdType, typename IndexType,
          typename ReturnType = ThresholdType>
bool VerifyGPUCodeGenerationOutput_Tiled_VariableBatchSize_AnyRep(
    TestArgs_t &args, const int32_t batchSize, ForestCreator &forestCreator,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation,
    int32_t tileSize, int32_t tileShapeBitwidth, int32_t childIndexBitWidth = 1,
    std::function<void(decisionforest::Schedule &)> scheduleManipulator =
        std::bind(GPUBasicSchedule, std::placeholders::_1, 4),
    const std::string &csvPath = "") {
  return VerifyGPUCodeGeneration<ThresholdType, IndexType, ReturnType>(
      args, batchSize, forestCreator, serializer, representation, tileSize,
      tileShapeBitwidth, childIndexBitWidth, scheduleManipulator, csvPath);
}

// ===---------------------------------------------------=== //
// GPU Basic Scalar Code Generation Tests
// ===---------------------------------------------------=== //

template <typename ThresholdType, typename IndexType>
bool VerifyGPUCodeGenerationOutput_Scalar_VariableBatchSize_AnyRep(
    TestArgs_t &args, const int32_t batchSize, ForestCreator &forestCreator,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation,
    int32_t childIndexBitWidth = 1, const std::string &csvPath = "") {
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(GPUBasicSchedule, std::placeholders::_1, 4);
  return VerifyGPUCodeGeneration<ThresholdType, IndexType>(
      args, batchSize, forestCreator, serializer, representation,
      1 /*tileSize*/, 1 /*tileShapeBitwidth*/, childIndexBitWidth,
      scheduleManipulator, csvPath);
}

template <typename ThresholdType, typename IndexType>
bool Test_GPUCodeGeneration_Scalar_VariableBatchSize_AnyRep(
    TestArgs_t &args, const int32_t batchSize,
    ForestConstructor_t forestConstructor,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation,
    int32_t childIndexBitWidth = 1) {

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  return VerifyGPUCodeGenerationOutput_Scalar_VariableBatchSize_AnyRep<
      ThresholdType, IndexType>(args, batchSize, irConstructor, serializer,
                                representation, childIndexBitWidth);
}

template <typename ThresholdType, typename IndexType>
bool Test_GPUCodeGeneration_Scalar_VariableBatchSize(
    TestArgs_t &args, const int32_t batchSize,
    ForestConstructor_t forestConstructor, int32_t childIndexBitWidth = 1) {

  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());

  return Test_GPUCodeGeneration_Scalar_VariableBatchSize_AnyRep<ThresholdType,
                                                                IndexType>(
      args, batchSize, forestConstructor,
      decisionforest::ConstructGPUModelSerializer(modelGlobalsJSONPath),
      decisionforest::ConstructGPURepresentation(), childIndexBitWidth);
}

template <typename ThresholdType, typename ReturnType, typename IndexType>
bool Test_GPUCodeGeneration_XGBoostModel_VariableBatchSize(
    TestArgs_t &args, const int32_t batchSize,
    const std::string &xgboostModelFile, const std::string &representation,
    int32_t tileSize = 1, int32_t tileShapeBitWidth = 1,
    int32_t childIndexBitWidth = 1,
    std::function<void(decisionforest::Schedule &)> scheduleManipulator =
        std::bind(GPUBasicSchedule, std::placeholders::_1, 4)) {
  using NodeIndexType = int32_t;
  auto xgboostModelPath = GetXGBoostModelPath(xgboostModelFile);
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          xgboostModelPath);
  auto csvPath = xgboostModelPath + ".csv";

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);

  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          representation, modelGlobalsJSONPath);
  TreeBeard::XGBoostJSONParser<ThresholdType, ReturnType, NodeIndexType,
                               NodeIndexType, ThresholdType>
      xgBoostParser(context, xgboostModelPath, serializer, batchSize);

  return VerifyGPUCodeGenerationOutput_Tiled_VariableBatchSize_AnyRep<
      ThresholdType, IndexType, ReturnType>(
      args, batchSize, xgBoostParser, serializer,
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          representation),
      tileSize, tileShapeBitWidth, childIndexBitWidth, scheduleManipulator,
      csvPath);
}

template <typename ThresholdType, typename ReturnType, typename IndexType>
bool Test_GPUCodeGeneration_XGBoostModel_MultipleRepresentations(
    TestArgs_t &args, const int32_t batchSize,
    const std::string &xgboostModelFile,
    std::vector<std::string> representations, int32_t tileSize = 1,
    int32_t tileShapeBitWidth = 1, int32_t childIndexBitWidth = 1,
    std::function<void(decisionforest::Schedule &)> scheduleManipulator =
        std::bind(GPUBasicSchedule, std::placeholders::_1, 4)) {
  bool success = true;
  for (auto representation : representations) {
    success &= Test_GPUCodeGeneration_XGBoostModel_VariableBatchSize<
        ThresholdType, ReturnType, IndexType>(
        args, batchSize, xgboostModelFile, representation, tileSize,
        tileShapeBitWidth, childIndexBitWidth, scheduleManipulator);
    assert(success);
  }
  return success;
}

template <typename ThresholdType, typename IndexType>
bool Test_GPUCodeGeneration_ReorgForestRep(
    TestArgs_t &args, const int32_t batchSize,
    ForestConstructor_t forestConstructor, int32_t childIndexBitWidth = 1) {

  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());

  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_reorg", modelGlobalsJSONPath);
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_reorg");

  return Test_GPUCodeGeneration_Scalar_VariableBatchSize_AnyRep<ThresholdType,
                                                                IndexType>(
      args, batchSize, forestConstructor, serializer, representation,
      childIndexBitWidth);
}
// ===---------------------------------------------------=== //
// GPU Basic Scalar Code Generation Tests -- Array Based
// ===---------------------------------------------------=== //

bool Test_GPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(
      args, 32, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUCodeGeneration_Balanced_DoubleInt32_BatchSize32(TestArgs_t &args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(
      args, 32, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(
      args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

bool Test_GPUCodeGeneration_LeftHeavy_FloatInt16_BatchSize32(TestArgs_t &args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUCodeGeneration_RightHeavy_FloatInt16_BatchSize32(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(
      args, 32, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUCodeGeneration_Balanced_FloatInt16_BatchSize32(TestArgs_t &args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(
      args, 32, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUCodeGeneration_LeftAndRightHeavy_FloatInt16_BatchSize32(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(
      args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

// ===---------------------------------------------------=== //
// GPU Basic Scalar Code Generation Tests -- Sparse
// ===---------------------------------------------------=== //

bool Test_SparseGPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>, 32);
}

bool Test_SparseGPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(
      args, 32, AddRightHeavyTree<DoubleInt32Tile>, 32);
}

bool Test_SparseGPUCodeGeneration_Balanced_DoubleInt32_BatchSize32(
    TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(
      args, 32, AddBalancedTree<DoubleInt32Tile>, 32);
}

bool Test_SparseGPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(
      args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 32);
}

bool Test_SparseGPUCodeGeneration_LeftHeavy_FloatInt16_ChI16_BatchSize32(
    TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>, 16);
}

bool Test_SparseGPUCodeGeneration_RightHeavy_FloatInt16_ChI16_BatchSize32(
    TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(
      args, 32, AddRightHeavyTree<DoubleInt32Tile>, 16);
}

bool Test_SparseGPUCodeGeneration_Balanced_FloatInt16_ChI16_BatchSize32(
    TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(
      args, 32, AddBalancedTree<DoubleInt32Tile>, 16);
}

bool Test_SparseGPUCodeGeneration_LeftAndRightHeavy_FloatInt16_ChI16_BatchSize32(
    TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(
      args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 16);
}

// ===---------------------------------------------------=== //
// GPU Basic Scalar Code Generation Tests -- Reorg
// ===---------------------------------------------------=== //

bool Test_ReorgGPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_ReorgForestRep<double, int32_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_ReorgGPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_ReorgForestRep<double, int32_t>(
      args, 32, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_ReorgGPUCodeGeneration_Balanced_DoubleInt32_BatchSize32(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_ReorgForestRep<double, int32_t>(
      args, 32, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_ReorgGPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_ReorgForestRep<double, int32_t>(
      args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

bool Test_ReorgGPUCodeGeneration_LeftHeavy_FloatInt16_BatchSize32(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_ReorgForestRep<float, int16_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_ReorgGPUCodeGeneration_RightHeavy_FloatInt16_BatchSize32(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_ReorgForestRep<float, int16_t>(
      args, 32, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_ReorgGPUCodeGeneration_LeftAndRightHeavy_FloatInt16_BatchSize32(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_ReorgForestRep<float, int16_t>(
      args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

bool Test_ReorgGPUCodeGeneration_LeftRightAndBalanced_FloatInt16_BatchSize32(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_ReorgForestRep<float, int16_t>(
      args, 32, AddRightLeftAndBalancedTrees<DoubleInt32Tile>);
}

// ===---------------------------------------------------=== //
// GPU Basic Model Initialization Tests -- Reorg
// ===---------------------------------------------------=== //

template <typename ThresholdType, typename IndexType>
bool CheckGPUModelInitialization_ReorgForest(
    TestArgs_t &args, ForestConstructor_t forestConstructor) {
  // Batch size and the exact number of inputs per thread do not affect the
  // model initialization. So just hard coding those.
  const int32_t batchSize = 32;

  TreeBeard::GPUCompileInfo compileInfo;
  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);

  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_reorg", modelGlobalsJSONPath);

  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  irConstructor.ConstructForest();
  irConstructor.SetChildIndexBitWidth(1);
  auto module = irConstructor.GetEvaluationFunction();

  auto schedule = irConstructor.GetSchedule();
  GPUBasicSchedule(*schedule, 4);

  // module->dump();
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(context, module);
  mlir::decisionforest::lowerReductionsAndCanonicalize(context, module);

  // module->dump();

  mlir::decisionforest::GreedilyMapParallelLoopsToGPU(module);
  // module->dump();

  mlir::decisionforest::ConvertParallelLoopsToGPU(context, module);
  // module->dump();

  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_reorg");
  mlir::decisionforest::LowerGPUEnsembleToMemrefs(context, module, serializer,
                                                  representation);

  mlir::decisionforest::ConvertNodeTypeToIndexType(context, module);
  AddGPUModelMemrefGetter_Reorg(module);
  // module->dump();

  mlir::decisionforest::LowerGPUToLLVM(context, module, representation,
                                       compileInfo);
  // module->dump();

  GPUInferenceRunnerForTest inferenceRunner(
      serializer, module, 1, sizeof(ThresholdType) * 8, sizeof(IndexType) * 8);

  auto reorgSerializer =
      reinterpret_cast<decisionforest::ReorgForestSerializer *>(
          serializer.get());
  auto thresholdGPUMemref = reorgSerializer->GetThresholdMemref();
  auto featureIndexGPUMemref = reorgSerializer->GetFeatureIndexMemref();
  auto numModelElements = reorgSerializer->GetNumberOfElements();
  auto actualThresholds = reorgSerializer->GetThresholds<ThresholdType>();
  auto actualFeatureIndices =
      reorgSerializer->GetFeatureIndices<ThresholdType>();

  std::vector<ThresholdType> thresholds(numModelElements, -42);
  std::vector<IndexType> featureIndices(numModelElements, -42);

  auto thresholdMemref = VectorToMemref(thresholds);
  auto featureIndicesMemref = VectorToMemref(featureIndices);

  int32_t retVal = -1;
  // Get the threshold values from the model memref
  std::vector<void *> funcArgs = {&thresholdGPUMemref.bufferPtr,
                                  &thresholdGPUMemref.alignedPtr,
                                  &thresholdGPUMemref.offset,
                                  &thresholdGPUMemref.lengths[0],
                                  &thresholdGPUMemref.strides[0],
                                  &featureIndexGPUMemref.bufferPtr,
                                  &featureIndexGPUMemref.alignedPtr,
                                  &featureIndexGPUMemref.offset,
                                  &featureIndexGPUMemref.lengths[0],
                                  &featureIndexGPUMemref.strides[0],
                                  &thresholdMemref.bufferPtr,
                                  &thresholdMemref.alignedPtr,
                                  &thresholdMemref.offset,
                                  &thresholdMemref.lengths[0],
                                  &thresholdMemref.strides[0],
                                  &featureIndicesMemref.bufferPtr,
                                  &featureIndicesMemref.alignedPtr,
                                  &featureIndicesMemref.offset,
                                  &featureIndicesMemref.lengths[0],
                                  &featureIndicesMemref.strides[0],
                                  &retVal};

  // TODO_Ashwin : This is a HACK!!
  // std::chrono::milliseconds timespan(1);
  // std::this_thread::sleep_for(timespan);
  inferenceRunner.ExecuteFunction("GetModelValues", funcArgs);

  auto thresholdZip = llvm::zip(thresholds, actualThresholds);
  for (auto thresholdTuple : thresholdZip) {
    // Test_ASSERT(FPEqual(std::get<0>(thresholdTuple),
    // std::get<1>(thresholdTuple)));
    Test_ASSERT(std::get<0>(thresholdTuple) == std::get<1>(thresholdTuple));
  }
  auto featureIndexZip = llvm::zip(featureIndices, actualFeatureIndices);
  for (auto indexTuple : featureIndexZip) {
    Test_ASSERT(std::get<0>(indexTuple) == std::get<1>(indexTuple));
  }
  return true;
}

// ===---------------------------------------------------=== //
// XGBoost benchmark GPU Tests
// ===---------------------------------------------------=== //

template <typename ThresholdType, typename ReturnType, typename IndexType,
          int32_t tileSize>
bool Test_GPUCodeGeneration_XGBoostModel(
    TestArgs_t &args, const std::string &modelName,
    std::function<void(decisionforest::Schedule &)> scheduleManipulator =
        std::bind(GPUBasicSchedule, std::placeholders::_1, 32)) {

  int32_t tileShapeBitWidth = tileSize == 1 ? 1 : 16;
  auto representations =
      tileSize == 1
          ? std::vector<std::string>{"gpu_array", "gpu_sparse", "gpu_reorg"}
          : std::vector<std::string>{"gpu_array", "gpu_sparse"};
  return Test_GPUCodeGeneration_XGBoostModel_MultipleRepresentations<
      ThresholdType, ReturnType, IndexType>(
      args, 32, modelName, representations, tileSize, tileShapeBitWidth,
      16 /*childIndexBitWidth*/, scheduleManipulator);
}

bool Test_GPUCodeGeneration_Abalone_TileSize1_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 1>(
      args, "abalone_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Abalone_TileSize2_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 2>(
      args, "abalone_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Abalone_TileSize4_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 4>(
      args, "abalone_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Abalone_TileSize8_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 8>(
      args, "abalone_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Airline_TileSize1_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 1>(
      args, "airline_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Airline_TileSize2_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 2>(
      args, "airline_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Airline_TileSize4_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 4>(
      args, "airline_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Airline_TileSize8_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 8>(
      args, "airline_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_AirlineOHE_TileSize1_BasicSchedule(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 1>(
      args, "airline-ohe_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_AirlineOHE_TileSize2_BasicSchedule(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 2>(
      args, "airline-ohe_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_AirlineOHE_TileSize4_BasicSchedule(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 4>(
      args, "airline-ohe_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_AirlineOHE_TileSize8_BasicSchedule(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 8>(
      args, "airline-ohe_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Bosch_TileSize1_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 1>(
      args, "bosch_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Bosch_TileSize2_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 2>(
      args, "bosch_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Bosch_TileSize4_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 4>(
      args, "bosch_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Bosch_TileSize8_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 8>(
      args, "bosch_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_CovType_TileSize1_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, int8_t, int32_t, 1>(
      args, "covtype_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_CovType_TileSize2_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, int8_t, int32_t, 2>(
      args, "covtype_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_CovType_TileSize4_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, int8_t, int32_t, 4>(
      args, "covtype_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_CovType_TileSize8_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, int8_t, int32_t, 8>(
      args, "covtype_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Epsilon_TileSize1_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 1>(
      args, "epsilon_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Epsilon_TileSize2_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 2>(
      args, "epsilon_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Epsilon_TileSize4_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 4>(
      args, "epsilon_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Epsilon_TileSize8_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 8>(
      args, "epsilon_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Higgs_TileSize1_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 1>(
      args, "higgs_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Higgs_TileSize2_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 2>(
      args, "higgs_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Higgs_TileSize4_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 4>(
      args, "higgs_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Higgs_TileSize8_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 8>(
      args, "higgs_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Letters_TileSize1_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, int8_t, int32_t, 1>(
      args, "letters_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Letters_TileSize2_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, int8_t, int32_t, 2>(
      args, "letters_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Letters_TileSize4_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, int8_t, int32_t, 4>(
      args, "letters_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Letters_TileSize8_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, int8_t, int32_t, 8>(
      args, "letters_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Year_TileSize1_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 1>(
      args, "year_prediction_msd_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Year_TileSize2_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 2>(
      args, "year_prediction_msd_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Year_TileSize4_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 4>(
      args, "year_prediction_msd_xgb_model_save.json");
}

bool Test_GPUCodeGeneration_Year_TileSize8_BasicSchedule(TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel<float, float, int32_t, 8>(
      args, "year_prediction_msd_xgb_model_save.json");
}

bool Test_GPUModelInit_LeftHeavy_Reorg_DoubleInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_ReorgForest<double, int32_t>(
      args, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_RightHeavy_Reorg_DoubleInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_ReorgForest<double, int32_t>(
      args, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_Balanced_Reorg_DoubleInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_ReorgForest<double, int32_t>(
      args, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftAndRightHeavy_Reorg_DoubleInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_ReorgForest<double, int32_t>(
      args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftHeavy_Reorg_FloatInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_ReorgForest<float, int32_t>(
      args, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_RightHeavy_Reorg_FloatInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_ReorgForest<float, int32_t>(
      args, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_Balanced_Reorg_FloatInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_ReorgForest<float, int32_t>(
      args, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftAndRightHeavy_Reorg_FloatInt(TestArgs_t &args) {
  return CheckGPUModelInitialization_ReorgForest<float, int32_t>(
      args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftHeavy_Reorg_FloatInt16(TestArgs_t &args) {
  return CheckGPUModelInitialization_ReorgForest<float, int16_t>(
      args, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_RightHeavy_Reorg_FloatInt16(TestArgs_t &args) {
  return CheckGPUModelInitialization_ReorgForest<float, int16_t>(
      args, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_Balanced_Reorg_FloatInt16(TestArgs_t &args) {
  return CheckGPUModelInitialization_ReorgForest<float, int16_t>(
      args, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftAndRightHeavy_Reorg_FloatInt16(TestArgs_t &args) {
  return CheckGPUModelInitialization_ReorgForest<float, int16_t>(
      args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

// ===---------------------------------------------------=== //
// GPU Shared Memory Tests
// ===---------------------------------------------------=== //

template <typename ThresholdType, typename IndexType>
bool Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize_AnyRep(
    TestArgs_t &args, const int32_t batchSize,
    ForestConstructor_t forestConstructor,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation,
    int32_t childIndexBitWidth,
    std::function<void(decisionforest::Schedule &)> scheduleManipulator) {
  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  return VerifyGPUCodeGeneration<ThresholdType, IndexType>(
      args, batchSize, irConstructor, serializer, representation,
      1 /*tileSize*/, 1 /*tileShapeBitwidth*/, childIndexBitWidth,
      scheduleManipulator);
}

template <typename ThresholdType, typename IndexType>
bool Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize(
    TestArgs_t &args, const int32_t batchSize,
    ForestConstructor_t forestConstructor, std::string rep = "gpu_array",
    int32_t childIndexBitWidth = 1) {

  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          rep, modelGlobalsJSONPath);
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(rep);
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(TahoeSharedForestStrategy, std::placeholders::_1, 8);
  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize_AnyRep<ThresholdType,
                                                                IndexType>(
      args, batchSize, forestConstructor, serializer, representation,
      childIndexBitWidth, scheduleManipulator);
}

template <typename ThresholdType, typename IndexType>
bool Test_GPUCodeGen_InputShdMem_Scalar(TestArgs_t &args,
                                        const int32_t batchSize,
                                        ForestConstructor_t forestConstructor,
                                        int32_t childIndexBitWidth = 1) {

  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(TahoeSharedDataStrategy_Modified, std::placeholders::_1, 8);
  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize_AnyRep<ThresholdType,
                                                                IndexType>(
      args, batchSize, forestConstructor,
      decisionforest::ConstructGPUModelSerializer(modelGlobalsJSONPath),
      decisionforest::ConstructGPURepresentation(), childIndexBitWidth,
      scheduleManipulator);
}

// ===---------------------------------------------------=== //
// GPU Array Rep Shared Memory Tests
// ===---------------------------------------------------=== //

bool Test_SimpleSharedMem_LeftHeavy(TestArgs_t &args) {
  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize<double, int32_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_SimpleSharedMem_LeftRightAndBalanced(TestArgs_t &args) {
  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize<double, int32_t>(
      args, 32, AddRightLeftAndBalancedTrees<DoubleInt32Tile>);
}

bool Test_SimpleSharedMem_LeftHeavy_F32I16(TestArgs_t &args) {
  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize<float, int16_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_SimpleSharedMem_LeftRightAndBalanced_F32I16(TestArgs_t &args) {
  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize<float, int16_t>(
      args, 32, AddRightLeftAndBalancedTrees<DoubleInt32Tile>);
}

// ===---------------------------------------------------=== //
// GPU Input Shared Memory Tests
// ===---------------------------------------------------=== //

bool Test_InputSharedMem_LeftHeavy(TestArgs_t &args) {
  return Test_GPUCodeGen_InputShdMem_Scalar<double, int32_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_InputSharedMem_RightHeavy(TestArgs_t &args) {
  return Test_GPUCodeGen_InputShdMem_Scalar<double, int32_t>(
      args, 32, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_InputSharedMem_LeftRightAndBalanced(TestArgs_t &args) {
  return Test_GPUCodeGen_InputShdMem_Scalar<double, int32_t>(
      args, 32, AddRightLeftAndBalancedTrees<DoubleInt32Tile>);
}

// ===---------------------------------------------------=== //
// GPU Multi-class Tests
// ===---------------------------------------------------=== //

bool Test_GPUCodeGeneration_Covtype_ArrayRep_DoubleInt32_BatchSize32(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel_VariableBatchSize<float, int8_t,
                                                               int32_t>(
      args, 32, "covtype_xgb_model_save.json", "gpu_array");
}

bool Test_GPUCodeGeneration_Covtype_SparseRep_DoubleInt32_BatchSize32(
    TestArgs_t &args) {
  mlir::decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_XGBoostModel_VariableBatchSize<float, int8_t,
                                                               int32_t>(
      args, 32, "covtype_xgb_model_save.json", "gpu_sparse", 16);
}

bool Test_GPUCodeGeneration_Covtype_ReorgRep_DoubleInt32_BatchSize32(
    TestArgs_t &args) {
  return Test_GPUCodeGeneration_XGBoostModel_VariableBatchSize<float, int8_t,
                                                               int32_t>(
      args, 32, "covtype_xgb_model_save.json", "gpu_reorg");
}

bool Test_GPUCodeGeneration_Covtype_SparseRep_f32i16_B32_iterativeCachedPartialForestStrategy_NoCache(
    TestArgs_t &args) {
  using ThresholdType = float;
  using IndexType = int16_t;
  using ReturnType = int8_t;

  int32_t batchSize = 64;
  int32_t numRowsPerBlock = 4, numTreesPerIter = 10;
  const std::string xgboostModelFile = "covtype_xgb_model_save.json";
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse");

  auto xgboostModelPath = GetXGBoostModelPath(xgboostModelFile);
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          xgboostModelPath);
  auto csvPath = xgboostModelPath + ".csv";

  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath);

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);

  TreeBeard::XGBoostJSONParser<ThresholdType, ReturnType, IndexType, IndexType,
                               ThresholdType>
      xgBoostParser(context, xgboostModelPath, serializer, batchSize);

  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::iterativeCachedPartialForestStrategy_NoCache,
                std::placeholders::_1, numTreesPerIter, numRowsPerBlock);
  return VerifyGPUCodeGeneration<ThresholdType, IndexType, ReturnType>(
      args, batchSize, xgBoostParser, serializer, representation,
      1,  // Tile size
      16, // Tile shape width
      16, // child index width
      scheduleManipulator, csvPath);

  //   return Test_GPUCodeGeneration_XGBoostModel_VariableBatchSize<float,
  //   int8_t,
  //                                                                int32_t>(
  //       args, 32, "covtype_xgb_model_save.json", "gpu_sparse", 16);
}

bool Test_GPUCodeGeneration_Covtype_SparseRep_f32i16_B32_iterativeCachedPartialForestStrategy_NoCache_SharedReduce(
    TestArgs_t &args) {
  using ThresholdType = float;
  using IndexType = int16_t;
  using ReturnType = int8_t;

  int32_t batchSize = 64;
  int32_t numRowsPerBlock = 4, numTreesPerIter = 10;
  const std::string xgboostModelFile = "covtype_xgb_model_save.json";
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse");

  auto xgboostModelPath = GetXGBoostModelPath(xgboostModelFile);
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          xgboostModelPath);
  auto csvPath = xgboostModelPath + ".csv";

  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath);

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);

  TreeBeard::XGBoostJSONParser<ThresholdType, ReturnType, IndexType, IndexType,
                               ThresholdType>
      xgBoostParser(context, xgboostModelPath, serializer, batchSize);

  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::
                    iterativeCachedPartialForestStrategy_NoCache_SharedReduce,
                std::placeholders::_1, numTreesPerIter, numRowsPerBlock);
  return VerifyGPUCodeGeneration<ThresholdType, IndexType, ReturnType>(
      args, batchSize, xgBoostParser, serializer, representation,
      1,  // Tile size
      16, // Tile shape width
      16, // child index width
      scheduleManipulator, csvPath);

  //   return Test_GPUCodeGeneration_XGBoostModel_VariableBatchSize<float,
  //   int8_t,
  //                                                                int32_t>(
  //       args, 32, "covtype_xgb_model_save.json", "gpu_sparse", 16);
}

// ===---------------------------------------------------=== //
// GPU Reorg Rep Shared Memory Tests
// ===---------------------------------------------------=== //

bool Test_SimpleSharedMem_LeftHeavy_ReorgRep(TestArgs_t &args) {
  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize<double, int32_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>, "gpu_reorg");
}

bool Test_SimpleSharedMem_LeftRightAndBalanced_Reorg(TestArgs_t &args) {
  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize<double, int32_t>(
      args, 32, AddRightLeftAndBalancedTrees<DoubleInt32Tile>, "gpu_reorg");
}

bool Test_SimpleSharedMem_LeftHeavy_ReorgRep_F32I16(TestArgs_t &args) {
  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize<float, int16_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>, "gpu_reorg");
}

bool Test_SimpleSharedMem_LeftRightAndBalanced_Reorg_F32I16(TestArgs_t &args) {
  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize<float, int16_t>(
      args, 32, AddRightLeftAndBalancedTrees<DoubleInt32Tile>, "gpu_reorg");
}
// ===---------------------------------------------------=== //
// GPU Sparse Rep Shared Memory Tests
// ===---------------------------------------------------=== //

bool Test_SimpleSharedMem_LeftHeavy_SparseRep(TestArgs_t &args) {
  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize<double, int32_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>, "gpu_sparse", 32);
}

bool Test_SimpleSharedMem_LeftRightAndBalanced_SparseRep(TestArgs_t &args) {
  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize<double, int32_t>(
      args, 32, AddRightLeftAndBalancedTrees<DoubleInt32Tile>, "gpu_sparse",
      32);
}

bool Test_SimpleSharedMem_LeftHeavy_SparseRep_F32I16(TestArgs_t &args) {
  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize<float, int16_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>, "gpu_sparse", 16);
}

bool Test_SimpleSharedMem_LeftRightAndBalanced_SparseRep_F32I16(
    TestArgs_t &args) {
  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize<float, int16_t>(
      args, 32, AddRightLeftAndBalancedTrees<DoubleInt32Tile>, "gpu_sparse",
      16);
}

// ===---------------------------------------------------=== //
// GPU Basic Tiled Code Generation Tests
// ===---------------------------------------------------=== //

template <typename ThresholdType, typename IndexType>
bool Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep(
    TestArgs_t &args, const int32_t batchSize,
    ForestConstructor_t forestConstructor,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation,
    int32_t tileSize, int32_t tileShapeBitWidth,
    int32_t childIndexBitWidth = 1) {

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  return VerifyGPUCodeGenerationOutput_Tiled_VariableBatchSize_AnyRep<
      ThresholdType, IndexType>(args, batchSize, irConstructor, serializer,
                                representation, tileSize, tileShapeBitWidth,
                                childIndexBitWidth);
}

// ===---------------------------------------------------=== //
// GPU Sparse Representation - Basic Tiled Code Generation Tests
// ===---------------------------------------------------=== //

bool Test_TiledSparseGPU_LeftHeavy_DblI32_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<double,
                                                                  int32_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      32,  // Tile shape width
      32); // child index width
}

bool Test_TiledSparseGPU_RightHeavy_DblI32_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<double,
                                                                  int32_t>(
      args, 32, AddRightHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      32,  // Tile shape width
      32); // child index width
}

bool Test_TiledSparseGPU_Balanced_DblI32_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<double,
                                                                  int32_t>(
      args, 32, AddBalancedTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      32,  // Tile shape width
      32); // child index width
}

bool Test_TiledSparseGPU_LeftAndRightHeavy_DblI32_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<double,
                                                                  int32_t>(
      args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      32,  // Tile shape width
      32); // child index width
}

bool Test_TiledSparseGPU_LeftHeavy_FltI16_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<float,
                                                                  int16_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      16,  // Tile shape width
      16); // child index width
}

bool Test_TiledSparseGPU_RightHeavy_FltI16_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<float,
                                                                  int16_t>(
      args, 32, AddRightHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      16,  // Tile shape width
      16); // child index width
}

bool Test_TiledSparseGPU_Balanced_FltI16_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<float,
                                                                  int16_t>(
      args, 32, AddBalancedTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      16,  // Tile shape width
      16); // child index width
}

bool Test_TiledSparseGPU_LeftAndRightHeavy_FltI16_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<float,
                                                                  int16_t>(
      args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      16,  // Tile shape width
      16); // child index width
}

// ===---------------------------------------------------=== //
// GPU Array Representation -- Basic Tiled Code Generation Tests
// ===---------------------------------------------------=== //

bool Test_TiledArrayGPU_LeftHeavy_DblI32_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<double,
                                                                  int32_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      32,  // Tile shape width
      32); // child index width
}

bool Test_TiledArrayGPU_RightHeavy_DblI32_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<double,
                                                                  int32_t>(
      args, 32, AddRightHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      32,  // Tile shape width
      32); // child index width
}

bool Test_TiledArrayGPU_Balanced_DblI32_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<double,
                                                                  int32_t>(
      args, 32, AddBalancedTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      32,  // Tile shape width
      32); // child index width
}

bool Test_TiledArrayGPU_LeftAndRightHeavy_DblI32_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<double,
                                                                  int32_t>(
      args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      32,  // Tile shape width
      32); // child index width
}

bool Test_TiledArrayGPU_LeftHeavy_FltI16_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<float,
                                                                  int16_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      16,  // Tile shape width
      16); // child index width
}

bool Test_TiledArrayGPU_RightHeavy_FltI16_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<float,
                                                                  int16_t>(
      args, 32, AddRightHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      16,  // Tile shape width
      16); // child index width
}

bool Test_TiledArrayGPU_Balanced_FltI16_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<float,
                                                                  int16_t>(
      args, 32, AddBalancedTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      16,  // Tile shape width
      16); // child index width
}

bool Test_TiledArrayGPU_LeftAndRightHeavy_FltI16_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_Tiled_VariableBatchSize_AnyRep<float,
                                                                  int16_t>(
      args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      16,  // Tile shape width
      16); // child index width
}

// ===---------------------------------------------------=== //
// GPU Tiling + Caching Tests
// ===---------------------------------------------------=== //

template <typename ThresholdType, typename IndexType>
bool VerifyGPUCodeGen_TiledAndCached_AnyRep(
    TestArgs_t &args, const int32_t batchSize, ForestCreator &forestCreator,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation,
    int32_t tileSize, int32_t tileShapeBitwidth, int32_t childIndexBitWidth = 1,
    const std::string &csvPath = "") {
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(TahoeSharedForestStrategy, std::placeholders::_1, 8);
  return VerifyGPUCodeGeneration<ThresholdType, IndexType>(
      args, batchSize, forestCreator, serializer, representation, tileSize,
      tileShapeBitwidth, childIndexBitWidth, scheduleManipulator, csvPath);
}

template <typename ThresholdType, typename IndexType>
bool Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep(
    TestArgs_t &args, const int32_t batchSize,
    ForestConstructor_t forestConstructor,
    std::shared_ptr<decisionforest::IModelSerializer> serializer,
    std::shared_ptr<decisionforest::IRepresentation> representation,
    int32_t tileSize, int32_t tileShapeBitWidth,
    int32_t childIndexBitWidth = 1) {

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  return VerifyGPUCodeGen_TiledAndCached_AnyRep<ThresholdType, IndexType>(
      args, batchSize, irConstructor, serializer, representation, tileSize,
      tileShapeBitWidth, childIndexBitWidth);
}

// ===---------------------------------------------------=== //
// GPU Array Representation - Tiling + Caching Code Generation Tests
// ===---------------------------------------------------=== //

bool Test_TiledCachedArrayGPU_LeftHeavy_DblI32_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<double, int32_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      32); // Tile shape width
}

bool Test_TiledCachedArrayGPU_RightHeavy_DblI32_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<double, int32_t>(
      args, 32, AddRightHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      32); // Tile shape width
}

bool Test_TiledCachedArrayGPU_Balanced_DblI32_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<double, int32_t>(
      args, 32, AddBalancedTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      32); // Tile shape width
}

bool Test_TiledCachedArrayGPU_LeftAndRightHeavy_DblI32_B32_TSz2(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<double, int32_t>(
      args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      32); // Tile shape width
}

bool Test_TiledCachedArrayGPU_LeftRightAndBalanced_DblI32_B32_TSz2(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<double, int32_t>(
      args, 32, AddRightLeftAndBalancedTrees<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      32); // Tile shape width
}

bool Test_TiledCachedArrayGPU_LeftHeavy_FltI16_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<float, int16_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      16); // Tile shape width
}

bool Test_TiledCachedArrayGPU_RightHeavy_FltI16_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<float, int16_t>(
      args, 32, AddRightHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      16); // Tile shape width
}

bool Test_TiledCachedArrayGPU_Balanced_FltI16_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<float, int16_t>(
      args, 32, AddBalancedTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      16); // Tile shape width
}

bool Test_TiledCachedArrayGPU_LeftAndRightHeavy_FltI16_B32_TSz2(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<float, int16_t>(
      args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      16); // Tile shape width
}

bool Test_TiledCachedArrayGPU_LeftRightAndBalanced_FltI16_B32_TSz2(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<float, int16_t>(
      args, 32, AddRightLeftAndBalancedTrees<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_array", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_array"),
      2,   // Tile size
      16); // Tile shape width
}

// ===---------------------------------------------------=== //
// GPU Sparse Representation - Tiling + Caching Code Generation Tests
// ===---------------------------------------------------=== //

bool Test_TiledCachedSparseGPU_LeftHeavy_DblI32_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<double, int32_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      32,  // Tile shape width
      32); // child index width
}

bool Test_TiledCachedSparseGPU_RightHeavy_DblI32_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<double, int32_t>(
      args, 32, AddRightHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      32,  // Tile shape width
      32); // child index width
}

bool Test_TiledCachedSparseGPU_Balanced_DblI32_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<double, int32_t>(
      args, 32, AddBalancedTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      32,  // Tile shape width
      32); // child index width
}

bool Test_TiledCachedSparseGPU_LeftAndRightHeavy_DblI32_B32_TSz2(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<double, int32_t>(
      args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      32,  // Tile shape width
      32); // child index width
}

bool Test_TiledCachedSparseGPU_LeftRightAndBalanced_DblI32_B32_TSz2(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<double, int32_t>(
      args, 32, AddRightLeftAndBalancedTrees<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      32,  // Tile shape width
      32); // child index width
}

bool Test_TiledCachedSparseGPU_LeftHeavy_FltI16_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<float, int16_t>(
      args, 32, AddLeftHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      16,  // Tile shape width
      16); // child index width
}

bool Test_TiledCachedSparseGPU_RightHeavy_FltI16_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<float, int16_t>(
      args, 32, AddRightHeavyTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      16,  // Tile shape width
      16); // child index width
}

bool Test_TiledCachedSparseGPU_Balanced_FltI16_B32_TSz2(TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<float, int16_t>(
      args, 32, AddBalancedTree<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      16,  // Tile shape width
      16); // child index width
}

bool Test_TiledCachedSparseGPU_LeftAndRightHeavy_FltI16_B32_TSz2(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<float, int16_t>(
      args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      16,  // Tile shape width
      16); // child index width
}

bool Test_TiledCachedSparseGPU_LeftRightAndBalanced_FltI16_B32_TSz2(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  return Test_GPU_FixedConstructor_SharedMemAndTiled_AnyRep<float, int16_t>(
      args, 32, AddRightLeftAndBalancedTrees<DoubleInt32Tile>,
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath),
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse"),
      2,   // Tile size
      16,  // Tile shape width
      16); // child index width
}

// ===---------------------------------------------------=== //
// GPU Sparse Representation - Tahoe shared input strategy
// ===---------------------------------------------------=== //
bool Test_ScalarSparseGPU_LeftRightAndBalanced_TahoeShdInp_FltI16_B32(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  using ThresholdType = float;
  using IndexType = int16_t;

  int32_t batchSize = 32;
  ForestConstructor_t forestConstructor =
      AddRightLeftAndBalancedTrees<DoubleInt32Tile>;
  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath);
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse");
  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      decisionforest::TahoeSharedDataStrategy;
  return VerifyGPUCodeGeneration<ThresholdType, IndexType>(
      args, batchSize, irConstructor, serializer, representation,
      1,  // Tile size
      16, // Tile shape width
      16, // child index width
      scheduleManipulator);
}

// ===---------------------------------------------------=== //
// GPU Sparse Representation - Tahoe shared input multi-row strategy
// ===---------------------------------------------------=== //
bool Test_ScalarSparseGPU_LeftRightAndBalanced_TahoeShdInpMultiRow_FltI16_B32(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  using ThresholdType = float;
  using IndexType = int16_t;

  int32_t batchSize = 64;
  int32_t numRowsPerBlock = 4;
  ForestConstructor_t forestConstructor =
      AddRightLeftAndBalancedTrees<DoubleInt32Tile>;
  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath);
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse");
  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::tahoeSharedDataStrategy_MultipleRowsPerBlock,
                std::placeholders::_1, numRowsPerBlock);
  return VerifyGPUCodeGeneration<ThresholdType, IndexType>(
      args, batchSize, irConstructor, serializer, representation,
      1,  // Tile size
      16, // Tile shape width
      16, // child index width
      scheduleManipulator);
}

bool Test_ScalarSparseGPU_TwiceLeftRightBalanced_TahoeShdInpMultiRow_FltI16_B32(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  using ThresholdType = float;
  using IndexType = int16_t;

  int32_t batchSize = 64;
  int32_t numRowsPerBlock = 4;
  ForestConstructor_t forestConstructor =
      AddRightLeftAndBalancedTreesTwice<DoubleInt32Tile>;
  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath);
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse");
  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::tahoeSharedDataStrategy_MultipleRowsPerBlock,
                std::placeholders::_1, numRowsPerBlock);
  return VerifyGPUCodeGeneration<ThresholdType, IndexType>(
      args, batchSize, irConstructor, serializer, representation,
      1,  // Tile size
      16, // Tile shape width
      16, // child index width
      scheduleManipulator);
}

bool Test_ScalarSparseGPU_TwiceLeftRightBalanced_IterShdPartialForest_FltI16_B32(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  using ThresholdType = float;
  using IndexType = int16_t;

  int32_t batchSize = 64;
  int32_t numRowsPerBlock = 4, numTreesPerIter = 3;
  ForestConstructor_t forestConstructor =
      AddRightLeftAndBalancedTreesTwice<DoubleInt32Tile>;
  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath);
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse");
  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::iterativeCachedPartialForestStrategy,
                std::placeholders::_1, numTreesPerIter, numRowsPerBlock);
  return VerifyGPUCodeGeneration<ThresholdType, IndexType>(
      args, batchSize, irConstructor, serializer, representation,
      1,  // Tile size
      16, // Tile shape width
      16, // child index width
      scheduleManipulator);
}

// ===------------------------------------------------------------=== //
// GPU Sparse Representation -
// iterativeCachedPartialForestStrategy_NoCache_SharedReduce strategy
// ===------------------------------------------------------------=== //
bool Test_ScalarSparseGPU_TwiceLeftRightBalanced_iterCachedPartialForest_NoCache_SharedReduce_FltI16_B64(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  using ThresholdType = float;
  using IndexType = int16_t;

  int32_t batchSize = 64;
  int32_t numRowsPerBlock = 4, numTreesPerIter = 3;
  ForestConstructor_t forestConstructor =
      AddRightLeftAndBalancedTreesTwice<DoubleInt32Tile>;
  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath);
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse");
  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::
                    iterativeCachedPartialForestStrategy_NoCache_SharedReduce,
                std::placeholders::_1, numTreesPerIter, numRowsPerBlock);
  return VerifyGPUCodeGeneration<ThresholdType, IndexType>(
      args, batchSize, irConstructor, serializer, representation,
      1,  // Tile size
      16, // Tile shape width
      16, // child index width
      scheduleManipulator);
}

// ===------------------------------------------------------------=== //
// GPU Sparse Representation -
// iterativeCachedPartialForestStrategy_NoCache_SpecializedTreeLoop strategy
// ===------------------------------------------------------------=== //
bool Test_ScalarSparseGPU_TwiceLeftRightBalanced_iterCachedPartialForest_NoCache_SpecializedTreeLoop_FltI16_B64(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  using ThresholdType = float;
  using IndexType = int16_t;

  int32_t batchSize = 64;
  int32_t numRowsPerBlock = 4, numTreesPerIter = 3;
  ForestConstructor_t forestConstructor =
      AddRightLeftAndBalancedTreesTwice<DoubleInt32Tile>;
  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath);
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse");
  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(
          decisionforest::
              iterativeCachedPartialForestStrategy_NoCache_SpecializeTreeLoop,
          std::placeholders::_1, numTreesPerIter, numRowsPerBlock);
  return VerifyGPUCodeGeneration<ThresholdType, IndexType>(
      args, batchSize, irConstructor, serializer, representation,
      1,  // Tile size
      16, // Tile shape width
      16, // child index width
      scheduleManipulator);
}

// ===------------------------------------------------------------=== //
// GPU Sparse Representation -
// GPU Auto schedule
// ===------------------------------------------------------------=== //

template <typename ThresholdType, typename IndexType, typename ReturnType>
bool Test_ScalarGPU_XGBoost_AutoScheduleBasic(
    TestArgs_t &args, const std::string &xgboostModelFile) {
  auto xgboostModelPath = GetXGBoostModelPath(xgboostModelFile);
  auto csvPath = xgboostModelPath + ".csv";
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          xgboostModelPath);

  int32_t batchSize = 256;

  // Maps to TB.x
  int32_t numRowsPerTB = 64;
  int32_t numRowsPerThread = 8;
  // Hpw many rows do we process together?
  // Pick one tree and process this many rows before moving to the next row
  int32_t rowTileSize = -1;

  // How many threads do we divide the trees across?
  // Maps to TB.y
  int32_t numTreeThreads = 2;

  // How many trees do we process at a time? Only useful if single threaded
  // TODO Do we really need this? Can't we always do one tree at a time
  int32_t numTreesAtATime = 1;
  bool cacheRows = false;
  bool cacheTrees = false;
  bool unrollTreeWalks = false;

  TreeBeard::GPUAutoScheduleOptions gpuAutoScheduleOptions{
      numRowsPerTB,    numRowsPerThread, rowTileSize, numTreeThreads,
      numTreesAtATime, cacheRows,        cacheTrees,  unrollTreeWalks};

  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath);
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse");

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  TreeBeard::XGBoostJSONParser<ThresholdType, ReturnType, IndexType, IndexType,
                               ThresholdType>
      xgBoostParser(context, xgboostModelPath, serializer, batchSize);
  return VerifyGPUAutoScheduleCodeGeneration<ThresholdType, IndexType,
                                             ReturnType>(
      args, batchSize, xgBoostParser, serializer, representation,
      1,                     // Tile size
      16,                    // Tile shape width
      sizeof(IndexType) * 8, // child index width
      gpuAutoScheduleOptions, csvPath);
}

bool Test_ScalarGPU_Abalone_AutoScheduleBasic(TestArgs_t &args) {
  // mlir::decisionforest::measureGpuKernelTime = true;
  return Test_ScalarGPU_XGBoost_AutoScheduleBasic<float, int32_t, float>(
      args, "abalone_xgb_model_save.json");
}

bool Test_ScalarGPU_Airline_AutoScheduleBasic(TestArgs_t &args) {
  return Test_ScalarGPU_XGBoost_AutoScheduleBasic<float, int32_t, float>(
      args, "airline_xgb_model_save.json");
}

bool Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleBasic(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  using ThresholdType = float;
  using IndexType = int16_t;

  int32_t batchSize = 256;

  // Maps to TB.x
  int32_t numRowsPerTB = 64;
  int32_t numRowsPerThread = 8;
  // Hpw many rows do we process together?
  // Pick one tree and process this many rows before moving to the next row
  int32_t rowTileSize = -1;

  // How many threads do we divide the trees across?
  // Maps to TB.y
  int32_t numTreeThreads = 2;

  // How many trees do we process at a time? Only useful if single threaded
  // TODO Do we really need this? Can't we always do one tree at a time
  int32_t numTreesAtATime = 1;
  bool cacheRows = false;
  bool cacheTrees = false;
  bool unrollTreeWalks = true;

  TreeBeard::GPUAutoScheduleOptions gpuAutoScheduleOptions{
      numRowsPerTB,    numRowsPerThread, rowTileSize, numTreeThreads,
      numTreesAtATime, cacheRows,        cacheTrees,  unrollTreeWalks};

  ForestConstructor_t forestConstructor =
      AddRightLeftAndBalancedTreesTwice<DoubleInt32Tile>;
  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath);
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse");
  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  return VerifyGPUAutoScheduleCodeGeneration<ThresholdType, IndexType,
                                             ThresholdType>(
      args, batchSize, irConstructor, serializer, representation,
      1,  // Tile size
      16, // Tile shape width
      16, // child index width
      gpuAutoScheduleOptions);
}

bool Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleCachedRows(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  using ThresholdType = float;
  using IndexType = int16_t;

  int32_t batchSize = 256;

  // Maps to TB.x
  int32_t numRowsPerTB = 64;
  int32_t numRowsPerThread = 8;
  // Hpw many rows do we process together?
  // Pick one tree and process this many rows before moving to the next row
  int32_t rowTileSize = -1;

  // How many threads do we divide the trees across?
  // Maps to TB.y
  int32_t numTreeThreads = 2;

  // How many trees do we process at a time? Only useful if single threaded
  // TODO Do we really need this? Can't we always do one tree at a time
  int32_t numTreesAtATime = 1;
  bool cacheRows = true;
  bool cacheTrees = false;
  bool unrollTreeWalks = true;

  TreeBeard::GPUAutoScheduleOptions gpuAutoScheduleOptions{
      numRowsPerTB,    numRowsPerThread, rowTileSize, numTreeThreads,
      numTreesAtATime, cacheRows,        cacheTrees,  unrollTreeWalks};

  ForestConstructor_t forestConstructor =
      AddRightLeftAndBalancedTreesTwice<DoubleInt32Tile>;
  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath);
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse");
  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  return VerifyGPUAutoScheduleCodeGeneration<ThresholdType, IndexType,
                                             ThresholdType>(
      args, batchSize, irConstructor, serializer, representation,
      1,  // Tile size
      16, // Tile shape width
      16, // child index width
      gpuAutoScheduleOptions);
}

bool Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleCachedTrees(
    TestArgs_t &args) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  using ThresholdType = float;
  using IndexType = int16_t;

  int32_t batchSize = 256;

  // Maps to TB.x
  int32_t numRowsPerTB = 64;
  int32_t numRowsPerThread = 8;
  // Hpw many rows do we process together?
  // Pick one tree and process this many rows before moving to the next row
  int32_t rowTileSize = -1;

  // How many threads do we divide the trees across?
  // Maps to TB.y
  int32_t numTreeThreads = 2;

  // How many trees do we process at a time? Only useful if single threaded
  // TODO Do we really need this? Can't we always do one tree at a time
  int32_t numTreesAtATime = 1;
  bool cacheRows = false;
  bool cacheTrees = true;
  bool unrollTreeWalks = false;

  TreeBeard::GPUAutoScheduleOptions gpuAutoScheduleOptions{
      numRowsPerTB,    numRowsPerThread, rowTileSize, numTreeThreads,
      numTreesAtATime, cacheRows,        cacheTrees,  unrollTreeWalks};

  ForestConstructor_t forestConstructor =
      AddRightLeftAndBalancedTreesTwice<DoubleInt32Tile>;
  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath);
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse");
  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType,
                         ThresholdType>
      irConstructor(context, serializer, batchSize, forestConstructor);
  return VerifyGPUAutoScheduleCodeGeneration<ThresholdType, IndexType,
                                             ThresholdType>(
      args, batchSize, irConstructor, serializer, representation,
      1,  // Tile size
      16, // Tile shape width
      16, // child index width
      gpuAutoScheduleOptions);
}

bool Test_GPUCodeGeneration_Covtype_SparseRep_f32i16_B512_AutoSched_SharedReduce(
    TestArgs_t &args) {
  using ThresholdType = float;
  using IndexType = int16_t;
  using ReturnType = int8_t;

  int32_t batchSize = 512;

  // Maps to TB.x
  int32_t numRowsPerTB = 16;
  int32_t numRowsPerThread = 2;
  // Hpw many rows do we process together?
  // Pick one tree and process this many rows before moving to the next row
  int32_t rowTileSize = -1;

  // How many threads do we divide the trees across?
  // Maps to TB.y
  int32_t numTreeThreads = 50;

  // How many trees do we process at a time? Only useful if single threaded
  // TODO Do we really need this? Can't we always do one tree at a time
  int32_t numTreesAtATime = 1;
  bool cacheRows = false;
  bool cacheTrees = false;
  bool unrollTreeWalks = true;
  bool useSharedReduce = true;

  TreeBeard::GPUAutoScheduleOptions gpuAutoScheduleOptions{
      numRowsPerTB,   numRowsPerThread, rowTileSize,
      numTreeThreads, numTreesAtATime,  cacheRows,
      cacheTrees,     unrollTreeWalks,  -1,
      useSharedReduce};

  const std::string xgboostModelFile = "covtype_xgb_model_save.json";
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse");

  auto xgboostModelPath = GetXGBoostModelPath(xgboostModelFile);
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          xgboostModelPath);
  auto csvPath = xgboostModelPath + ".csv";

  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath);

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);

  TreeBeard::XGBoostJSONParser<ThresholdType, ReturnType, IndexType, IndexType,
                               ThresholdType>
      xgBoostParser(context, xgboostModelPath, serializer, batchSize);

  return VerifyGPUAutoScheduleCodeGeneration<ThresholdType, IndexType,
                                             ReturnType>(
      args, batchSize, xgBoostParser, serializer, representation,
      1,  // Tile size
      16, // Tile shape width
      16, // child index width
      gpuAutoScheduleOptions, csvPath);
}

bool Test_GPUCodeGeneration_Letters_SparseRep_f32i16_B512_AutoSched_SharedReduce(
    TestArgs_t &args) {
  using ThresholdType = float;
  using IndexType = int16_t;
  using ReturnType = int8_t;

  int32_t batchSize = 512;

  // Maps to TB.x
  int32_t numRowsPerTB = 16;
  int32_t numRowsPerThread = 1;
  // Hpw many rows do we process together?
  // Pick one tree and process this many rows before moving to the next row
  int32_t rowTileSize = -1;

  // How many threads do we divide the trees across?
  // Maps to TB.y
  int32_t numTreeThreads = 20;

  // How many trees do we process at a time? Only useful if single threaded
  // TODO Do we really need this? Can't we always do one tree at a time
  int32_t numTreesAtATime = 1;
  bool cacheRows = false;
  bool cacheTrees = false;
  bool unrollTreeWalks = true;
  bool useSharedReduce = true;

  TreeBeard::GPUAutoScheduleOptions gpuAutoScheduleOptions{
      numRowsPerTB,   numRowsPerThread, rowTileSize,
      numTreeThreads, numTreesAtATime,  cacheRows,
      cacheTrees,     unrollTreeWalks,  -1,
      useSharedReduce};

  const std::string xgboostModelFile = "letters_xgb_model_save.json";
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse");

  auto xgboostModelPath = GetXGBoostModelPath(xgboostModelFile);
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          xgboostModelPath);
  auto csvPath = xgboostModelPath + ".test.sampled.csv";

  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath);

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);

  TreeBeard::XGBoostJSONParser<ThresholdType, ReturnType, IndexType, IndexType,
                               ThresholdType>
      xgBoostParser(context, xgboostModelPath, serializer, batchSize);

  return VerifyGPUAutoScheduleCodeGeneration<ThresholdType, IndexType,
                                             ReturnType>(
      args, batchSize, xgBoostParser, serializer, representation,
      1,  // Tile size
      16, // Tile shape width
      16, // child index width
      gpuAutoScheduleOptions, csvPath);
}

// ===------------------------------------------------------------=== //
// GPU Shared Reduce Tests - Non-Multiclass
// ===------------------------------------------------------------=== //
bool Test_GPUCodeGeneration_Abalone_SparseRep_f32i16_B32_iterativeCachedPartialForestStrategy_NoCache_SharedReduce(
    TestArgs_t &args) {
  using ThresholdType = float;
  using IndexType = int16_t;
  using ReturnType = float;

  int32_t batchSize = 64;
  int32_t numRowsPerBlock = 4, numTreesPerIter = 10;
  const std::string xgboostModelFile = "abalone_xgb_model_save.json";
  auto representation =
      decisionforest::RepresentationFactory::Get().GetRepresentation(
          "gpu_sparse");

  auto xgboostModelPath = GetXGBoostModelPath(xgboostModelFile);
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          xgboostModelPath);
  auto csvPath = xgboostModelPath + ".csv";

  auto serializer =
      decisionforest::ModelSerializerFactory::Get().GetModelSerializer(
          "gpu_sparse", modelGlobalsJSONPath);

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);

  TreeBeard::XGBoostJSONParser<ThresholdType, ReturnType, IndexType, IndexType,
                               ThresholdType>
      xgBoostParser(context, xgboostModelPath, serializer, batchSize);

  std::function<void(decisionforest::Schedule &)> scheduleManipulator =
      std::bind(decisionforest::
                    iterativeCachedPartialForestStrategy_NoCache_SharedReduce,
                std::placeholders::_1, numTreesPerIter, numRowsPerBlock);
  return VerifyGPUCodeGeneration<ThresholdType, IndexType, ReturnType>(
      args, batchSize, xgBoostParser, serializer, representation,
      1,  // Tile size
      16, // Tile shape width
      16, // child index width
      scheduleManipulator, csvPath);
}

} // namespace test
} // namespace TreeBeard

#endif // #ifdef TREEBEARD_GPU_SUPPORT