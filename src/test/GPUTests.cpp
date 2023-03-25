#include <vector>
#include <sstream>
#include <chrono>
#include <thread>

#include "TreeTilingUtils.h"
#include "ExecutionHelpers.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "xgboostparser.h"
#include "TiledTree.h"

#include "TestUtilsCommon.h"
#include "ForestTestUtils.h"
#include "ModelSerializers.h"
#include "Representations.h"
#include "GPUSupportUtils.h"
#include "GPUExecutionHelper.h"
#include "GPUModelSerializers.h"
#include "ReorgForestRepresentation.h"

using namespace mlir;

namespace TreeBeard
{
namespace test
{

class GPUInferenceRunnerForTest : public InferenceRunnerForTestTemplate<decisionforest::GPUInferenceRunner> {
public:
  using InferenceRunnerForTestTemplate<decisionforest::GPUInferenceRunner>::InferenceRunnerForTestTemplate;
  decisionforest::ModelMemrefType GetModelMemref() { 
    return reinterpret_cast<decisionforest::GPUArraySparseSerializerBase*>(m_serializer.get())->GetModelMemref();
  }
};

// ===---------------------------------------------------=== //
// GPU Model Initialization Test Helpers
// ===---------------------------------------------------=== //

void AddGPUModelMemrefGetter_Scalar(mlir::ModuleOp module) {
  // Get the tiled node type from the model memref type by finding the Init_Model method
  MemRefType modelMemrefType;
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getName() == "Init_Model")
      modelMemrefType = func.getResultTypes()[0].cast<mlir::MemRefType>();
  });
  decisionforest::TiledNumericalNodeType tileType = modelMemrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  auto thresholdMemrefType = MemRefType::get(modelMemrefType.getShape()[0], tileType.getThresholdElementType());
  auto featureIndexMemrefType = MemRefType::get(modelMemrefType.getShape()[0], tileType.getIndexElementType());
  auto getModelFunctionType = FunctionType::get(tileType.getContext(), 
                                                TypeRange{modelMemrefType, thresholdMemrefType, featureIndexMemrefType},
                                                TypeRange{IntegerType::get(tileType.getContext(), 32)});

  auto location = module.getLoc();
  auto getModelFunc = func::FuncOp::create(location, "GetModelValues", getModelFunctionType);
  getModelFunc.setPublic();

  auto entryBlock = getModelFunc.addEntryBlock();
  mlir::OpBuilder builder(getModelFunc.getContext());
  builder.setInsertionPointToStart(entryBlock);

  // 1. Allocate the threshold and feature index buffers on the GPU
  // 2. Launch kernel to copy model values to these buffers
  // 3. Copy results in to the argument memrefs

  auto waitOp = builder.create<gpu::WaitOp>(location, gpu::AsyncTokenType::get(module.getContext()), ValueRange{});
  auto allocThreshold = builder.create<gpu::AllocOp>(location, thresholdMemrefType, 
                                                     waitOp.getAsyncToken().getType(),
                                                     ValueRange{waitOp.getAsyncToken()}, 
                                                     ValueRange{}, ValueRange{});
  auto allocIndices = builder.create<gpu::AllocOp>(location, featureIndexMemrefType, 
                                                   waitOp.getAsyncToken().getType(),
                                                   ValueRange{allocThreshold.getAsyncToken()}, 
                                                   ValueRange{}, ValueRange{});                                                     

  auto oneIndexConst = builder.create<arith::ConstantIndexOp>(location, 1);
  int32_t numThreadsPerBlock = 32;
  int32_t numBlocks = std::ceil((double)modelMemrefType.getShape()[0]/numThreadsPerBlock);
  auto numThreadBlocksConst = builder.create<arith::ConstantIndexOp>(location, numBlocks);
  auto numThreadsPerBlockConst = builder.create<arith::ConstantIndexOp>(location, numThreadsPerBlock);
  auto gpuLaunch = builder.create<gpu::LaunchOp>(location, numThreadBlocksConst, oneIndexConst, oneIndexConst, 
                                                numThreadsPerBlockConst, oneIndexConst, oneIndexConst,
                                                nullptr, waitOp.getAsyncToken().getType(), 
                                                allocIndices.getAsyncToken());

  builder.setInsertionPointToStart(&gpuLaunch.getBody().front());
  
  // // Generate the body of the launch op
  auto memrefLengthConst = builder.create<arith::ConstantIndexOp>(location, modelMemrefType.getShape()[0]);
  auto firstThreadNum = builder.create<arith::MulIOp>(location, gpuLaunch.getBlockSizeX(), gpuLaunch.getBlockIds().x);
  auto elementIndex = builder.create<arith::AddIOp>(location, firstThreadNum, gpuLaunch.getThreadIds().x);
  auto inBoundsCondition = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::slt, elementIndex, memrefLengthConst);
  auto ifInBounds = builder.create<scf::IfOp>(location, inBoundsCondition, false);
  {
    // Generate the initialization code
    auto thenBuilder = ifInBounds.getThenBodyBuilder();
    auto loadThreshold = thenBuilder.create<decisionforest::LoadTileThresholdsOp>(location, tileType.getThresholdElementType(), 
                                                                                  getModelFunc.getArgument(0), elementIndex, oneIndexConst);
    auto loadIndex = thenBuilder.create<decisionforest::LoadTileFeatureIndicesOp>(location, tileType.getIndexElementType(), 
                                                                                  getModelFunc.getArgument(0), elementIndex, oneIndexConst);
    /*auto writeThreshold =*/ thenBuilder.create<memref::StoreOp>(location, static_cast<Value>(loadThreshold),
                                                                  allocThreshold.getMemref(), static_cast<Value>(elementIndex));
    /*auto writeIndex =*/ thenBuilder.create<memref::StoreOp>(location, static_cast<Value>(loadIndex), 
                                                              allocIndices.getMemref(), static_cast<Value>(elementIndex));
  }
  builder.create<gpu::TerminatorOp>(location);
  builder.setInsertionPointAfter(gpuLaunch); 

  // Transfer back the offsets and the thresholds
  auto transferThresholds = builder.create<gpu::MemcpyOp>(location, gpuLaunch.getAsyncToken().getType(),  
                                                          ValueRange{gpuLaunch.getAsyncToken()}, getModelFunc.getArgument(1), 
                                                          allocThreshold.getMemref());

  auto transferIndices = builder.create<gpu::MemcpyOp>(location, transferThresholds.getAsyncToken().getType(), 
                                                       ValueRange{transferThresholds.getAsyncToken()}, getModelFunc.getArgument(2), 
                                                       allocIndices.getMemref());

  // Wait and return 
  /*auto waitBeforeReturn =*/ builder.create<gpu::WaitOp>(location, transferIndices.getAsyncToken().getType(), transferIndices.getAsyncToken());
  auto returnVal = builder.create<arith::ConstantIntOp>(location, 42 /*value*/, 32 /*width*/);
  builder.create<func::ReturnOp>(location, static_cast<Value>(returnVal));

  module.push_back(getModelFunc);
}

void AddGPUModelMemrefGetter_Reorg(mlir::ModuleOp module) {
  // Get the tiled node type from the model memref type by finding the Init_Model method
  MemRefType modelMemrefType, featureIndexMemrefType;
  module.walk([&](mlir::func::FuncOp func) {
    if (func.getName() == "Init_Thresholds")
      modelMemrefType = func.getResultTypes()[0].cast<mlir::MemRefType>();
    if (func.getName() == "Init_FeatureIndices")
      featureIndexMemrefType = func.getResultTypes()[0].cast<mlir::MemRefType>();
  });
  auto getModelFunctionType = FunctionType::get(featureIndexMemrefType.getContext(), 
                                                TypeRange{modelMemrefType, featureIndexMemrefType, modelMemrefType, featureIndexMemrefType},
                                                TypeRange{IntegerType::get(featureIndexMemrefType.getContext(), 32)});

  auto location = module.getLoc();
  auto getModelFunc = func::FuncOp::create(location, "GetModelValues", getModelFunctionType);
  getModelFunc.setPublic();

  auto entryBlock = getModelFunc.addEntryBlock();
  mlir::OpBuilder builder(getModelFunc.getContext());
  builder.setInsertionPointToStart(entryBlock);

  auto waitOp = builder.create<gpu::WaitOp>(location, gpu::AsyncTokenType::get(module.getContext()), ValueRange{});

  // auto oneIndexConst = builder.create<arith::ConstantIndexOp>(location, 1);
  // auto numThreadBlocksConst = builder.create<arith::ConstantIndexOp>(location, 1);
  // auto numThreadsPerBlockConst = builder.create<arith::ConstantIndexOp>(location, 7);
  // auto gpuLaunch = builder.create<gpu::LaunchOp>(location, numThreadBlocksConst, 
  //                                               oneIndexConst, oneIndexConst, 
  //                                               numThreadsPerBlockConst, oneIndexConst, oneIndexConst,
  //                                               nullptr, waitOp.getAsyncToken().getType(), 
  //                                               waitOp.getAsyncToken());
  
  // builder.setInsertionPointToStart(&gpuLaunch.getBody().front());
  // auto thresholdVal = builder.create<memref::LoadOp>(location, getModelFunc.getArgument(0), ValueRange{gpuLaunch.getThreadIds().x});
  // auto indexVal = builder.create<memref::LoadOp>(location, getModelFunc.getArgument(1), ValueRange{gpuLaunch.getThreadIds().x});
  // // builder.create<gpu::PrintfOp>(location, "Threshold[%ld]: %lf\tIndices[%ld]: %d\n", ValueRange{ gpuLaunch.getThreadIds().x, static_cast<Value>(thresholdVal), 
  // //                                                                                            gpuLaunch.getThreadIds().x, static_cast<Value>(indexVal)});
  // builder.create<gpu::TerminatorOp>(location);
  // // Wait and return 
  // builder.setInsertionPointAfter(gpuLaunch); 

  // Transfer back the offsets and the thresholds
  auto transferThresholds = builder.create<gpu::MemcpyOp>(location, waitOp.getAsyncToken().getType(),  
                                                          ValueRange{waitOp.getAsyncToken()}, getModelFunc.getArgument(2), 
                                                          getModelFunc.getArgument(0));

  auto transferIndices = builder.create<gpu::MemcpyOp>(location, transferThresholds.getAsyncToken().getType(), 
                                                       ValueRange{transferThresholds.getAsyncToken()}, getModelFunc.getArgument(3), 
                                                       getModelFunc.getArgument(1));

  // Wait and return 
  /*auto waitBeforeReturn =*/ builder.create<gpu::WaitOp>(location, transferIndices.getAsyncToken().getType(), transferIndices.getAsyncToken());
  auto returnVal = builder.create<arith::ConstantIntOp>(location, 42 /*value*/, 32 /*width*/);
  builder.create<func::ReturnOp>(location, static_cast<Value>(returnVal));

  module.push_back(getModelFunc);
}

// ===---------------------------------------------------=== //
// GPU Model Initialization Tests
// ===---------------------------------------------------=== //

void GPUBasicSchedule(decisionforest::Schedule* schedule, int32_t gridXSize) {
  auto& batchIndex = schedule->GetBatchIndex();
  auto& blockIndex = schedule->NewIndexVariable("gridX");
  auto& threadIndex = schedule->NewIndexVariable("blockX");
  
  schedule->Tile(batchIndex, blockIndex, threadIndex, gridXSize);
  blockIndex.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid, decisionforest::IndexVariable::Dimension::X);
  threadIndex.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock, decisionforest::IndexVariable::Dimension::X);
}

template <typename ThresholdType, typename IndexType>
bool CheckGPUModelInitialization_Scalar(TestArgs_t& args, ForestConstructor_t forestConstructor) {
  // Batch size and the exact number of inputs per thread do not affect the model 
  // initialization. So just hard coding those.
  const int32_t batchSize = 32;

  auto modelGlobalsJSONPath = TreeBeard::ModelJSONParser<ThresholdType,
                                                         ThresholdType,
                                                         IndexType,
                                                         IndexType,
                                                         ThresholdType>::ModelGlobalJSONFilePathFromJSONFilePath(TreeBeard::test::GetGlobalJSONNameForTests());
  auto serializer = decisionforest::ConstructGPUModelSerializer(modelGlobalsJSONPath);
  
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType, ThresholdType> 
                         irConstructor(args.context, serializer, batchSize, forestConstructor);
  irConstructor.Parse();
  irConstructor.SetChildIndexBitWidth(1);
  auto module = irConstructor.GetEvaluationFunction();

  auto schedule = irConstructor.GetSchedule();
  GPUBasicSchedule(schedule, 4);

  // module->dump();
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(args.context, module);
  // module->dump();

  mlir::decisionforest::GreedilyMapParallelLoopsToGPU(module);
  // module->dump();

  mlir::decisionforest::ConvertParallelLoopsToGPU(args.context, module);
  // module->dump();

  auto representation = decisionforest::ConstructGPURepresentation();
  mlir::decisionforest::LowerEnsembleToMemrefs(args.context, 
                                               module,
                                               serializer,
                                               representation);
  
  mlir::decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  AddGPUModelMemrefGetter_Scalar(module);
  // module->dump();

  mlir::decisionforest::LowerGPUToLLVM(args.context, module, representation);
  // module->dump();

  GPUInferenceRunnerForTest inferenceRunner(serializer, 
                                            module,
                                            1,
                                            sizeof(ThresholdType)*8,
                                            sizeof(IndexType)*8);
  
  // TODO this is a hack. We are kind of breaking the abstraction to get hold of
  // information that would otherwise be hard to get.
  auto numModelElements = decisionforest::ForestJSONReader::GetInstance().GetTotalNumberOfTiles();
  std::vector<ThresholdType> actualThresholds;
  std::vector<IndexType> actualFeatureIndices, actualTileShapes;
  decisionforest::ForestJSONReader::GetInstance().GetModelValues(1 /*tileSize*/,
                                                                 sizeof(ThresholdType)*8,
                                                                 sizeof(IndexType)*8,
                                                                 actualThresholds,
                                                                 actualFeatureIndices,
                                                                 actualTileShapes);

  std::vector<ThresholdType> thresholds(numModelElements, -42);
  std::vector<IndexType> featureIndices(numModelElements, -42);

  auto thresholdMemref = VectorToMemref(thresholds);
  auto featureIndicesMemref = VectorToMemref(featureIndices);
  auto modelMemref = inferenceRunner.GetModelMemref();

  int32_t retVal = -1;
  // Get the threshold values from the model memref
  std::vector<void*> funcArgs = { &modelMemref.bufferPtr, &modelMemref.alignedPtr, &modelMemref.offset, &modelMemref.lengths[0], &modelMemref.strides[0],
                              &thresholdMemref.bufferPtr, &thresholdMemref.alignedPtr, &thresholdMemref.offset, &thresholdMemref.lengths[0], &thresholdMemref.strides[0],
                              &featureIndicesMemref.bufferPtr, &featureIndicesMemref.alignedPtr, &featureIndicesMemref.offset, &featureIndicesMemref.lengths[0], &featureIndicesMemref.strides[0],
                              &retVal };

  inferenceRunner.ExecuteFunction("GetModelValues", funcArgs);

  auto thresholdZip = llvm::zip(thresholds, actualThresholds);
  for (auto thresholdTuple: thresholdZip) {
    // Test_ASSERT(FPEqual(std::get<0>(thresholdTuple), std::get<1>(thresholdTuple)));
    Test_ASSERT(std::get<0>(thresholdTuple) == std::get<1>(thresholdTuple));
  }
  auto featureIndexZip = llvm::zip(featureIndices, actualFeatureIndices);
  for (auto indexTuple : featureIndexZip) {
    Test_ASSERT(std::get<0>(indexTuple) == std::get<1>(indexTuple));
  }
  return true;
}

bool Test_GPUModelInit_LeftHeavy_Scalar_DoubleInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_Scalar<double, int32_t>(args, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_RightHeavy_Scalar_DoubleInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_Scalar<double, int32_t>(args, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_Balanced_Scalar_DoubleInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_Scalar<double, int32_t>(args, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftAndRightHeavy_Scalar_DoubleInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_Scalar<double, int32_t>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftHeavy_Scalar_FloatInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_Scalar<float, int32_t>(args, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_RightHeavy_Scalar_FloatInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_Scalar<float, int32_t>(args, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_Balanced_Scalar_FloatInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_Scalar<float, int32_t>(args, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftAndRightHeavy_Scalar_FloatInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_Scalar<float, int32_t>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftHeavy_Scalar_FloatInt16(TestArgs_t& args) {
  return CheckGPUModelInitialization_Scalar<float, int16_t>(args, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_RightHeavy_Scalar_FloatInt16(TestArgs_t& args) {
  return CheckGPUModelInitialization_Scalar<float, int16_t>(args, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_Balanced_Scalar_FloatInt16(TestArgs_t& args) {
  return CheckGPUModelInitialization_Scalar<float, int16_t>(args, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftAndRightHeavy_Scalar_FloatInt16(TestArgs_t& args) {
  return CheckGPUModelInitialization_Scalar<float, int16_t>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

// ===---------------------------------------------------=== //
// GPU Basic Scalar Code Generation Tests
// ===---------------------------------------------------=== //

template<typename ThresholdType, typename IndexType>
bool Test_GPUCodeGeneration_Scalar_VariableBatchSize_AnyRep(TestArgs_t& args, 
                                                            const int32_t batchSize,
                                                            ForestConstructor_t forestConstructor,
                                                            std::shared_ptr<decisionforest::IModelSerializer> serializer,
                                                            std::shared_ptr<decisionforest::IRepresentation> representation,
                                                            int32_t childIndexBitWidth=1) {
                                                    
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType, ThresholdType> 
                         irConstructor(args.context, serializer, batchSize, forestConstructor);
  irConstructor.Parse();
  
  // If sparse representation is turned on, then child index bit width should be passed
  assert (!mlir::decisionforest::UseSparseTreeRepresentation || childIndexBitWidth!=1 );
  irConstructor.SetChildIndexBitWidth(childIndexBitWidth);
  
  auto module = irConstructor.GetEvaluationFunction();

  auto schedule = irConstructor.GetSchedule();
  GPUBasicSchedule(schedule, 4);

  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(args.context, module);
  // module->dump();

  mlir::decisionforest::GreedilyMapParallelLoopsToGPU(module);
  // module->dump();

  mlir::decisionforest::ConvertParallelLoopsToGPU(args.context, module);
  // module->dump();

  mlir::decisionforest::LowerEnsembleToMemrefs(args.context,
                                               module,
                                               serializer,
                                               representation);
  
  mlir::decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  // module->dump();

  mlir::decisionforest::LowerGPUToLLVM(args.context, module, representation);
  // module->dump();

  GPUInferenceRunnerForTest inferenceRunner(serializer,
                                            module,
                                            1 /*tileSize*/, 
                                            sizeof(ThresholdType)*8, sizeof(IndexType)*8);

  assert (batchSize%2 == 0);
  std::vector<std::vector<ThresholdType>> inputData;
  inputData.emplace_back(std::vector<ThresholdType>());
  auto& firstVec = inputData.front();
  for (int32_t i=0 ; i<batchSize/2 ; ++i) {
    auto data=GetBatchSize2Data();
    firstVec.insert(firstVec.end(), data.front().begin(), data.front().end());
  }
  for(auto& batch : inputData) {
    assert (batch.size() % batchSize == 0);
    size_t rowSize = batch.size()/batchSize;
    std::vector<ThresholdType> result(batchSize, -1);
    inferenceRunner.RunInference<ThresholdType, ThresholdType>(batch.data(), result.data());
    for(int64_t rowIdx=0 ; rowIdx<batchSize ; ++rowIdx) {
      std::vector<double> row(batch.begin() + rowIdx*rowSize, batch.begin() + (rowIdx+1)*rowSize);
      ThresholdType expectedResult = static_cast<ThresholdType>(irConstructor.GetForest().Predict(row));
      Test_ASSERT(FPEqual(result[rowIdx], expectedResult));
    }
  }
  return true;
}

template<typename ThresholdType, typename IndexType>
bool Test_GPUCodeGeneration_Scalar_VariableBatchSize(TestArgs_t& args, 
                                                     const int32_t batchSize,
                                                     ForestConstructor_t forestConstructor,
                                                     int32_t childIndexBitWidth=1) {

  auto modelGlobalsJSONPath = TreeBeard::ModelJSONParser<ThresholdType,
                                                         ThresholdType,
                                                         IndexType,
                                                         IndexType,
                                                         ThresholdType>::ModelGlobalJSONFilePathFromJSONFilePath(TreeBeard::test::GetGlobalJSONNameForTests());

  return Test_GPUCodeGeneration_Scalar_VariableBatchSize_AnyRep<ThresholdType, IndexType>(args, 
                                                                                          batchSize,
                                                                                          forestConstructor,
                                                                                          decisionforest::ConstructGPUModelSerializer(modelGlobalsJSONPath),
                                                                                          decisionforest::ConstructGPURepresentation(),
                                                                                          childIndexBitWidth);
}

template<typename ThresholdType, typename IndexType>
bool Test_GPUCodeGeneration_ReorgForestRep(TestArgs_t& args, 
                                           const int32_t batchSize,
                                           ForestConstructor_t forestConstructor,
                                           int32_t childIndexBitWidth=1) {

  auto modelGlobalsJSONPath = TreeBeard::ModelJSONParser<ThresholdType,
                                                         ThresholdType,
                                                         IndexType,
                                                         IndexType,
                                                         ThresholdType>::ModelGlobalJSONFilePathFromJSONFilePath(TreeBeard::test::GetGlobalJSONNameForTests());

  auto serializer = decisionforest::ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg", modelGlobalsJSONPath);
  auto representation = decisionforest::RepresentationFactory::Get().GetRepresentation("gpu_reorg");

  return Test_GPUCodeGeneration_Scalar_VariableBatchSize_AnyRep<ThresholdType, IndexType>(args, 
                                                                                          batchSize,
                                                                                          forestConstructor,
                                                                                          serializer,
                                                                                          representation,
                                                                                          childIndexBitWidth);
}
// ===---------------------------------------------------=== //
// GPU Basic Scalar Code Generation Tests -- Array Based
// ===---------------------------------------------------=== //

bool Test_GPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(args, 32, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(args, 32, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUCodeGeneration_Balanced_DoubleInt32_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(args, 32, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

bool Test_GPUCodeGeneration_LeftHeavy_FloatInt16_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(args, 32, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUCodeGeneration_RightHeavy_FloatInt16_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(args, 32, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUCodeGeneration_Balanced_FloatInt16_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(args, 32, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUCodeGeneration_LeftAndRightHeavy_FloatInt16_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

// ===---------------------------------------------------=== //
// GPU Basic Scalar Code Generation Tests -- Sparse
// ===---------------------------------------------------=== //

bool Test_SparseGPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(args, 32, AddLeftHeavyTree<DoubleInt32Tile>, 32);
}

bool Test_SparseGPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(args, 32, AddRightHeavyTree<DoubleInt32Tile>, 32);
}

bool Test_SparseGPUCodeGeneration_Balanced_DoubleInt32_BatchSize32(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(args, 32, AddBalancedTree<DoubleInt32Tile>, 32);
}

bool Test_SparseGPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<double, int32_t>(args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 32);
}

bool Test_SparseGPUCodeGeneration_LeftHeavy_FloatInt16_ChI16_BatchSize32(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(args, 32, AddLeftHeavyTree<DoubleInt32Tile>, 16);
}

bool Test_SparseGPUCodeGeneration_RightHeavy_FloatInt16_ChI16_BatchSize32(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(args, 32, AddRightHeavyTree<DoubleInt32Tile>, 16);
}

bool Test_SparseGPUCodeGeneration_Balanced_FloatInt16_ChI16_BatchSize32(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(args, 32, AddBalancedTree<DoubleInt32Tile>, 16);
}

bool Test_SparseGPUCodeGeneration_LeftAndRightHeavy_FloatInt16_ChI16_BatchSize32(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  return Test_GPUCodeGeneration_Scalar_VariableBatchSize<float, int16_t>(args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 16);
}

// ===---------------------------------------------------=== //
// GPU Basic Scalar Code Generation Tests -- Reorg
// ===---------------------------------------------------=== //

bool Test_ReorgGPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_ReorgForestRep<double, int32_t>(args, 32, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_ReorgGPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_ReorgForestRep<double, int32_t>(args, 32, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_ReorgGPUCodeGeneration_Balanced_DoubleInt32_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_ReorgForestRep<double, int32_t>(args, 32, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_ReorgGPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_ReorgForestRep<double, int32_t>(args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

bool Test_ReorgGPUCodeGeneration_LeftHeavy_FloatInt16_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_ReorgForestRep<float, int16_t>(args, 32, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_ReorgGPUCodeGeneration_RightHeavy_FloatInt16_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_ReorgForestRep<float, int16_t>(args, 32, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_ReorgGPUCodeGeneration_LeftAndRightHeavy_FloatInt16_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_ReorgForestRep<float, int16_t>(args, 32, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

bool Test_ReorgGPUCodeGeneration_LeftRightAndBalanced_FloatInt16_BatchSize32(TestArgs_t& args) {
  return Test_GPUCodeGeneration_ReorgForestRep<float, int16_t>(args, 32, AddRightLeftAndBalancedTrees<DoubleInt32Tile>);
}

// ===---------------------------------------------------=== //
// GPU Basic Model Initialization Tests -- Reorg
// ===---------------------------------------------------=== //

template <typename ThresholdType, typename IndexType>
bool CheckGPUModelInitialization_ReorgForest(TestArgs_t& args, ForestConstructor_t forestConstructor) {
  // Batch size and the exact number of inputs per thread do not affect the model 
  // initialization. So just hard coding those.
  const int32_t batchSize = 32;

  auto modelGlobalsJSONPath = TreeBeard::ModelJSONParser<ThresholdType,
                                                         ThresholdType,
                                                         IndexType,
                                                         IndexType,
                                                         ThresholdType>::ModelGlobalJSONFilePathFromJSONFilePath(TreeBeard::test::GetGlobalJSONNameForTests());
  auto serializer = decisionforest::ModelSerializerFactory::Get().GetModelSerializer("gpu_reorg", modelGlobalsJSONPath);
  
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType, ThresholdType> 
                         irConstructor(args.context, serializer, batchSize, forestConstructor);
  irConstructor.Parse();
  irConstructor.SetChildIndexBitWidth(1);
  auto module = irConstructor.GetEvaluationFunction();

  auto schedule = irConstructor.GetSchedule();
  GPUBasicSchedule(schedule, 4);

  // module->dump();
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(args.context, module);
  // module->dump();

  mlir::decisionforest::GreedilyMapParallelLoopsToGPU(module);
  // module->dump();

  mlir::decisionforest::ConvertParallelLoopsToGPU(args.context, module);
  // module->dump();

  auto representation = decisionforest::RepresentationFactory::Get().GetRepresentation("gpu_reorg");
  mlir::decisionforest::LowerEnsembleToMemrefs(args.context, 
                                               module,
                                               serializer,
                                               representation);
  
  mlir::decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  AddGPUModelMemrefGetter_Reorg(module);
  // module->dump();

  mlir::decisionforest::LowerGPUToLLVM(args.context, module, representation);
  // module->dump();

  GPUInferenceRunnerForTest inferenceRunner(serializer, 
                                            module,
                                            1,
                                            sizeof(ThresholdType)*8,
                                            sizeof(IndexType)*8);
  
  auto reorgSerializer = reinterpret_cast<decisionforest::ReorgForestSerializer*>(serializer.get());
  auto thresholdGPUMemref = reorgSerializer->GetThresholdMemref();
  auto featureIndexGPUMemref = reorgSerializer->GetFeatureIndexMemref();
  auto numModelElements = reorgSerializer->GetNumberOfElements();
  auto actualThresholds = reorgSerializer->GetThresholds<ThresholdType>();
  auto actualFeatureIndices = reorgSerializer->GetFeatureIndices<ThresholdType>();

  std::vector<ThresholdType> thresholds(numModelElements, -42);
  std::vector<IndexType> featureIndices(numModelElements, -42);

  auto thresholdMemref = VectorToMemref(thresholds);
  auto featureIndicesMemref = VectorToMemref(featureIndices);

  int32_t retVal = -1;
  // Get the threshold values from the model memref
  std::vector<void*> funcArgs = { &thresholdGPUMemref.bufferPtr, &thresholdGPUMemref.alignedPtr, &thresholdGPUMemref.offset, &thresholdGPUMemref.lengths[0], &thresholdGPUMemref.strides[0],
                              &featureIndexGPUMemref.bufferPtr, &featureIndexGPUMemref.alignedPtr, &featureIndexGPUMemref.offset, &featureIndexGPUMemref.lengths[0], &featureIndexGPUMemref.strides[0],
                              &thresholdMemref.bufferPtr, &thresholdMemref.alignedPtr, &thresholdMemref.offset, &thresholdMemref.lengths[0], &thresholdMemref.strides[0],
                              &featureIndicesMemref.bufferPtr, &featureIndicesMemref.alignedPtr, &featureIndicesMemref.offset, &featureIndicesMemref.lengths[0], &featureIndicesMemref.strides[0],
                              &retVal };

  // TODO_Ashwin : This is a HACK!!
  std::chrono::milliseconds timespan(1);
  std::this_thread::sleep_for(timespan);
  inferenceRunner.ExecuteFunction("GetModelValues", funcArgs);

  auto thresholdZip = llvm::zip(thresholds, actualThresholds);
  for (auto thresholdTuple: thresholdZip) {
    // Test_ASSERT(FPEqual(std::get<0>(thresholdTuple), std::get<1>(thresholdTuple)));
    Test_ASSERT(std::get<0>(thresholdTuple) == std::get<1>(thresholdTuple));
  }
  auto featureIndexZip = llvm::zip(featureIndices, actualFeatureIndices);
  for (auto indexTuple : featureIndexZip) {
    Test_ASSERT(std::get<0>(indexTuple) == std::get<1>(indexTuple));
  }
  return true;
}

bool Test_GPUModelInit_LeftHeavy_Reorg_DoubleInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_ReorgForest<double, int32_t>(args, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_RightHeavy_Reorg_DoubleInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_ReorgForest<double, int32_t>(args, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_Balanced_Reorg_DoubleInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_ReorgForest<double, int32_t>(args, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftAndRightHeavy_Reorg_DoubleInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_ReorgForest<double, int32_t>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftHeavy_Reorg_FloatInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_ReorgForest<float, int32_t>(args, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_RightHeavy_Reorg_FloatInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_ReorgForest<float, int32_t>(args, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_Balanced_Reorg_FloatInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_ReorgForest<float, int32_t>(args, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftAndRightHeavy_Reorg_FloatInt(TestArgs_t& args) {
  return CheckGPUModelInitialization_ReorgForest<float, int32_t>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftHeavy_Reorg_FloatInt16(TestArgs_t& args) {
  return CheckGPUModelInitialization_ReorgForest<float, int16_t>(args, AddLeftHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_RightHeavy_Reorg_FloatInt16(TestArgs_t& args) {
  return CheckGPUModelInitialization_ReorgForest<float, int16_t>(args, AddRightHeavyTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_Balanced_Reorg_FloatInt16(TestArgs_t& args) {
  return CheckGPUModelInitialization_ReorgForest<float, int16_t>(args, AddBalancedTree<DoubleInt32Tile>);
}

bool Test_GPUModelInit_LeftAndRightHeavy_Reorg_FloatInt16(TestArgs_t& args) {
  return CheckGPUModelInitialization_ReorgForest<float, int16_t>(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>);
}

void TahoeSharedForestStrategy(decisionforest::Schedule& schedule, int32_t rowsPerThreadBlock) {
  auto& batchIndex = schedule.GetBatchIndex();
  auto& treeIndex = schedule.GetTreeIndex();
  
  auto& b0 = schedule.NewIndexVariable("b0");
  auto& b1 = schedule.NewIndexVariable("b1");

  auto& t0 = schedule.NewIndexVariable("t0");
  auto& t1 = schedule.NewIndexVariable("t1");
  
  schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);
  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid, decisionforest::IndexVariable::Dimension::X);
  b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock, decisionforest::IndexVariable::Dimension::X);
  
  schedule.Tile(treeIndex, t0, t1, schedule.GetForestSize());
  schedule.Cache(t0);
}

void TahoeSharedPartialForestStrategy(decisionforest::Schedule& schedule,
                                      int32_t treesPerThreadBlock,
                                      int32_t rowsPerThreadBlock) {
  auto& batchIndex = schedule.GetBatchIndex();
  auto& treeIndex = schedule.GetTreeIndex();
  
  auto& b0 = schedule.NewIndexVariable("b0");
  auto& b1 = schedule.NewIndexVariable("b1");

  auto& t0 = schedule.NewIndexVariable("t0");
  auto& t0Inner = schedule.NewIndexVariable("t0Inner");
  auto& t1 = schedule.NewIndexVariable("t1");
  auto& t2 = schedule.NewIndexVariable("t2");
  
  schedule.Tile(batchIndex, b0, b1, rowsPerThreadBlock);
  
  schedule.Tile(treeIndex, t0, t1, treesPerThreadBlock);
  schedule.Tile(t0Inner, t1, t2, treesPerThreadBlock);
  schedule.Cache(t1);
  schedule.Reorder(std::vector<decisionforest::IndexVariable*>{&b0, &t0, &b1, &t1, &t2});

  b0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid, decisionforest::IndexVariable::Dimension::X);
  t0.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::Grid, decisionforest::IndexVariable::Dimension::Y);
  b1.SetGPUDimension(decisionforest::IndexVariable::GPUConstruct::ThreadBlock, decisionforest::IndexVariable::Dimension::X);
}

template<typename ThresholdType, typename IndexType>
bool Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize_AnyRep(TestArgs_t& args, 
                                                            const int32_t batchSize,
                                                            ForestConstructor_t forestConstructor,
                                                            std::shared_ptr<decisionforest::IModelSerializer> serializer,
                                                            std::shared_ptr<decisionforest::IRepresentation> representation,
                                                            int32_t childIndexBitWidth=1) {
                                                    
  FixedTreeIRConstructor<ThresholdType, ThresholdType, IndexType, IndexType, ThresholdType> 
                         irConstructor(args.context, serializer, batchSize, forestConstructor);
  irConstructor.Parse();
  
  // If sparse representation is turned on, then child index bit width should be passed
  assert (!mlir::decisionforest::UseSparseTreeRepresentation || childIndexBitWidth!=1 );
  irConstructor.SetChildIndexBitWidth(childIndexBitWidth);
  
  auto module = irConstructor.GetEvaluationFunction();

  auto schedule = irConstructor.GetSchedule();
  TahoeSharedForestStrategy(*schedule, 4);

  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(args.context, module);
  // module->dump();

  mlir::decisionforest::GreedilyMapParallelLoopsToGPU(module);
  // // module->dump();

  mlir::decisionforest::ConvertParallelLoopsToGPU(args.context, module);
  // module->dump();

  mlir::decisionforest::LowerEnsembleToMemrefs(args.context,
                                               module,
                                               serializer,
                                               representation);
  // module->dump();
  
  mlir::decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  module->dump();

  // mlir::decisionforest::LowerGPUToLLVM(args.context, module, representation);
  // // module->dump();

  // GPUInferenceRunnerForTest inferenceRunner(serializer,
  //                                           module,
  //                                           1 /*tileSize*/, 
  //                                           sizeof(ThresholdType)*8, sizeof(IndexType)*8);

  // assert (batchSize%2 == 0);
  // std::vector<std::vector<ThresholdType>> inputData;
  // inputData.emplace_back(std::vector<ThresholdType>());
  // auto& firstVec = inputData.front();
  // for (int32_t i=0 ; i<batchSize/2 ; ++i) {
  //   auto data=GetBatchSize2Data();
  //   firstVec.insert(firstVec.end(), data.front().begin(), data.front().end());
  // }
  // for(auto& batch : inputData) {
  //   assert (batch.size() % batchSize == 0);
  //   size_t rowSize = batch.size()/batchSize;
  //   std::vector<ThresholdType> result(batchSize, -1);
  //   inferenceRunner.RunInference<ThresholdType, ThresholdType>(batch.data(), result.data());
  //   for(int64_t rowIdx=0 ; rowIdx<batchSize ; ++rowIdx) {
  //     std::vector<double> row(batch.begin() + rowIdx*rowSize, batch.begin() + (rowIdx+1)*rowSize);
  //     ThresholdType expectedResult = static_cast<ThresholdType>(irConstructor.GetForest().Predict(row));
  //     Test_ASSERT(FPEqual(result[rowIdx], expectedResult));
  //   }
  // }
  return true;
}

template<typename ThresholdType, typename IndexType>
bool Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize(TestArgs_t& args, 
                                                     const int32_t batchSize,
                                                     ForestConstructor_t forestConstructor,
                                                     int32_t childIndexBitWidth=1) {

  auto modelGlobalsJSONPath = TreeBeard::ModelJSONParser<ThresholdType,
                                                         ThresholdType,
                                                         IndexType,
                                                         IndexType,
                                                         ThresholdType>::ModelGlobalJSONFilePathFromJSONFilePath(TreeBeard::test::GetGlobalJSONNameForTests());

  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize_AnyRep<ThresholdType, IndexType>(args, 
                                                                                          batchSize,
                                                                                          forestConstructor,
                                                                                          decisionforest::ConstructGPUModelSerializer(modelGlobalsJSONPath),
                                                                                          decisionforest::ConstructGPURepresentation(),
                                                                                          childIndexBitWidth);
}

bool Test_SimpleSharedMem_LeftHeavy(TestArgs_t& args) {
  return Test_GPUCodeGen_ShdMem_Scalar_VariableBatchSize<double, int32_t>(args, 32, AddLeftHeavyTree<DoubleInt32Tile>);
}

}
}