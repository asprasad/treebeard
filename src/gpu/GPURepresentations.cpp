#include "Dialect.h"
#include "OpLoweringUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "GPURepresentations.h"
#include "LIRLoweringHelpers.h"

using namespace mlir;
using namespace mlir::decisionforest::helpers;

namespace mlir
{
namespace decisionforest
{
// ===---------------------------------------------------=== //
// Helpers
// ===---------------------------------------------------=== //

void GenerateSimpleInitializer(const std::string& funcName, ConversionPatternRewriter &rewriter, Location location, 
                               ModuleOp module, MemRefType memrefType) {
  // TODO why did this not work when I used the rewriter instead of the builder?
  // auto insertPoint = rewriter.saveInsertionPoint();
  auto functionType = FunctionType::get(rewriter.getContext(), {memrefType}, {memrefType});
  NamedAttribute visibilityAttribute{module.getSymVisibilityAttrName(), rewriter.getStringAttr("public")};
  auto initFunc = func::FuncOp::create(location, funcName, functionType, ArrayRef<NamedAttribute>(visibilityAttribute));
  auto &entryBlock = *initFunc.addEntryBlock();
  // rewriter.setInsertionPointToStart(&entryBlock);
  mlir::OpBuilder builder(initFunc.getContext());
  builder.setInsertionPointToStart(&entryBlock);
  auto waitOp = builder.create<gpu::WaitOp>(location, gpu::AsyncTokenType::get(module.getContext()), ValueRange{});
  auto alloc = builder.create<gpu::AllocOp>(location, memrefType, waitOp.getAsyncToken().getType(), ValueRange{waitOp.getAsyncToken()}, ValueRange{}, ValueRange{});
  auto transfer = builder.create<gpu::MemcpyOp>(location, alloc.getAsyncToken().getType(), ValueRange{alloc.getAsyncToken()}, 
                                                      alloc.getMemref(), static_cast<Value>(initFunc.getArgument(0)));
  /*auto waitBeforeReturn =*/ builder.create<gpu::WaitOp>(location, gpu::AsyncTokenType::get(module.getContext()), ValueRange{transfer.getAsyncToken()});
  builder.create<mlir::func::ReturnOp>(location, static_cast<Value>(alloc.getMemref()));
  module.push_back(initFunc);
  // rewriter.setInsertionPoint(insertPoint.getBlock(), insertPoint.getPoint());
}

void GenerateCleanupProc(const std::string& funcName,
                         ConversionPatternRewriter &rewriter,
                         Location location, 
                         ModuleOp module,
                         const std::vector<Type>& memrefTypes) {
  auto functionType = FunctionType::get(rewriter.getContext(), memrefTypes, {rewriter.getI32Type()});
  NamedAttribute visibilityAttribute{module.getSymVisibilityAttrName(), rewriter.getStringAttr("public")};
  auto initFunc = func::FuncOp::create(location, funcName, functionType, ArrayRef<NamedAttribute>(visibilityAttribute));
  auto &entryBlock = *initFunc.addEntryBlock();
  // rewriter.setInsertionPointToStart(&entryBlock);
  mlir::OpBuilder builder(initFunc.getContext());
  builder.setInsertionPointToStart(&entryBlock);
  auto waitOp = builder.create<gpu::WaitOp>(location, gpu::AsyncTokenType::get(module.getContext()), ValueRange{});
  auto asyncToken = waitOp.getAsyncToken();
  for (size_t i=0 ; i<memrefTypes.size() ; ++i) {
    auto dealloc = builder.create<gpu::DeallocOp>(location, asyncToken.getType(), ValueRange{asyncToken}, initFunc.getArgument(i));
    asyncToken = dealloc.getAsyncToken();
  }

  /*auto waitBeforeReturn =*/ builder.create<gpu::WaitOp>(location, asyncToken.getType(), ValueRange{asyncToken});
  auto constRetVal = builder.create<arith::ConstantIntOp>(location, 0 /*value*/, 32 /*width*/);
  builder.create<mlir::func::ReturnOp>(location, static_cast<Value>(constRetVal));
  module.push_back(initFunc);
}

template<typename BodyCreator_t>
void GenerateModelMemrefInitializerImpl(const std::string& funcName, ConversionPatternRewriter &rewriter, Location location, 
                                        ModuleOp module, MemRefType memrefType, bool sparseRep, BodyCreator_t createBody) {
  assert (memrefType.getShape().size() == 1);
  // SaveAndRestoreInsertionPoint saveAndRestoreEntryPoint(rewriter);
  auto modelMemrefElementType = memrefType.getElementType().cast<decisionforest::TiledNumericalNodeType>();
  int32_t tileSize = modelMemrefElementType.getTileSize();
  auto thresholdArgType = MemRefType::get({ memrefType.getShape()[0] * tileSize }, modelMemrefElementType.getThresholdElementType());
  auto indexArgType = MemRefType::get({ memrefType.getShape()[0] * tileSize }, modelMemrefElementType.getIndexElementType());
  auto tileShapeIDArgType = MemRefType::get(memrefType.getShape(), modelMemrefElementType.getTileShapeType());

  MemRefType childIndexArgType;
  FunctionType initModelMemrefFuncType;
  if (sparseRep) {
    childIndexArgType = MemRefType::get(memrefType.getShape(), modelMemrefElementType.getChildIndexType());
    initModelMemrefFuncType = rewriter.getFunctionType(TypeRange{thresholdArgType, indexArgType, tileShapeIDArgType, childIndexArgType}, memrefType);
  }
  else {
    initModelMemrefFuncType = rewriter.getFunctionType(TypeRange{thresholdArgType, indexArgType, tileShapeIDArgType}, memrefType);
  }
  NamedAttribute visibilityAttribute{module.getSymVisibilityAttrName(), rewriter.getStringAttr("public")};
  auto initModelMemrefFunc = mlir::func::FuncOp::create(location, funcName, initModelMemrefFuncType, ArrayRef<NamedAttribute>(visibilityAttribute));
  auto &entryBlock = *initModelMemrefFunc.addEntryBlock();
  // rewriter.setInsertionPointToStart(&entryBlock);
  mlir::OpBuilder builder(initModelMemrefFunc.getContext());
  builder.setInsertionPointToStart(&entryBlock);
  // auto& builder = rewriter;

  // Allocate the model memref
  auto waitOp = builder.create<gpu::WaitOp>(location, gpu::AsyncTokenType::get(module.getContext()), ValueRange{});
  auto modelMemrefGPUAlloc = builder.create<gpu::AllocOp>(location, memrefType, waitOp.getAsyncToken().getType(), ValueRange{waitOp.getAsyncToken()}, ValueRange{}, ValueRange{});

  auto asyncTokenType = modelMemrefGPUAlloc.getAsyncToken().getType();
  // Allocate and transfer all the arguments
  auto allocThresholds = builder.create<gpu::AllocOp>(location, thresholdArgType, asyncTokenType, ValueRange{modelMemrefGPUAlloc.getAsyncToken()}, ValueRange{}, ValueRange{});
  auto transferThresholds = builder.create<gpu::MemcpyOp>(location, asyncTokenType, ValueRange{allocThresholds.getAsyncToken()}, 
                                                      allocThresholds.getMemref(), static_cast<Value>(initModelMemrefFunc.getArgument(0)));

  auto allocFeatureIndices = builder.create<gpu::AllocOp>(location, indexArgType, asyncTokenType, ValueRange{transferThresholds.getAsyncToken()}, ValueRange{}, ValueRange{});
  auto transferFeatureIndices = builder.create<gpu::MemcpyOp>(location, asyncTokenType, ValueRange{allocFeatureIndices.getAsyncToken()}, 
                                                      allocFeatureIndices.getMemref(), static_cast<Value>(initModelMemrefFunc.getArgument(1)));
  
  Value currentAsyncToken = transferFeatureIndices.getAsyncToken();
  mlir::gpu::AllocOp allocTileShapeIds;
  mlir::gpu::MemcpyOp transferTileShapeIds;
  if (tileSize != 1) {
    allocTileShapeIds = builder.create<gpu::AllocOp>(location, tileShapeIDArgType, asyncTokenType, ValueRange{transferFeatureIndices.getAsyncToken()}, 
                                                     ValueRange{}, ValueRange{});
    transferTileShapeIds = builder.create<gpu::MemcpyOp>(location, asyncTokenType, ValueRange{allocTileShapeIds.getAsyncToken()}, 
                                                        allocTileShapeIds.getMemref(), static_cast<Value>(initModelMemrefFunc.getArgument(2)));
    currentAsyncToken = transferTileShapeIds.getAsyncToken();
  }

  mlir::gpu::AllocOp allocChildIndices;
  mlir::gpu::MemcpyOp transferChildIndices;
  if (sparseRep) {
    allocChildIndices = builder.create<gpu::AllocOp>(location, childIndexArgType, 
                                                     asyncTokenType, 
                                                     ValueRange{currentAsyncToken}, 
                                                     ValueRange{}, ValueRange{});
    transferChildIndices = builder.create<gpu::MemcpyOp>(location, asyncTokenType, 
                                                        ValueRange{allocChildIndices.getAsyncToken()}, 
                                                        allocChildIndices.getMemref(),
                                                        static_cast<Value>(initModelMemrefFunc.getArgument(3)));
    currentAsyncToken = transferChildIndices.getAsyncToken();
  }

  // Create the gpu.launch op
  auto oneIndexConst = builder.create<arith::ConstantIndexOp>(location, 1);
  int32_t numThreadsPerBlock = 32;
  int32_t numBlocks = std::ceil((double)memrefType.getShape()[0]/numThreadsPerBlock);
  auto numThreadBlocksConst = builder.create<arith::ConstantIndexOp>(location, numBlocks);
  auto numThreadsPerBlockConst = builder.create<arith::ConstantIndexOp>(location, numThreadsPerBlock);
  auto gpuLaunch = builder.create<gpu::LaunchOp>(location, numThreadBlocksConst, oneIndexConst, oneIndexConst, 
                                                numThreadsPerBlockConst, oneIndexConst, oneIndexConst,
                                                nullptr, asyncTokenType, 
                                                currentAsyncToken);

  builder.setInsertionPointToStart(&gpuLaunch.getBody().front());
  
  // // Generate the body of the launch op
  auto memrefLengthConst = builder.create<arith::ConstantIndexOp>(location, memrefType.getShape()[0]);
  auto firstThreadNum = builder.create<arith::MulIOp>(location, gpuLaunch.getBlockSizeX(), gpuLaunch.getBlockIds().x);
  auto elementIndex = builder.create<arith::AddIOp>(location, firstThreadNum, gpuLaunch.getThreadIds().x);
  auto inBoundsCondition = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::slt, elementIndex, memrefLengthConst);
  auto ifInBounds = builder.create<scf::IfOp>(location, inBoundsCondition, false);
  {
    // Generate the initialization code
    auto thenBuilder = ifInBounds.getThenBodyBuilder();
    createBody(memrefType, modelMemrefGPUAlloc.getMemref(), 
               thenBuilder, location, elementIndex, 
               allocThresholds.getMemref(), 
               allocFeatureIndices.getMemref(), 
               tileSize!=1 ? allocTileShapeIds.getMemref() : Value(),
               sparseRep ? allocChildIndices.getMemref() : Value());
  }
  builder.create<gpu::TerminatorOp>(location);
  // Wait and return 
  builder.setInsertionPointAfter(gpuLaunch); 
  /*auto waitBeforeReturn =*/ builder.create<gpu::WaitOp>(location, gpu::AsyncTokenType::get(module.getContext()), ValueRange{gpuLaunch.getAsyncToken()});
  builder.create<mlir::func::ReturnOp>(location, static_cast<Value>(modelMemrefGPUAlloc.getMemref()));
  module.push_back(initModelMemrefFunc);
}


// ===---------------------------------------------------=== //
// GPU array based representation
// ===---------------------------------------------------=== //

void GPUArrayBasedRepresentation::GenerateModelMemrefInitializer(const std::string& funcName, ConversionPatternRewriter &rewriter, Location location, 
                                                                 ModuleOp module, MemRefType memrefType) {
  GenerateModelMemrefInitializerImpl(funcName, rewriter, location, module, memrefType, false /*sparseRep*/,
                                     [&](MemRefType memrefType, 
                                         Value memrefValue,
                                         mlir::OpBuilder &builder, 
                                         Location location,
                                         Value tileIndex,
                                         Value thresholdMemref,
                                         Value indexMemref,
                                         Value tileShapeIdMemref,
                                         Value childIndexMemref) {
                                      this->GenModelMemrefInitFunctionBody(memrefType, 
                                                                           memrefValue,
                                                                           builder,
                                                                           location,
                                                                           tileIndex,
                                                                           thresholdMemref,
                                                                           indexMemref,
                                                                           tileShapeIdMemref);
                                     });
}

mlir::LogicalResult GPUArrayBasedRepresentation::GenerateModelGlobals(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter,
                                                                      std::shared_ptr<decisionforest::IModelSerializer> m_serializer) {
  auto location = op->getLoc();
  // Generate a new function with the extra arguments that are needed
  auto ensembleConstOp = AssertOpIsOfType<decisionforest::EnsembleConstantOp>(op);
  auto module = op->getParentOfType<mlir::ModuleOp>();
  assert (module);
  auto func = op->getParentOfType<func::FuncOp>();
  assert (func);

  mlir::decisionforest::DecisionForestAttribute forestAttribute = ensembleConstOp.getForest();
  mlir::decisionforest::DecisionForest<>& forest = forestAttribute.GetDecisionForest();
  auto forestType = ensembleConstOp.getResult().getType().cast<decisionforest::TreeEnsembleType>();
  assert (forestType.doAllTreesHaveSameTileSize()); // There is still an assumption here that all trees have the same tile size
  auto treeType = forestType.getTreeType(0).cast<decisionforest::TreeType>();

  auto thresholdType = treeType.getThresholdType();
  auto featureIndexType = treeType.getFeatureIndexType(); 
  auto tileSize = treeType.getTileSize();
  auto tileShapeType = treeType.getTileShapeType();
  auto childIndexType = treeType.getChildIndexType();

  m_tileSize = tileSize;
  m_thresholdType = thresholdType;
  m_featureIndexType = featureIndexType;
  m_tileShapeType = tileShapeType;

  Type modelMemrefElementType = decisionforest::TiledNumericalNodeType::get(thresholdType, featureIndexType, tileShapeType, 
                                                                            tileSize, childIndexType);

  m_serializer->Persist(forest, forestType);
  
  auto modelMemrefSize = decisionforest::GetTotalNumberOfTiles();
  auto modelMemrefType = MemRefType::get({modelMemrefSize}, modelMemrefElementType);
  func.insertArgument(func.getNumArguments(), modelMemrefType, mlir::DictionaryAttr(), location);
  m_modelMemrefArgIndex = func.getNumArguments() - 1;

  auto offsetSize = (int32_t)forest.NumTrees();
  auto offsetMemrefType = MemRefType::get({offsetSize}, rewriter.getIndexType());
  func.insertArgument(func.getNumArguments(), offsetMemrefType, mlir::DictionaryAttr(), location);
  m_offsetMemrefArgIndex = func.getNumArguments() - 1;

  // Add the length argument
  func.insertArgument(func.getNumArguments(), offsetMemrefType, mlir::DictionaryAttr(), location);
  m_lengthMemrefArgIndex = func.getNumArguments() - 1;

  // Add the class info argument
  auto classInfoSize = forest.IsMultiClassClassifier() ? offsetSize : 0;
  auto classInfoMemrefType = MemRefType::get({classInfoSize}, rewriter.getI8Type());
  func.insertArgument(func.getNumArguments(), classInfoMemrefType, mlir::DictionaryAttr(), location);
  m_classInfoMemrefArgIndex = func.getNumArguments() - 1;

  m_modelMemref = func.getArgument(m_modelMemrefArgIndex);
  
  GenerateModelMemrefInitializer("Init_Model", rewriter, location, module, modelMemrefType);
  GenerateSimpleInitializer("Init_Offsets", rewriter, location, module, offsetMemrefType);
  GenerateSimpleInitializer("Init_Lengths", rewriter, location, module, offsetMemrefType);
  GenerateSimpleInitializer("Init_ClassIds", rewriter, location, module, classInfoMemrefType);

  GenerateCleanupProc("Dealloc_Buffers", 
                      rewriter, 
                      location,
                      module, 
                      std::vector<Type>{modelMemrefType, offsetMemrefType, offsetMemrefType}); //, classInfoMemrefType})
  EnsembleConstantLoweringInfo info 
  {
    static_cast<Value>(m_modelMemref),
    static_cast<Value>(func.getArgument(m_offsetMemrefArgIndex)),
    static_cast<Value>(func.getArgument(m_lengthMemrefArgIndex)),
    static_cast<Value>(func.getArgument(m_classInfoMemrefArgIndex)),
    modelMemrefType,
    offsetMemrefType,
    offsetMemrefType,
    classInfoMemrefType,
  };
  ensembleConstantToMemrefsMap[op] = info;
  return mlir::success();
}

// mlir::Value GPUArrayBasedRepresentation::GetTreeMemref(mlir::Value treeValue) {
//   return m_modelMemref;
// }

// void GPUArrayBasedRepresentation::GenerateTreeMemref(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) {

// }

// mlir::Value GPUArrayBasedRepresentation::GenerateGetTreeClassId(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op, Value ensemble, Value treeIndex) {
//   return mlir::Value();
// }

std::shared_ptr<IRepresentation> ConstructGPUArrayBasedRepresentation() {
  return std::make_shared<GPUArrayBasedRepresentation>();
}

REGISTER_REPRESENTATION(gpu_array, ConstructGPUArrayBasedRepresentation)

// ===---------------------------------------------------=== //
// GPU sparse representation
// ===---------------------------------------------------=== //

void GPUSparseRepresentation::GenerateModelMemrefInitializer(const std::string& funcName, 
                                                             ConversionPatternRewriter &rewriter, Location location, 
                                                             ModuleOp module, MemRefType memrefType) {
  GenerateModelMemrefInitializerImpl(funcName, rewriter, location, module, memrefType, true /*sparseRep*/,
                                     [&](MemRefType memrefType, 
                                         Value memrefValue,
                                         mlir::OpBuilder &builder, 
                                         Location location,
                                         Value tileIndex,
                                         Value thresholdMemref,
                                         Value indexMemref,
                                         Value tileShapeIdMemref,
                                         Value childIndexMemref) {
                                      this->GenModelMemrefInitFunctionBody(memrefType, 
                                                                           memrefValue,
                                                                           builder,
                                                                           location,
                                                                           tileIndex,
                                                                           thresholdMemref,
                                                                           indexMemref,
                                                                           tileShapeIdMemref,
                                                                           childIndexMemref);
                                     });
}

mlir::LogicalResult GPUSparseRepresentation::GenerateModelGlobals(Operation *op, 
                                                                  ArrayRef<Value> operands,
                                                                  ConversionPatternRewriter &rewriter,
                                                                  std::shared_ptr<decisionforest::IModelSerializer> m_serializer) {
  auto location = op->getLoc();
  // Generate a new function with the extra arguments that are needed
  auto ensembleConstOp = AssertOpIsOfType<decisionforest::EnsembleConstantOp>(op);
  auto module = op->getParentOfType<mlir::ModuleOp>();
  assert (module);
  auto func = op->getParentOfType<func::FuncOp>();
  assert (func);

  mlir::decisionforest::DecisionForestAttribute forestAttribute = ensembleConstOp.getForest();
  mlir::decisionforest::DecisionForest<>& forest = forestAttribute.GetDecisionForest();
  auto forestType = ensembleConstOp.getResult().getType().cast<decisionforest::TreeEnsembleType>();
  assert (forestType.doAllTreesHaveSameTileSize()); // There is still an assumption here that all trees have the same tile size
  auto treeType = forestType.getTreeType(0).cast<decisionforest::TreeType>();

  auto thresholdType = treeType.getThresholdType();
  auto featureIndexType = treeType.getFeatureIndexType(); 
  auto tileSize = treeType.getTileSize();
  auto tileShapeType = treeType.getTileShapeType();
  auto childIndexType = treeType.getChildIndexType();

  m_tileSize = tileSize;
  m_thresholdType = thresholdType;
  m_featureIndexType = featureIndexType;
  m_tileShapeType = tileShapeType;

  Type modelMemrefElementType = decisionforest::TiledNumericalNodeType::get(thresholdType, featureIndexType, tileShapeType, 
                                                                            tileSize, childIndexType);

  m_serializer->Persist(forest, forestType);
  
  auto modelMemrefSize = decisionforest::GetTotalNumberOfTiles();
  auto modelMemrefType = MemRefType::get({modelMemrefSize}, modelMemrefElementType);
  func.insertArgument(func.getNumArguments(), modelMemrefType, mlir::DictionaryAttr(), location);
  m_modelMemrefArgIndex = func.getNumArguments() - 1;

  auto offsetSize = (int32_t)forest.NumTrees();
  auto offsetMemrefType = MemRefType::get({offsetSize}, rewriter.getIndexType());
  func.insertArgument(func.getNumArguments(), offsetMemrefType, mlir::DictionaryAttr(), location);
  m_offsetMemrefArgIndex = func.getNumArguments() - 1;

  // Add the length argument
  func.insertArgument(func.getNumArguments(), offsetMemrefType, mlir::DictionaryAttr(), location);
  m_lengthMemrefArgIndex = func.getNumArguments() - 1;

  // Add the class info argument
  auto classInfoSize = forest.IsMultiClassClassifier() ? offsetSize : 0;
  auto classInfoMemrefType = MemRefType::get({classInfoSize}, treeType.getResultType());
  func.insertArgument(func.getNumArguments(), classInfoMemrefType, mlir::DictionaryAttr(), location);
  m_classInfoMemrefArgIndex = func.getNumArguments() - 1;

  m_modelMemref = func.getArgument(m_modelMemrefArgIndex);
  
  GenerateModelMemrefInitializer("Init_Model", rewriter, location, module, modelMemrefType);
  GenerateSimpleInitializer("Init_Offsets", rewriter, location, module, offsetMemrefType);
  GenerateSimpleInitializer("Init_Lengths", rewriter, location, module, offsetMemrefType);
  GenerateSimpleInitializer("Init_ClassIds", rewriter, location, module, classInfoMemrefType);

  GenerateCleanupProc("Dealloc_Buffers", 
                    rewriter, 
                    location,
                    module, 
                    std::vector<Type>{modelMemrefType, offsetMemrefType, offsetMemrefType}); //, classInfoMemrefType})

  SparseEnsembleConstantLoweringInfo info 
  {
    static_cast<Value>(m_modelMemref),
    static_cast<Value>(func.getArgument(m_offsetMemrefArgIndex)),
    static_cast<Value>(func.getArgument(m_lengthMemrefArgIndex)),
    Value(), // LUT
    Value(), // leaves 
    Value(), // leavesOffset
    Value(), // leavesLength
    static_cast<Value>(func.getArgument(m_classInfoMemrefArgIndex)),
    modelMemrefType,
    offsetMemrefType,
    offsetMemrefType,
    Type(), // LUT type
    Type(), // Leaves type
    classInfoMemrefType,
  };
  sparseEnsembleConstantToMemrefsMap[op] = info;
  return mlir::success();
}

std::shared_ptr<IRepresentation> ConstructGPUSparseRepresentation() {
  return std::make_shared<GPUSparseRepresentation>();
}

REGISTER_REPRESENTATION(gpu_sparse, ConstructGPUSparseRepresentation)

}
}