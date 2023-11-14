#ifdef TREEBEARD_GPU_SUPPORT

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#include "../json/JSONHelpers.h"
#include "LIRLoweringHelpers.h"
#include "Logger.h"
#include "ModelSerializers.h"
#include "OpLoweringUtils.h"
#include "ReorgForestRepresentation.h"
#include "Representations.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::decisionforest::helpers;

namespace {

const int32_t kAlignedPointerIndexInMemrefStruct = 1;
const int32_t kOffsetIndexInMemrefStruct = 2;

Type GetMemrefElementType(Value memrefOperand) {
  auto memrefType = memrefOperand.getType();
  auto memrefStructType = memrefType.cast<LLVM::LLVMStructType>();
  auto alignedPtrType =
      memrefStructType.getBody()[kAlignedPointerIndexInMemrefStruct]
          .cast<LLVM::LLVMPointerType>();
  auto memrefElemType = alignedPtrType.getElementType();
  return memrefElemType;
}

Value GenerateGetElementPtr(Location location,
                            ConversionPatternRewriter &rewriter,
                            Value memrefOperand, Value indexVal) {

  auto memrefType = memrefOperand.getType();
  auto memrefStructType = memrefType.cast<LLVM::LLVMStructType>();
  auto alignedPtrType =
      memrefStructType.getBody()[kAlignedPointerIndexInMemrefStruct]
          .cast<LLVM::LLVMPointerType>();
  auto memrefElemType = alignedPtrType.getElementType();

  auto indexType = indexVal.getType();
  assert(indexType.isa<IntegerType>());

  // Extract the memref's aligned pointer
  auto extractMemrefBufferPointer = rewriter.create<LLVM::ExtractValueOp>(
      location, alignedPtrType, memrefOperand,
      rewriter.getDenseI64ArrayAttr(kAlignedPointerIndexInMemrefStruct));

  auto extractMemrefOffset = rewriter.create<LLVM::ExtractValueOp>(
      location, indexType, memrefOperand,
      rewriter.getDenseI64ArrayAttr(kOffsetIndexInMemrefStruct));

  auto actualIndex = rewriter.create<LLVM::AddOp>(
      location, indexType, static_cast<Value>(extractMemrefOffset),
      static_cast<Value>(indexVal));

  auto elementPtrType = LLVM::LLVMPointerType::get(memrefElemType);
  auto elementPtr = rewriter.create<LLVM::GEPOp>(
      location, elementPtrType, static_cast<Value>(extractMemrefBufferPointer),
      ValueRange({static_cast<Value>(actualIndex)}));

  return elementPtr;
}

Value GenerateMemrefLoadForLoadFromTile(ConversionPatternRewriter &rewriter,
                                        mlir::Location location, Value buffer,
                                        Value nodeIndex, Value treeIndex,
                                        int32_t numTrees) {
  // First generate the index into the buffer
  //  index = nodeIndex*numTrees + treeIndex
  auto numTreesConst = rewriter.create<LLVM::ConstantOp>(
      location, nodeIndex.getType(), numTrees);
  auto numTreesTimesNodeIndex = rewriter.create<LLVM::MulOp>(
      location, nodeIndex, static_cast<Value>(numTreesConst));
  auto memrefIndex = rewriter.create<LLVM::AddOp>(
      location, treeIndex, static_cast<Value>(numTreesTimesNodeIndex));

  // Then generate the memref load
  auto elementPtr =
      GenerateGetElementPtr(location, rewriter, buffer, memrefIndex);
  // Load the element
  auto elementType = GetMemrefElementType(buffer);
  auto elementVal = rewriter.create<LLVM::LoadOp>(
      location, elementType, static_cast<Value>(elementPtr));
  return static_cast<Value>(elementVal);
}

struct LoadTileThresholdOpLowering : public ConversionPattern {
  decisionforest::ReorgForestRepresentation *m_representation;
  LoadTileThresholdOpLowering(
      LLVMTypeConverter &typeConverter,
      decisionforest::ReorgForestRepresentation *representation)
      : ConversionPattern(
            typeConverter,
            mlir::decisionforest::LoadTileThresholdsOp::getOperationName(),
            1 /*benefit*/, &typeConverter.getContext()),
        m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 3);
    auto loadTileThresholdOp =
        AssertOpIsOfType<mlir::decisionforest::LoadTileThresholdsOp>(op);
    auto treeMemrefType =
        loadTileThresholdOp.getTreeMemref().getType().cast<MemRefType>();
    auto treeMemrefElementType =
        treeMemrefType.getElementType()
            .cast<decisionforest::ReorgMemrefElementType>();
    auto numTrees = treeMemrefElementType.getNumTrees();
    auto threshold =
        GenerateMemrefLoadForLoadFromTile(rewriter, op->getLoc(), operands[0],
                                          operands[1], operands[2], numTrees);
    rewriter.replaceOp(op, static_cast<Value>(threshold));
    return mlir::success();
  }
};

struct LoadTileFeatureIndicesOpLowering : public ConversionPattern {
  decisionforest::ReorgForestRepresentation *m_representation;
  LoadTileFeatureIndicesOpLowering(
      LLVMTypeConverter &typeConverter,
      decisionforest::ReorgForestRepresentation *representation)
      : ConversionPattern(
            typeConverter,
            mlir::decisionforest::LoadTileFeatureIndicesOp::getOperationName(),
            1 /*benefit*/, &typeConverter.getContext()),
        m_representation(representation) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    assert(operands.size() == 3);
    auto loadTileFeatureIndexOp =
        AssertOpIsOfType<mlir::decisionforest::LoadTileFeatureIndicesOp>(op);
    auto treeMemrefType =
        loadTileFeatureIndexOp.getTreeMemref().getType().cast<MemRefType>();
    auto treeMemrefElementType =
        treeMemrefType.getElementType()
            .cast<decisionforest::ReorgMemrefElementType>();
    auto numTrees = treeMemrefElementType.getNumTrees();
    auto featureIndex =
        GenerateMemrefLoadForLoadFromTile(rewriter, op->getLoc(), operands[0],
                                          operands[1], operands[2], numTrees);
    rewriter.replaceOp(op, static_cast<Value>(featureIndex));
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace decisionforest {

// ===---------------------------------------------------=== //
// Helpers
// ===---------------------------------------------------=== //

void GenerateSimpleInitializer(const std::string &funcName,
                               ConversionPatternRewriter &rewriter,
                               Location location, ModuleOp module,
                               MemRefType memrefType);
int64_t
GetNumberOfTreesToCache(decisionforest::CacheTreesFromEnsembleOp cacheTreesOp);

// ===---------------------------------------------------=== //
// Reorg forest serializer methods
// ===---------------------------------------------------=== //

ReorgForestSerializer::ReorgForestSerializer(const std::string &filename)
    : IModelSerializer(filename) {}

int32_t ComputeBufferSize(decisionforest::DecisionForest &forest,
                          int32_t numTrees = -1) {
  int32_t maxDepth = -1;
  for (auto &tree : forest.GetTrees()) {
    auto depth = tree->GetTreeDepth();
    if (depth > maxDepth)
      maxDepth = depth;
  }
  if (numTrees == -1)
    numTrees = static_cast<int32_t>(forest.NumTrees());
  return numTrees * (std::pow(2, maxDepth) - 1);
}

void ReorgForestSerializer::WriteSingleTreeIntoReorgBuffer(
    mlir::decisionforest::DecisionForest &forest, int32_t treeIndex) {
  auto &tree = forest.GetTree(treeIndex);
  auto thresholds = tree.GetThresholdArray();
  auto featureIndices = tree.GetFeatureIndexArray();
  assert(thresholds.size() == featureIndices.size());
  for (size_t i = 0; i < thresholds.size(); ++i) {
    auto bufferIndex = i * forest.NumTrees() + treeIndex;
    m_thresholds.at(bufferIndex) = thresholds[i];
    m_featureIndices.at(bufferIndex) = featureIndices[i];
  }
}

void ReorgForestSerializer::ReadData() {
  m_json.clear();
  std::ifstream fin(m_filepath);
  assert(fin);
  fin >> m_json;
  fin.close();

  m_numberOfTrees = m_json["NumberOfTrees"];
  m_numberOfClasses = m_json["NumberOfClasses"];

  // m_thresholds = m_json["Thresholds"];
  m_thresholds.clear();
  ParseJSONList<std::vector<double>, double>(m_thresholds,
                                             m_json["Thresholds"]);
  // m_featureIndices = m_json["FeatureIndices"];
  m_featureIndices.clear();
  ParseJSONList<std::vector<int32_t>, int32_t>(m_featureIndices,
                                               m_json["FeatureIndices"]);
  m_classIds.clear();
  ParseJSONList<std::vector<int8_t>, int8_t>(m_classIds, m_json["ClassIDs"]);
}

void ReorgForestSerializer::WriteJSONFile() {
  assert(m_filepath != "");
  m_json.clear();

  m_json["NumberOfTrees"] = m_numberOfTrees;
  m_json["NumberOfClasses"] = m_numberOfClasses;

  m_json["Thresholds"] = m_thresholds;
  m_json["FeatureIndices"] = m_featureIndices;
  m_json["ClassIDs"] = m_classIds;

  std::ofstream fout(m_filepath);
  assert(fout);
  fout << m_json;
  fout.close();
}

void ReorgForestSerializer::Persist(
    mlir::decisionforest::DecisionForest &forest,
    mlir::decisionforest::TreeEnsembleType forestType) {
  m_numberOfTrees = forestType.getNumberOfTrees();
  m_numberOfClasses =
      forest.IsMultiClassClassifier() ? forest.GetNumClasses() : 1;
  auto treeType = forestType.getTreeType(0).cast<decisionforest::TreeType>();
  m_thresholdBitWidth = treeType.getThresholdType().getIntOrFloatBitWidth();
  m_featureIndexBitWidth =
      treeType.getFeatureIndexType().getIntOrFloatBitWidth();

  auto bufferSize = ComputeBufferSize(forest);
  m_thresholds.resize(bufferSize, -1e6);
  m_featureIndices.resize(bufferSize, -1);

  for (int32_t i = 0; i < (int32_t)forest.NumTrees(); ++i) {
    this->WriteSingleTreeIntoReorgBuffer(forest, i);
    if (forest.IsMultiClassClassifier()) {
      m_classIds.push_back(forest.GetTree(i).GetClassId());
    }
  }

  // TODO Write out a JSON file
  WriteJSONFile();
}

template <typename VectorElemType, typename MemrefElemType>
void ReorgForestSerializer::InitializeSingleBuffer(
    const std::string &initFuncName, const std::vector<VectorElemType> &vals,
    Memref<VectorElemType, 1> &gpuMemref) {
  using InitFunc_t = Memref<VectorElemType, 1> (*)(
      MemrefElemType *, MemrefElemType *, int64_t, int64_t, int64_t);
  auto initMemref = GetFunctionAddress<InitFunc_t>(initFuncName);
  std::vector<MemrefElemType> castVector(vals.begin(), vals.end());
  gpuMemref = initMemref(castVector.data(), castVector.data(), 0 /*offset*/,
                         vals.size() /*length*/, 1 /*stride*/);
}

void ReorgForestSerializer::InitializeThresholds() {
  if (m_thresholdBitWidth == 32)
    InitializeSingleBuffer<double, float>("Init_Thresholds", m_thresholds,
                                          m_thresholdMemref);
  else if (m_thresholdBitWidth == 64)
    InitializeSingleBuffer<double, double>("Init_Thresholds", m_thresholds,
                                           m_thresholdMemref);
  else
    assert(false && "Unsupported threshold bit width");
}

void ReorgForestSerializer::InitializeFeatureIndices() {
  if (m_featureIndexBitWidth == 8)
    InitializeSingleBuffer<int32_t, int8_t>(
        "Init_FeatureIndices", m_featureIndices, m_featureIndexMemref);
  else if (m_featureIndexBitWidth == 16)
    InitializeSingleBuffer<int32_t, int16_t>(
        "Init_FeatureIndices", m_featureIndices, m_featureIndexMemref);
  else if (m_featureIndexBitWidth == 32)
    InitializeSingleBuffer<int32_t, int32_t>(
        "Init_FeatureIndices", m_featureIndices, m_featureIndexMemref);
  else
    assert(false && "Unsupported feature index bit width");
}

void ReorgForestSerializer::InitializeBuffersImpl() {
  InitializeThresholds();
  InitializeFeatureIndices();
  if (m_numberOfClasses > 1) {
    InitializeSingleBuffer<int8_t, int8_t>("Init_ClassIds", m_classIds,
                                           m_classIDMemref);
  }
}

void ReorgForestSerializer::CallPredictionMethod(void *predictFuncPtr,
                                                 Memref<double, 2> inputs,
                                                 Memref<double, 1> results) {
  using InputElementType = double;
  using ReturnType = double;

  typedef Memref<ReturnType, 1> (*InferenceFunc_t)(
      InputElementType *, InputElementType *, int64_t, int64_t, int64_t,
      int64_t, int64_t, ReturnType *, ReturnType *, int64_t, int64_t, int64_t,
      double *, double *, int64_t, int64_t, int64_t, int32_t *, int32_t *,
      int64_t, int64_t, int64_t, int8_t *, int8_t *, int64_t, int64_t, int64_t);
  auto inferenceFuncPtr = reinterpret_cast<InferenceFunc_t>(predictFuncPtr);
  inferenceFuncPtr(inputs.bufferPtr, inputs.alignedPtr, inputs.offset,
                   inputs.lengths[0], inputs.lengths[1], inputs.strides[0],
                   inputs.strides[1], results.bufferPtr, results.alignedPtr,
                   results.offset, results.lengths[0], results.strides[0],
                   m_thresholdMemref.bufferPtr, m_thresholdMemref.alignedPtr,
                   m_thresholdMemref.offset, m_thresholdMemref.lengths[0],
                   m_thresholdMemref.strides[0], m_featureIndexMemref.bufferPtr,
                   m_featureIndexMemref.alignedPtr, m_featureIndexMemref.offset,
                   m_featureIndexMemref.lengths[0],
                   m_featureIndexMemref.strides[0], m_classIDMemref.bufferPtr,
                   m_classIDMemref.alignedPtr, m_classIDMemref.offset,
                   m_classIDMemref.lengths[0], m_classIDMemref.strides[0]);
  return;
}

bool ReorgForestSerializer::HasCustomPredictionMethod() { return true; }

void ReorgForestSerializer::CleanupBuffers() {
  if (m_numberOfClasses > 0) {
    using CleanupFunc_t =
        int32_t (*)(double *, double *, int64_t, int64_t, int64_t, int32_t *,
                    int32_t *, int64_t, int64_t, int64_t, int8_t *, int8_t *,
                    int64_t, int64_t, int64_t);

    auto cleanupFuncPtr =
        this->GetFunctionAddress<CleanupFunc_t>("Dealloc_Buffers");
    cleanupFuncPtr(m_thresholdMemref.bufferPtr, m_thresholdMemref.alignedPtr,
                   m_thresholdMemref.offset, m_thresholdMemref.lengths[0],
                   m_thresholdMemref.strides[0], m_featureIndexMemref.bufferPtr,
                   m_featureIndexMemref.alignedPtr, m_featureIndexMemref.offset,
                   m_featureIndexMemref.lengths[0],
                   m_featureIndexMemref.strides[0], m_classIDMemref.bufferPtr,
                   m_classIDMemref.alignedPtr, m_classIDMemref.offset,
                   m_classIDMemref.lengths[0], m_classIDMemref.strides[0]);
  } else {
    using CleanupFunc_t =
        int32_t (*)(double *, double *, int64_t, int64_t, int64_t, int32_t *,
                    int32_t *, int64_t, int64_t, int64_t);

    auto cleanupFuncPtr =
        this->GetFunctionAddress<CleanupFunc_t>("Dealloc_Buffers");
    cleanupFuncPtr(m_thresholdMemref.bufferPtr, m_thresholdMemref.alignedPtr,
                   m_thresholdMemref.offset, m_thresholdMemref.lengths[0],
                   m_thresholdMemref.strides[0], m_featureIndexMemref.bufferPtr,
                   m_featureIndexMemref.alignedPtr, m_featureIndexMemref.offset,
                   m_featureIndexMemref.lengths[0],
                   m_featureIndexMemref.strides[0]);
  }
}

std::shared_ptr<IModelSerializer>
ConstructGPUReorgForestSerializer(const std::string &jsonFilename) {
  return std::make_shared<ReorgForestSerializer>(jsonFilename);
}

REGISTER_SERIALIZER(gpu_reorg, ConstructGPUReorgForestSerializer)

// ===---------------------------------------------------=== //
// Reorg forest representation methods
// ===---------------------------------------------------=== //

void GenerateCleanupProc(const std::string &funcName,
                         ConversionPatternRewriter &rewriter, Location location,
                         ModuleOp module, const std::vector<Type> &memrefTypes);

mlir::LogicalResult ReorgForestRepresentation::GenerateModelGlobals(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter,
    std::shared_ptr<decisionforest::IModelSerializer> m_serializer) {
  auto location = op->getLoc();
  // Generate a new function with the extra arguments that are needed
  auto ensembleConstOp =
      AssertOpIsOfType<decisionforest::EnsembleConstantOp>(op);
  auto module = op->getParentOfType<mlir::ModuleOp>();
  assert(module);
  auto func = op->getParentOfType<func::FuncOp>();
  assert(func);

  mlir::decisionforest::DecisionForestAttribute forestAttribute =
      ensembleConstOp.getForest();
  mlir::decisionforest::DecisionForest &forest =
      forestAttribute.GetDecisionForest();
  auto forestType = ensembleConstOp.getResult()
                        .getType()
                        .cast<decisionforest::TreeEnsembleType>();
  assert(
      forestType.doAllTreesHaveSameTileSize()); // There is still an assumption
                                                // here that all trees have the
                                                // same tile size
  auto treeType = forestType.getTreeType(0).cast<decisionforest::TreeType>();

  auto thresholdType = treeType.getThresholdType();
  auto featureIndexType = treeType.getFeatureIndexType();
  auto tileSize = treeType.getTileSize();

  assert(tileSize == 1 &&
         "Only scalar code generation supported by this representation");
  m_tileSize = tileSize;
  m_thresholdType = thresholdType;
  m_featureIndexType = featureIndexType;
  m_numTrees = forestType.getNumberOfTrees();

  m_serializer->Persist(forest, forestType);

  auto memrefSize = static_cast<int64_t>(
      reinterpret_cast<ReorgForestSerializer *>(m_serializer.get())
          ->GetNumberOfNodes());

  auto thresholdMemrefElemType =
      decisionforest::ReorgMemrefElementType::get(m_thresholdType, m_numTrees);
  auto thresholdMemrefType =
      MemRefType::get({memrefSize}, thresholdMemrefElemType);
  func.insertArgument(func.getNumArguments(), thresholdMemrefType,
                      mlir::DictionaryAttr(), location);
  m_thresholdMemrefArgIndex = func.getNumArguments() - 1;

  auto featureIndexMemrefElemType = decisionforest::ReorgMemrefElementType::get(
      m_featureIndexType, m_numTrees);
  auto featureIndexMemrefType =
      MemRefType::get({memrefSize}, featureIndexMemrefElemType);
  func.insertArgument(func.getNumArguments(), featureIndexMemrefType,
                      mlir::DictionaryAttr(), location);
  m_featureIndexMemrefArgIndex = func.getNumArguments() - 1;

  auto classInfoSize = forest.IsMultiClassClassifier()
                           ? static_cast<int64_t>(forestType.getNumberOfTrees())
                           : 0;
  auto classInfoMemrefType =
      MemRefType::get({classInfoSize}, rewriter.getI8Type());
  func.insertArgument(func.getNumArguments(), classInfoMemrefType,
                      mlir::DictionaryAttr(), location);
  m_classInfoMemrefArgIndex = func.getNumArguments() - 1;

  auto actualThresholdMemrefType =
      MemRefType::get({memrefSize}, m_thresholdType);
  GenerateSimpleInitializer("Init_Thresholds", rewriter, location, module,
                            actualThresholdMemrefType);
  auto actualFeatureIndexMemrefType =
      MemRefType::get({memrefSize}, m_featureIndexType);
  GenerateSimpleInitializer("Init_FeatureIndices", rewriter, location, module,
                            actualFeatureIndexMemrefType);

  auto cleanupArgs = std::vector<Type>{actualThresholdMemrefType,
                                       actualFeatureIndexMemrefType};
  if (forest.IsMultiClassClassifier()) {
    GenerateSimpleInitializer("Init_ClassIds", rewriter, location, module,
                              classInfoMemrefType);
    cleanupArgs.push_back(classInfoMemrefType);
  }

  GenerateCleanupProc("Dealloc_Buffers", rewriter, location, module,
                      cleanupArgs);

  m_thresholdMemref = func.getArgument(m_thresholdMemrefArgIndex);
  m_featureIndexMemref = func.getArgument(m_featureIndexMemrefArgIndex);
  m_classInfoMemref = func.getArgument(m_classInfoMemrefArgIndex);

  return mlir::success();
}

mlir::Value ReorgForestRepresentation::GenerateMoveToChild(
    mlir::Location location, ConversionPatternRewriter &rewriter,
    mlir::Value nodeIndex, mlir::Value childNumber, int32_t tileSize,
    std::vector<mlir::Value> &extraLoads) {
  // nodeIndex = 2 * nodeIndex + childNumber
  assert(tileSize == 1);
  auto oneConstant = rewriter.create<arith::ConstantIndexOp>(location, 1);
  auto tileSizeConstant =
      rewriter.create<arith::ConstantIndexOp>(location, tileSize + 1);
  auto tileSizeTimesIndex = rewriter.create<arith::MulIOp>(
      location, rewriter.getIndexType(), static_cast<Value>(nodeIndex),
      static_cast<Value>(tileSizeConstant));
  auto tileSizeTimesIndexPlus1 = rewriter.create<arith::AddIOp>(
      location, rewriter.getIndexType(), static_cast<Value>(tileSizeTimesIndex),
      static_cast<Value>(oneConstant));

  auto newIndex = rewriter.create<arith::AddIOp>(
      location, rewriter.getIndexType(), tileSizeTimesIndexPlus1, childNumber);
  return newIndex;
}

mlir::Value ReorgForestRepresentation::GenerateGetTreeClassId(
    mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op,
    Value ensemble, Value treeIndex) {
  auto treeClassMemrefType =
      m_classInfoMemref.getType().cast<mlir::MemRefType>();

  auto classId = rewriter.create<memref::LoadOp>(
      op->getLoc(), treeClassMemrefType.getElementType(), m_classInfoMemref,
      treeIndex);
  return classId;
}

mlir::Value ReorgForestRepresentation::GenerateGetLeafValueOp(
    ConversionPatternRewriter &rewriter, mlir::Operation *op,
    mlir::Value treeValue, mlir::Value nodeIndex) {
  auto location = op->getLoc();

  auto treeMemref = this->GetThresholdsMemref(treeValue);
  auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
  assert(treeMemrefType);

  // Load threshold
  // TODO Ideally, this should be a different op for when we deal with tile
  // sizes != 1. We will then need to load a single threshold value and cast it
  // the trees return type
  Value treeIndex = GetTreeIndex(treeValue);
  auto loadThresholdOp = rewriter.create<decisionforest::LoadTileThresholdsOp>(
      location, m_thresholdType, treeMemref, static_cast<Value>(nodeIndex),
      treeIndex);
  Value leafValue = loadThresholdOp;
  return leafValue;
}

mlir::Value ReorgForestRepresentation::GenerateIsLeafOp(
    ConversionPatternRewriter &rewriter, mlir::Operation *op,
    mlir::Value treeValue, mlir::Value nodeIndex) {
  auto location = op->getLoc();
  auto treeMemref = this->GetFeatureIndexMemref(treeValue);
  auto treeMemrefType = treeMemref.getType().cast<MemRefType>();
  assert(treeMemrefType);

  auto treeIndex = GetTreeIndex(treeValue);
  auto loadFeatureIndexOp =
      rewriter.create<decisionforest::LoadTileFeatureIndicesOp>(
          location, m_featureIndexType, treeMemref,
          static_cast<Value>(nodeIndex), treeIndex);

  Value featureIndexValue = loadFeatureIndexOp;
  auto minusOneConstant = rewriter.create<arith::ConstantIntOp>(
      location, int64_t(-1), m_featureIndexType);
  auto comparison = rewriter.create<arith::CmpIOp>(
      location, mlir::arith::CmpIPredicate::eq, featureIndexValue,
      static_cast<Value>(minusOneConstant));

  return static_cast<Value>(comparison);
}

mlir::Value ReorgForestRepresentation::GenerateIsLeafTileOp(
    ConversionPatternRewriter &rewriter, mlir::Operation *op,
    mlir::Value treeValue, mlir::Value nodeIndex) {
  return this->GenerateIsLeafOp(rewriter, op, treeValue, nodeIndex);
}

void ReorgForestRepresentation::AddTypeConversions(
    mlir::MLIRContext &context, LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([&](decisionforest::ReorgMemrefElementType type) {
    auto elemType = type.getElementType();
    return elemType;
  });
}

void ReorgForestRepresentation::AddLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<LoadTileFeatureIndicesOpLowering, LoadTileThresholdOpLowering>(
      converter, this);
}

mlir::Value
ReorgForestRepresentation::GetThresholdsMemref(mlir::Value treeValue) {
  auto definingOp = treeValue.getDefiningOp();
  // if defining op is in m_getTreeFromEnsembleMap, then return threshold memref
  // from map
  auto getTreeMapIter = m_getTreeFromEnsembleMap.find(definingOp);
  if (getTreeMapIter != m_getTreeFromEnsembleMap.end()) {
    return getTreeMapIter->second.thresholdBuffer;
  }
  return m_thresholdMemref;
}

mlir::Value
ReorgForestRepresentation::GetFeatureIndexMemref(mlir::Value treeValue) {
  auto definingOp = treeValue.getDefiningOp();
  // if defining op is in m_getTreeFromEnsembleMap, then return threshold memref
  // from map
  auto getTreeMapIter = m_getTreeFromEnsembleMap.find(definingOp);
  if (getTreeMapIter != m_getTreeFromEnsembleMap.end()) {
    return getTreeMapIter->second.featureIndexBuffer;
  }
  return m_featureIndexMemref;
}

void ReorgForestRepresentation::GenerateTreeMemref(
    mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op,
    Value ensemble, Value treeIndex) {
  // if the source of the ensemble is a cache op, then subtract the start index
  // from the tree index
  auto ensembleDefiningOp = ensemble.getDefiningOp();
  auto ensembleConstantOp =
      llvm::dyn_cast<mlir::decisionforest::EnsembleConstantOp>(
          ensembleDefiningOp);
  if (ensembleConstantOp)
    return;

  auto cacheTreesOp =
      AssertOpIsOfType<decisionforest::CacheTreesFromEnsembleOp>(
          ensembleDefiningOp);
  auto correctedTreeIndex = rewriter.create<arith::SubIOp>(
      op->getLoc(), treeIndex, cacheTreesOp.getStartTreeIndex());

  auto cacheInfoIter = m_treeCacheMap.find(cacheTreesOp.getOperation());
  assert(cacheInfoIter != m_treeCacheMap.end());

  auto getTreeMapIter = m_getTreeFromEnsembleMap.find(op);
  assert(getTreeMapIter == m_getTreeFromEnsembleMap.end());
  m_getTreeFromEnsembleMap[op] = {correctedTreeIndex.getResult(),
                                  cacheInfoIter->second.thresholdMemref,
                                  cacheInfoIter->second.featureIndexMemref};
}

mlir::Value ReorgForestRepresentation::GetTreeIndex(Value tree) {
  auto *definingOp = tree.getDefiningOp();
  auto getTreeMapIter = m_getTreeFromEnsembleMap.find(definingOp);
  if (getTreeMapIter == m_getTreeFromEnsembleMap.end())
    return ::GetTreeIndexValue(tree);
  else
    return getTreeMapIter->second.correctedIndex;
}

ReorgForestRepresentation::CacheGlobalValues
ReorgForestRepresentation::AddTreeCacheGlobals(
    ConversionPatternRewriter &rewriter, Operation *op, int32_t bufferLen,
    int64_t numTreesToCache) {
  auto location = op->getLoc();
  auto cacheTreesOp =
      AssertOpIsOfType<decisionforest::CacheTreesFromEnsembleOp>(op);
  int64_t cacheId = cacheTreesOp.getCacheID();
  auto owningModule = op->getParentOfType<mlir::ModuleOp>();
  assert(owningModule);
  std::string thresholdCacheBufferName =
      std::string("treeCache_thresholds_") + std::to_string(cacheId);
  std::string featureIndexCacheBufferName =
      std::string("treeCache_featureIndices_") + std::to_string(cacheId);

  auto thresholdCacheElemType = decisionforest::ReorgMemrefElementType::get(
      m_thresholdType, numTreesToCache);
  auto featureIndexCacheElemType = decisionforest::ReorgMemrefElementType::get(
      m_featureIndexType, numTreesToCache);
  auto thresholdCacheBufferType =
      MemRefType::get({bufferLen}, thresholdCacheElemType, {}, // Affine map
                      3); // Address space ID -- shared memory
  auto featureIndexCacheBufferType =
      MemRefType::get({bufferLen}, featureIndexCacheElemType, {}, // Affine map
                      3); // Address space ID -- shared memory

  if (m_cacheBufferNamesMap.find(cacheId) == m_cacheBufferNamesMap.end()) {
    SaveAndRestoreInsertionPoint saveAndRestoreInsertPoint(rewriter);
    rewriter.setInsertionPoint(&owningModule.front());
    rewriter.create<memref::GlobalOp>(
        location, thresholdCacheBufferName,
        /*sym_visibility=*/rewriter.getStringAttr("private"),
        /*type=*/thresholdCacheBufferType,
        /*initial_value=*/rewriter.getUnitAttr(),
        /*constant=*/false,
        /*alignment*/ IntegerAttr());
    rewriter.create<memref::GlobalOp>(
        location, featureIndexCacheBufferName,
        /*sym_visibility=*/rewriter.getStringAttr("private"),
        /*type=*/featureIndexCacheBufferType,
        /*initial_value=*/rewriter.getUnitAttr(),
        /*constant=*/false,
        /*alignment*/ IntegerAttr());
    m_cacheBufferNamesMap[cacheId] = thresholdCacheBufferName;
  }
  auto getThresholdCache = rewriter.create<memref::GetGlobalOp>(
      location, thresholdCacheBufferType, thresholdCacheBufferName);
  auto getFeatureIndexCache = rewriter.create<memref::GetGlobalOp>(
      location, featureIndexCacheBufferType, featureIndexCacheBufferName);
  return CacheGlobalValues{getThresholdCache.getResult(),
                           getFeatureIndexCache.getResult()};
}

void ReorgForestRepresentation::LowerCacheTreeOp(
    ConversionPatternRewriter &rewriter, mlir::Operation *op,
    ArrayRef<Value> operands,
    std::shared_ptr<decisionforest::IModelSerializer> m_serializer) {

  auto location = op->getLoc();
  auto cacheTreesOp =
      AssertOpIsOfType<decisionforest::CacheTreesFromEnsembleOp>(op);
  auto ensembleValue = cacheTreesOp.getForest();
  auto ensembleConst = AssertOpIsOfType<decisionforest::EnsembleConstantOp>(
      ensembleValue.getDefiningOp());
  auto forestType =
      ensembleValue.getType().cast<decisionforest::TreeEnsembleType>();
  assert(forestType.doAllTreesHaveSameType() &&
         forestType.doAllTreesHaveSameTileSize());

  // All threads need to be synchronized before we can start caching
  // rewriter.create<gpu::BarrierOp>(location);
  rewriter.create<decisionforest::CacheOpBeginOp>(location);

  // TODO_Ashwin This is a BUG!! We may not always have a for loop surrounding a
  // cache operation The surrounding loops for grid and thread block have
  // already been lowered to gpu.launch!!
  int64_t numTreesToCache = GetNumberOfTreesToCache(cacheTreesOp);

  // Create globals for threshold and feature indices
  //    - Size to be determined from the forest?
  auto bufferLen = ComputeBufferSize(
      ensembleConst.getForest().GetDecisionForest(), numTreesToCache);
  auto numNodesPerTree = bufferLen / numTreesToCache;
  assert(bufferLen % numTreesToCache == 0);
  auto cacheGlobals =
      AddTreeCacheGlobals(rewriter, op, bufferLen, numTreesToCache);

  // Get the number of threads in the thread block
  auto owningGPULaunchOp = cacheTreesOp->getParentOfType<gpu::LaunchOp>();
  assert(owningGPULaunchOp);
  auto numThreadsX = owningGPULaunchOp.getBlockSizeX();
  auto numThreadsY = owningGPULaunchOp.getBlockSizeY();
  auto threadNum = owningGPULaunchOp.getThreadIds();

  // TODO_Ashwin everything below assumes that thread blocks are 2D!
  rewriter.create<arith::MulIOp>(location, numThreadsX, numThreadsY);

  // index = numThreadsX*threadNum.Y + threadNum.X
  auto nxTimesTy =
      rewriter.create<arith::MulIOp>(location, numThreadsX, threadNum.y);
  auto threadIndex = rewriter.create<arith::AddIOp>(
      location, static_cast<Value>(nxTimesTy), threadNum.x);

  auto numTreesToCacheConst =
      rewriter.create<arith::ConstantIndexOp>(location, numTreesToCache);
  auto numTreesConst =
      rewriter.create<arith::ConstantIndexOp>(location, m_numTrees);
  auto inRange = rewriter.create<arith::CmpIOp>(
      location, arith::CmpIPredicate::slt, threadIndex.getResult(),
      numTreesToCacheConst.getResult());
  auto ifInRange = rewriter.create<scf::IfOp>(location, TypeRange{},
                                              inRange.getResult(), false);
  /*
  if (tid >= numTreesToCache)
    return
  for i=0 : numNodes
    threadIndex = (i*numTrees) + startTree + tid
    cacheIndex = (i*numTreesToCache) + tid
    cache[cacheIndex] = buffer[threadIndex]
  */
  {
    auto thenBodyBuilder = ifInRange.getThenBodyBuilder();
    auto zeroIndexConst =
        thenBodyBuilder.create<arith::ConstantIndexOp>(location, 0);
    auto oneIndexConst =
        thenBodyBuilder.create<arith::ConstantIndexOp>(location, 1);
    auto numNodesConst = thenBodyBuilder.create<arith::ConstantIndexOp>(
        location, numNodesPerTree);
    auto forLoop = thenBodyBuilder.create<scf::ForOp>(
        location, static_cast<Value>(zeroIndexConst),
        static_cast<Value>(numNodesConst), static_cast<Value>(oneIndexConst));
    thenBodyBuilder.setInsertionPointToStart(forLoop.getBody());

    auto i = forLoop.getInductionVar();
    auto iTimesNumTrees = thenBodyBuilder.create<arith::MulIOp>(
        location, i, numTreesConst.getResult());
    auto iXnumTreesPlusStartTree = thenBodyBuilder.create<arith::AddIOp>(
        location, iTimesNumTrees.getResult(), cacheTreesOp.getStartTreeIndex());
    auto globalBufferIndex = thenBodyBuilder.create<arith::AddIOp>(
        location, iXnumTreesPlusStartTree.getResult(), threadIndex.getResult());

    auto iTimesNumTreesToCache = thenBodyBuilder.create<arith::MulIOp>(
        location, i, numTreesToCacheConst.getResult());
    auto cacheIndex = thenBodyBuilder.create<arith::AddIOp>(
        location, iTimesNumTreesToCache.getResult(), threadIndex.getResult());

    auto thresholdVal = thenBodyBuilder.create<memref::LoadOp>(
        location, m_thresholdMemref, ValueRange{globalBufferIndex});
    auto featureIndexVal = thenBodyBuilder.create<memref::LoadOp>(
        location, m_featureIndexMemref, ValueRange{globalBufferIndex});

    auto targetThresholdType = cacheGlobals.thresholdCache.getType()
                                   .cast<MemRefType>()
                                   .getElementType();
    auto castedThreshold = thenBodyBuilder.create<UnrealizedConversionCastOp>(
        location, targetThresholdType, thresholdVal.getResult());
    thenBodyBuilder.create<memref::StoreOp>(
        location, castedThreshold.getResult(0), cacheGlobals.thresholdCache,
        ValueRange{cacheIndex});

    auto targetFeatureIndexType = cacheGlobals.featureIndexCache.getType()
                                      .cast<MemRefType>()
                                      .getElementType();
    auto castedFeatureIndex =
        thenBodyBuilder.create<UnrealizedConversionCastOp>(
            location, targetFeatureIndexType, featureIndexVal.getResult());
    thenBodyBuilder.create<memref::StoreOp>(
        location, castedFeatureIndex.getResult(0),
        cacheGlobals.featureIndexCache, ValueRange{cacheIndex});
  }
  // rewriter.create<gpu::BarrierOp>(location);
  rewriter.create<decisionforest::CacheOpEndOp>(location);

  // Add the information about this cache to the map
  m_treeCacheMap[op] = {cacheGlobals.thresholdCache,
                        cacheGlobals.featureIndexCache, numTreesToCache};
}

// Definition in GPURepresentations.cpp
void LowerCacheRowsOpToGPU(ConversionPatternRewriter &rewriter,
                           mlir::Operation *op, ArrayRef<Value> operands,
                           std::map<int64_t, std::string> &cacheBufferNamesMap);

void ReorgForestRepresentation::LowerCacheRowsOp(
    ConversionPatternRewriter &rewriter, mlir::Operation *op,
    ArrayRef<Value> operands) {
  LowerCacheRowsOpToGPU(rewriter, op, operands, m_cacheBufferNamesMap);
}

std::shared_ptr<IRepresentation> ConstructGPUReorgForestRepresentation() {
  return std::make_shared<ReorgForestRepresentation>();
}

REGISTER_REPRESENTATION(gpu_reorg, ConstructGPUReorgForestRepresentation)

} // namespace decisionforest
} // namespace mlir

#endif // TREEBEARD_GPU_SUPPORT
