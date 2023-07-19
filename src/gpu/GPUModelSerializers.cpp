#ifdef TREEBEARD_GPU_SUPPORT

#include "ExecutionHelpers.h"
#include "TreeTilingDescriptor.h"
#include "TreeTilingUtils.h"
#include "TiledTree.h"
#include "Logger.h"
#include "ModelSerializers.h"
#include "GPUModelSerializers.h"
#include <cstdint>

namespace mlir
{
namespace decisionforest
{

// ===---------------------------------------------------=== //
// GPUArraySparseSerializerBase Methods
// ===---------------------------------------------------=== //

int32_t GPUArraySparseSerializerBase::InitializeLengthsArray() {
  using InitLengthFunc_t = LengthMemrefType (*)(int64_t *, int64_t *, int64_t, int64_t, int64_t);
  auto initLengthPtr = GetFunctionAddress<InitLengthFunc_t>("Init_Lengths");
  auto tileSize = m_inferenceRunner->GetTileSize();
  auto thresholdSize = m_inferenceRunner->GetThresholdWidth();
  auto featureIndexSize = m_inferenceRunner->GetFeatureIndexWidth();
  std::vector<int64_t> lengths(mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfTrees(), -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengths.data(), 
                                                                               tileSize,
                                                                               thresholdSize,
                                                                               featureIndexSize);
  LengthMemrefType lengthsMemref{lengths.data(), lengths.data(), 0, {static_cast<int64_t>(lengths.size())}, {1}};
  m_lengthsMemref = initLengthPtr(lengthsMemref.bufferPtr, lengthsMemref.alignedPtr, lengthsMemref.offset, lengthsMemref.lengths[0], lengthsMemref.strides[0]);

  return 0;
}

// TODO All the initialize methods are doing the same thing, except that the getter they're calling are different. 
// Refactor them into a shared method.
int32_t GPUArraySparseSerializerBase::InitializeOffsetsArray() {
  using InitLengthFunc_t = LengthMemrefType (*)(int64_t *, int64_t *, int64_t, int64_t, int64_t);
  auto initOffsetPtr = GetFunctionAddress<InitLengthFunc_t>("Init_Offsets");
  auto tileSize = m_inferenceRunner->GetTileSize();
  auto thresholdSize = m_inferenceRunner->GetThresholdWidth();
  auto featureIndexSize = m_inferenceRunner->GetFeatureIndexWidth();
  
  std::vector<int64_t> offsets(mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfTrees(), -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsets.data(),
                                                                               tileSize,
                                                                               thresholdSize,
                                                                               featureIndexSize);
  LengthMemrefType offsetsMemref{offsets.data(), offsets.data(), 0, {static_cast<int64_t>(offsets.size())}, {1}};
  m_offsetsMemref = initOffsetPtr(offsetsMemref.bufferPtr,
                                  offsetsMemref.alignedPtr,
                                  offsetsMemref.offset,
                                  offsetsMemref.lengths[0],
                                  offsetsMemref.strides[0]);

  return 0;
}

template<typename ThresholdType, typename FeatureIndexType, typename TileShapeType>
int32_t GPUArraySparseSerializerBase::ResolveChildIndexType() {
  if (!m_sparseRepresentation)
    return CallInitMethod<ThresholdType, FeatureIndexType, TileShapeType, int32_t>();
  else {
    auto childIndexBitWidth = mlir::decisionforest::ForestJSONReader::GetInstance().GetChildIndexBitWidth();
    assert (childIndexBitWidth > 0);
    if (childIndexBitWidth == 8)
      return CallInitMethod<ThresholdType, FeatureIndexType, TileShapeType, int8_t>();
    else if (childIndexBitWidth == 16)
      return CallInitMethod<ThresholdType, FeatureIndexType, TileShapeType, int16_t>();
    else if (childIndexBitWidth == 32)
      return CallInitMethod<ThresholdType, FeatureIndexType, TileShapeType, int32_t>();
    else
      assert (false && "Unsupported child index bit width");
    return -1;
  }
}

template<typename ThresholdType, typename FeatureIndexType>
int32_t GPUArraySparseSerializerBase::ResolveTileShapeType() {
  auto tileShapeBitWidth = mlir::decisionforest::ForestJSONReader::GetInstance().GetTileShapeBitWidth();
  if (tileShapeBitWidth == 8)
    return ResolveChildIndexType<ThresholdType, FeatureIndexType, int8_t>();
  else if (tileShapeBitWidth == 16)
    return ResolveChildIndexType<ThresholdType, FeatureIndexType, int16_t>();
  else if (tileShapeBitWidth == 32)
    return ResolveChildIndexType<ThresholdType, FeatureIndexType, int32_t>();
  else
    assert (false && "Unsupported tile shape bit width");
  return -1;
}

template<typename ThresholdType, typename FeatureIndexType, typename TileShapeType, typename ChildIndexType>
int32_t GPUArraySparseSerializerBase::CallInitMethod() {
  auto tileSize = m_inferenceRunner->GetTileSize();
  auto thresholdSize = m_inferenceRunner->GetThresholdWidth();
  auto featureIndexSize = m_inferenceRunner->GetFeatureIndexWidth();
  std::vector<ThresholdType> thresholds;
  std::vector<FeatureIndexType> featureIndices;
  std::vector<TileShapeType> tileShapeIDs;
  std::vector<ChildIndexType> childIndices;

  if (!m_sparseRepresentation)
    mlir::decisionforest::ForestJSONReader::GetInstance().GetModelValues(tileSize,
                                                                         thresholdSize,
                                                                         featureIndexSize,
                                                                         thresholds,
                                                                         featureIndices,
                                                                         tileShapeIDs);
  else {
    assert (m_sparseRepresentation);
    mlir::decisionforest::ForestJSONReader::GetInstance().GetModelValues(tileSize, 
                                                                         thresholdSize, 
                                                                         featureIndexSize, 
                                                                         thresholds,
                                                                         featureIndices,
                                                                         tileShapeIDs,
                                                                         childIndices);
  }

  Memref<ThresholdType, 1> thresholdsMemref{thresholds.data(), thresholds.data(), 0, {(int64_t)thresholds.size()}, 1};
  Memref<FeatureIndexType, 1> featureIndexMemref{featureIndices.data(), featureIndices.data(), 0, {(int64_t)featureIndices.size()}, 1};
  Memref<TileShapeType, 1> tileShapeIDMemref{tileShapeIDs.data(), tileShapeIDs.data(), 0, {(int64_t)tileShapeIDs.size()}, 1};
  
  if (!m_sparseRepresentation) {
    using InitModelPtr_t = ModelMemrefType (*)(ThresholdType *, ThresholdType *, int64_t, int64_t, int64_t, FeatureIndexType *, FeatureIndexType *, int64_t, int64_t, int64_t, TileShapeType *, TileShapeType *, int64_t, int64_t, int64_t);
    auto initModelPtr = GetFunctionAddress<InitModelPtr_t>("Init_Model");

    m_modelMemref = initModelPtr(thresholdsMemref.bufferPtr, thresholdsMemref.alignedPtr, thresholdsMemref.offset, thresholdsMemref.lengths[0], thresholdsMemref.strides[0],
                featureIndexMemref.bufferPtr, featureIndexMemref.alignedPtr, featureIndexMemref.offset, featureIndexMemref.lengths[0], featureIndexMemref.strides[0],
                tileShapeIDMemref.bufferPtr, tileShapeIDMemref.alignedPtr, tileShapeIDMemref.offset, tileShapeIDMemref.lengths[0], tileShapeIDMemref.strides[0]);
  }
  else {
    assert (m_sparseRepresentation);
    Memref<ChildIndexType, 1> childIndexMemref{childIndices.data(), childIndices.data(), 0, {(int64_t)childIndices.size()}, 1};
    using InitModelPtr_t = ModelMemrefType (*)(ThresholdType *, ThresholdType *, int64_t, int64_t, int64_t, FeatureIndexType *, FeatureIndexType *, int64_t, int64_t, int64_t, TileShapeType *, TileShapeType *, int64_t, int64_t, int64_t, ChildIndexType *, ChildIndexType *, int64_t, int64_t, int64_t);
    auto initModelPtr = GetFunctionAddress<InitModelPtr_t>("Init_Model");

    m_modelMemref = initModelPtr(thresholdsMemref.bufferPtr, thresholdsMemref.alignedPtr, thresholdsMemref.offset, thresholdsMemref.lengths[0], thresholdsMemref.strides[0],
                featureIndexMemref.bufferPtr, featureIndexMemref.alignedPtr, featureIndexMemref.offset, featureIndexMemref.lengths[0], featureIndexMemref.strides[0],
                tileShapeIDMemref.bufferPtr, tileShapeIDMemref.alignedPtr, tileShapeIDMemref.offset, tileShapeIDMemref.lengths[0], tileShapeIDMemref.strides[0],
                childIndexMemref.bufferPtr, childIndexMemref.alignedPtr, childIndexMemref.offset, childIndexMemref.lengths[0], childIndexMemref.strides[0]);
  }
  // TODO We expect to log the size of model memref etc here. But we can't do it yet 
  // for GPU code because we can't 
  // if (TreeBeard::Logging::loggingOptions.logGenCodeStats) {
  //   TreeBeard::Logging::Log("Model memref size : " + std::to_string(returnValue));
  //   std::set<int32_t> tileShapes(tileShapeIDs.begin(), tileShapeIDs.end());
  //   TreeBeard::Logging::Log("Number of unique tile shapes : " + std::to_string(tileShapes.size()));
  // }
  return 0;
}

int32_t GPUArraySparseSerializerBase::InitializeModelArray() {
  // TODO The ForestJSONReader class needs to provide an interface to iterate over tile sizes and bit widths
  // We need to construct the name of the getter function based on those.
  auto thresholdSize = m_inferenceRunner->GetThresholdWidth();
  auto featureIndexSize = m_inferenceRunner->GetFeatureIndexWidth();
  if (thresholdSize == 64) {
    if (featureIndexSize == 32) {
      return ResolveTileShapeType<double, int32_t>();
    }
    if (featureIndexSize == 16) {
      return ResolveTileShapeType<double, int16_t>();
    }
    if (featureIndexSize == 8) {
      return ResolveTileShapeType<double, int8_t>();
    }
    assert (false);
   
  }
  else if (thresholdSize == 32) {
    if (featureIndexSize == 32) {
      return ResolveTileShapeType<float, int32_t>();
    }
    if (featureIndexSize == 16) {
      return ResolveTileShapeType<float, int16_t>();
    }
    if (featureIndexSize == 8) {
      return ResolveTileShapeType<float, int8_t>();
    }
    assert (false);
   
  }
  else {
    assert (false);
  }
  return 0;
}

void GPUArraySparseSerializerBase::InitializeClassInformation() {
  if (mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfClasses() == 0) return;

  std::vector<int8_t> classIds(mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfTrees(), -1);
  auto tileSize = m_inferenceRunner->GetTileSize();
  auto thresholdSize = m_inferenceRunner->GetThresholdWidth();
  auto featureIndexSize = m_inferenceRunner->GetFeatureIndexWidth();
  mlir::decisionforest::ForestJSONReader::GetInstance()
      .InitializeClassInformation((void *)classIds.data(), tileSize,
                                  thresholdSize, featureIndexSize);

  using InitClassIdsFunc_t = ClassMemrefType (*)(int8_t *, int8_t *, int64_t, int64_t, int64_t);

  auto initClassInfoFuncPtr = GetFunctionAddress<InitClassIdsFunc_t>("Init_ClassIds");
  m_classIDMemref = initClassInfoFuncPtr(classIds.data(), classIds.data(), (int64_t)0, (int64_t)classIds.size(), (int64_t)1);
}

void GPUArraySparseSerializerBase::CallPredictionMethod(void* predictFuncPtr,
                                                        Memref<double, 2> inputs,
                                                        Memref<double, 1> results) {
    using InputElementType = double;
    using ReturnType = double;

    using InferenceFunc_t = Memref<ReturnType, 1> (*)(
        InputElementType *, InputElementType *, int64_t, int64_t, int64_t,
        int64_t, int64_t, ReturnType *, ReturnType *, int64_t, int64_t, int64_t,
        Tile *, Tile *, int64_t, int64_t, int64_t, int64_t *, int64_t *,
        int64_t, int64_t, int64_t, int64_t *, int64_t *, int64_t, int64_t,
        int64_t, int8_t *, int8_t *, int64_t, int64_t, int64_t);
    auto inferenceFuncPtr = reinterpret_cast<InferenceFunc_t>(predictFuncPtr);
    inferenceFuncPtr(inputs.bufferPtr, inputs.alignedPtr, inputs.offset, inputs.lengths[0], inputs.lengths[1], inputs.strides[0], inputs.strides[1],
                     results.bufferPtr, results.alignedPtr, results.offset, results.lengths[0], results.strides[0],
                     m_modelMemref.bufferPtr, m_modelMemref.alignedPtr, m_modelMemref.offset, m_modelMemref.lengths[0], m_modelMemref.strides[0],
                     m_offsetsMemref.bufferPtr, m_offsetsMemref.alignedPtr, m_offsetsMemref.offset, m_offsetsMemref.lengths[0], m_offsetsMemref.strides[0],
                     m_lengthsMemref.bufferPtr, m_lengthsMemref.alignedPtr, m_lengthsMemref.offset, m_lengthsMemref.lengths[0], m_lengthsMemref.strides[0],
                     m_classIDMemref.bufferPtr, m_classIDMemref.alignedPtr, m_classIDMemref.offset, m_classIDMemref.lengths[0], m_classIDMemref.strides[0]);
}    

void GPUArraySparseSerializerBase::CleanupBuffers() {

  if (mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfClasses() > 0) {
    using CleanupFunc_t = int32_t (*)(
        Tile *, Tile *, int64_t, int64_t, int64_t, int64_t *, int64_t *, int64_t,
        int64_t, int64_t, int64_t *, int64_t *, int64_t, int64_t, int64_t,
        int8_t*, int8_t*, int64_t, int64_t, int64_t);

    auto cleanupFuncPtr = this->GetFunctionAddress<CleanupFunc_t>("Dealloc_Buffers");
    cleanupFuncPtr(m_modelMemref.bufferPtr, m_modelMemref.alignedPtr, m_modelMemref.offset, m_modelMemref.lengths[0], m_modelMemref.strides[0],
                  m_offsetsMemref.bufferPtr, m_offsetsMemref.alignedPtr, m_offsetsMemref.offset, m_offsetsMemref.lengths[0], m_offsetsMemref.strides[0],
                  m_lengthsMemref.bufferPtr, m_lengthsMemref.alignedPtr, m_lengthsMemref.offset, m_lengthsMemref.lengths[0], m_lengthsMemref.strides[0],
                  m_classIDMemref.bufferPtr, m_classIDMemref.alignedPtr, m_classIDMemref.offset, m_classIDMemref.lengths[0], m_classIDMemref.strides[0]);
  }
  else {
      using CleanupFunc_t = int32_t (*)(
      Tile *, Tile *, int64_t, int64_t, int64_t, int64_t *, int64_t *, int64_t,
      int64_t, int64_t, int64_t *, int64_t *, int64_t, int64_t, int64_t);

    auto cleanupFuncPtr =
        this->GetFunctionAddress<CleanupFunc_t>("Dealloc_Buffers");
    cleanupFuncPtr(m_modelMemref.bufferPtr, m_modelMemref.alignedPtr,
                   m_modelMemref.offset, m_modelMemref.lengths[0],
                   m_modelMemref.strides[0], m_offsetsMemref.bufferPtr,
                   m_offsetsMemref.alignedPtr, m_offsetsMemref.offset,
                   m_offsetsMemref.lengths[0], m_offsetsMemref.strides[0],
                   m_lengthsMemref.bufferPtr, m_lengthsMemref.alignedPtr,
                   m_lengthsMemref.offset, m_lengthsMemref.lengths[0],
                   m_lengthsMemref.strides[0]);
  }

}

void GPUArraySparseSerializerBase::ReadData() {
    decisionforest::ForestJSONReader::GetInstance().SetFilePath(m_filepath);
    decisionforest::ForestJSONReader::GetInstance().ParseJSONFile();
}

// ===---------------------------------------------------=== //
// Persistence Helper Methods
// ===---------------------------------------------------=== //

void PersistDecisionForestArrayBased(mlir::decisionforest::DecisionForest& forest, mlir::decisionforest::TreeEnsembleType forestType);
void PersistDecisionForestSparse(mlir::decisionforest::DecisionForest& forest, mlir::decisionforest::TreeEnsembleType forestType);

// ===---------------------------------------------------=== //
// GPUSparseRepresentationSerializer Methods
// ===---------------------------------------------------=== //

void GPUSparseRepresentationSerializer::Persist(mlir::decisionforest::DecisionForest& forest, mlir::decisionforest::TreeEnsembleType forestType) {
    mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(m_filepath);
    PersistDecisionForestSparse(forest, forestType);
}

template<typename ThresholdType>
LeafValueMemref GPUSparseRepresentationSerializer::InitLeafValues(int32_t tileSize, int32_t thresholdBitWidth, int32_t featureIndexBitWidth) {
  int32_t numLeaves = mlir::decisionforest::ForestJSONReader::GetInstance().GetTotalNumberOfLeaves();
  std::vector<ThresholdType> leafVals(numLeaves);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLeaves(reinterpret_cast<void*>(leafVals.data()), 
                                                                         tileSize,
                                                                         thresholdBitWidth,
                                                                         featureIndexBitWidth);
  
  using InitLeafValuesFunc_t = LeafValueMemref (*)(ThresholdType*, ThresholdType*, int64_t, int64_t, int64_t);

  auto initLeafValsFuncPtr = GetFunctionAddress<InitLeafValuesFunc_t>("Init_Leaves");
  auto leafValsMemref = initLeafValsFuncPtr(leafVals.data(), leafVals.data(), (int64_t)0, (int64_t)leafVals.size(), (int64_t)1);
  return leafValsMemref;
}

int32_t GPUSparseRepresentationSerializer::InitializeLeafValues(){
  auto tileSize = m_inferenceRunner->GetTileSize();
  auto thresholdSize = m_inferenceRunner->GetThresholdWidth();
  auto featureIndexSize = m_inferenceRunner->GetFeatureIndexWidth();
  if (thresholdSize == 32)
    m_leafValues = InitLeafValues<float>(tileSize, thresholdSize, featureIndexSize);
  else if(thresholdSize == 64)
    m_leafValues = InitLeafValues<double>(tileSize, thresholdSize, featureIndexSize);
  else
    assert (false && "Unsupport threshold type");
  return 0;
}

int32_t GPUSparseRepresentationSerializer::InitializeLeafLengths(){
  std::vector<int64_t> leafLengths(mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfTrees(), -1);
  auto tileSize = m_inferenceRunner->GetTileSize();
  auto thresholdSize = m_inferenceRunner->GetThresholdWidth();
  auto featureIndexSize = m_inferenceRunner->GetFeatureIndexWidth();
  mlir::decisionforest::ForestJSONReader::GetInstance()
      .InitializeLeavesLengthBuffer((void *)leafLengths.data(), tileSize,
                                    thresholdSize, featureIndexSize);

  using InitLeafLengthsFunc_t = OffsetMemrefType (*)(int64_t *, int64_t *, int64_t, int64_t, int64_t);

  auto initLeafLengthFuncPtr = GetFunctionAddress<InitLeafLengthsFunc_t>("Init_LeafLengths");
  m_leafLengthsMemref = initLeafLengthFuncPtr(leafLengths.data(), leafLengths.data(), (int64_t)0, (int64_t)leafLengths.size(), (int64_t)1);
  return 0;
}

int32_t GPUSparseRepresentationSerializer::InitializeLeafOffsets(){
  std::vector<int64_t> leafOffsets(mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfTrees(), -1);
  auto tileSize = m_inferenceRunner->GetTileSize();
  auto thresholdSize = m_inferenceRunner->GetThresholdWidth();
  auto featureIndexSize = m_inferenceRunner->GetFeatureIndexWidth();
  mlir::decisionforest::ForestJSONReader::GetInstance()
      .InitializeLeavesOffsetBuffer((void *)leafOffsets.data(), tileSize,
                                    thresholdSize, featureIndexSize);

  using InitLeafOffsetsFunc_t = OffsetMemrefType (*)(int64_t *, int64_t *, int64_t, int64_t, int64_t);

  auto initLeafOffsetsFuncPtr = GetFunctionAddress<InitLeafOffsetsFunc_t>("Init_LeafOffsets");
  m_leafOffsetsMemref = initLeafOffsetsFuncPtr(leafOffsets.data(), leafOffsets.data(), (int64_t)0, (int64_t)leafOffsets.size(), (int64_t)1);
  return 0;
}


int32_t GPUSparseRepresentationSerializer::InitializeLeafArrays() {
  auto tileSize = m_inferenceRunner->GetTileSize();
  if (tileSize == 1)
    return 0;
  InitializeLeafValues();
  InitializeLeafLengths();
  InitializeLeafOffsets();
  return 0;
}

void GPUSparseRepresentationSerializer::InitializeBuffersImpl() {
    InitializeLengthsArray();
    InitializeOffsetsArray();
    InitializeModelArray();
    InitializeClassInformation();
    InitializeLeafArrays();
}

void GPUSparseRepresentationSerializer::CallPredictionMethod(void* predictFuncPtr,
                                                             Memref<double, 2> inputs,
                                                             Memref<double, 1> results) {
    using InputElementType = double;
    using ReturnType = double;

    auto tileSize = m_inferenceRunner->GetTileSize();
    if (tileSize == 1) {
      GPUArraySparseSerializerBase::CallPredictionMethod(predictFuncPtr, inputs, results);
      return;
    }
    
    assert (m_sparseRepresentation);
    using InferenceFunc_t = Memref<ReturnType, 1> (*)(
        InputElementType *, InputElementType *, int64_t, int64_t, int64_t, int64_t, int64_t, // Input data
        ReturnType *, ReturnType *, int64_t, int64_t, int64_t, // Return values
        Tile *, Tile *, int64_t, int64_t, int64_t, // Model memref
        int64_t *, int64_t *, int64_t, int64_t, int64_t, // Tree offsets
        int64_t *, int64_t *, int64_t, int64_t, int64_t, // Tree lengths
        int8_t *, int8_t *, int64_t, int64_t, int64_t,    // class IDs
        InputElementType *, InputElementType *, int64_t, int64_t, int64_t, // Leaf values
        int64_t *, int64_t *, int64_t, int64_t, int64_t, // Leaf offsets
        int64_t *, int64_t *, int64_t, int64_t, int64_t, // Leaf lengths
        int8_t *, int8_t *, int64_t, int64_t, int64_t, int64_t, int64_t // LUT
        );
    auto lutMemref = m_inferenceRunner->GetLUTMemref();
    auto inferenceFuncPtr = reinterpret_cast<InferenceFunc_t>(predictFuncPtr);
    inferenceFuncPtr(inputs.bufferPtr, inputs.alignedPtr, inputs.offset, inputs.lengths[0], inputs.lengths[1], inputs.strides[0], inputs.strides[1],
                    results.bufferPtr, results.alignedPtr, results.offset, results.lengths[0], results.strides[0],
                    m_modelMemref.bufferPtr, m_modelMemref.alignedPtr, m_modelMemref.offset, m_modelMemref.lengths[0], m_modelMemref.strides[0],
                    m_offsetsMemref.bufferPtr, m_offsetsMemref.alignedPtr, m_offsetsMemref.offset, m_offsetsMemref.lengths[0], m_offsetsMemref.strides[0],
                    m_lengthsMemref.bufferPtr, m_lengthsMemref.alignedPtr, m_lengthsMemref.offset, m_lengthsMemref.lengths[0], m_lengthsMemref.strides[0],
                    m_classIDMemref.bufferPtr, m_classIDMemref.alignedPtr, m_classIDMemref.offset, m_classIDMemref.lengths[0], m_classIDMemref.strides[0],
                    m_leafValues.bufferPtr, m_leafValues.alignedPtr, m_leafValues.offset, m_leafValues.lengths[0], m_leafValues.strides[0],
                    m_leafOffsetsMemref.bufferPtr, m_leafOffsetsMemref.alignedPtr, m_leafOffsetsMemref.offset, m_leafOffsetsMemref.lengths[0], m_leafOffsetsMemref.strides[0],
                    m_leafLengthsMemref.bufferPtr, m_leafLengthsMemref.alignedPtr, m_leafLengthsMemref.offset, m_leafLengthsMemref.lengths[0], m_leafLengthsMemref.strides[0],
                    lutMemref.bufferPtr, lutMemref.alignedPtr, lutMemref.offset, lutMemref.lengths[0], lutMemref.lengths[1], lutMemref.strides[0], lutMemref.strides[1]);
}

void GPUSparseRepresentationSerializer::CleanupBuffers() {

  auto tileSize = m_inferenceRunner->GetTileSize();
  // For non tiled case, use the shared code since there are no leaf arrays
  if (tileSize == 1) {
    GPUArraySparseSerializerBase::CleanupBuffers();
    return;
  }

  if (mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfClasses() > 0) {
    using CleanupFunc_t = int32_t (*)(
        Tile *, Tile *, int64_t, int64_t, int64_t, // Model
        int64_t *, int64_t *, int64_t, int64_t, int64_t, // Offsets
        int64_t *, int64_t *, int64_t, int64_t, int64_t, // Lengths
        int8_t*, int8_t*, int64_t, int64_t, int64_t, // classIDs
        ThresholdType*, ThresholdType*, int64_t, int64_t, int64_t, // Leaf values
        int64_t *, int64_t *, int64_t, int64_t, int64_t, // Leaf offsets
        int64_t *, int64_t *, int64_t, int64_t, int64_t // Leaf lengths
        );

    auto cleanupFuncPtr = this->GetFunctionAddress<CleanupFunc_t>("Dealloc_Buffers");
    cleanupFuncPtr(m_modelMemref.bufferPtr, m_modelMemref.alignedPtr, m_modelMemref.offset, m_modelMemref.lengths[0], m_modelMemref.strides[0],
                  m_offsetsMemref.bufferPtr, m_offsetsMemref.alignedPtr, m_offsetsMemref.offset, m_offsetsMemref.lengths[0], m_offsetsMemref.strides[0],
                  m_lengthsMemref.bufferPtr, m_lengthsMemref.alignedPtr, m_lengthsMemref.offset, m_lengthsMemref.lengths[0], m_lengthsMemref.strides[0],
                  m_classIDMemref.bufferPtr, m_classIDMemref.alignedPtr, m_classIDMemref.offset, m_classIDMemref.lengths[0], m_classIDMemref.strides[0],
                  m_leafValues.bufferPtr, m_leafValues.alignedPtr, m_leafValues.offset, m_leafValues.lengths[0], m_leafValues.strides[0],
                  m_leafOffsetsMemref.bufferPtr, m_leafOffsetsMemref.alignedPtr, m_leafOffsetsMemref.offset, m_leafOffsetsMemref.lengths[0], m_leafOffsetsMemref.strides[0],
                  m_leafLengthsMemref.bufferPtr, m_leafLengthsMemref.alignedPtr, m_leafLengthsMemref.offset, m_leafLengthsMemref.lengths[0], m_leafLengthsMemref.strides[0]);
  }
  else {
    // No classIDs buffer unless we're doing multi-class classification
    using CleanupFunc_t = int32_t (*)(
        Tile *, Tile *, int64_t, int64_t, int64_t, // Model
        int64_t *, int64_t *, int64_t, int64_t, int64_t, // Offsets
        int64_t *, int64_t *, int64_t, int64_t, int64_t, // Lengths
        ThresholdType*, ThresholdType*, int64_t, int64_t, int64_t, // Leaf values
        int64_t *, int64_t *, int64_t, int64_t, int64_t, // Leaf offsets
        int64_t *, int64_t *, int64_t, int64_t, int64_t // Leaf lengths
        );

    auto cleanupFuncPtr = this->GetFunctionAddress<CleanupFunc_t>("Dealloc_Buffers");
    cleanupFuncPtr(m_modelMemref.bufferPtr, m_modelMemref.alignedPtr, m_modelMemref.offset, m_modelMemref.lengths[0], m_modelMemref.strides[0],
                  m_offsetsMemref.bufferPtr, m_offsetsMemref.alignedPtr, m_offsetsMemref.offset, m_offsetsMemref.lengths[0], m_offsetsMemref.strides[0],
                  m_lengthsMemref.bufferPtr, m_lengthsMemref.alignedPtr, m_lengthsMemref.offset, m_lengthsMemref.lengths[0], m_lengthsMemref.strides[0],
                  m_leafValues.bufferPtr, m_leafValues.alignedPtr, m_leafValues.offset, m_leafValues.lengths[0], m_leafValues.strides[0],
                  m_leafOffsetsMemref.bufferPtr, m_leafOffsetsMemref.alignedPtr, m_leafOffsetsMemref.offset, m_leafOffsetsMemref.lengths[0], m_leafOffsetsMemref.strides[0],
                  m_leafLengthsMemref.bufferPtr, m_leafLengthsMemref.alignedPtr, m_leafLengthsMemref.offset, m_leafLengthsMemref.lengths[0], m_leafLengthsMemref.strides[0]);
  }
}

std::shared_ptr<IModelSerializer> ConstructGPUSparseRepresentation(const std::string& jsonFilename) {
  return std::make_shared<GPUSparseRepresentationSerializer>(jsonFilename);
}

REGISTER_SERIALIZER(gpu_sparse, ConstructGPUSparseRepresentation)

// ===---------------------------------------------------=== //
// GPUArrayRepresentationSerializer Methods
// ===---------------------------------------------------=== //

void GPUArrayRepresentationSerializer::Persist(mlir::decisionforest::DecisionForest& forest, mlir::decisionforest::TreeEnsembleType forestType) {
    mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(m_filepath);
    PersistDecisionForestArrayBased(forest, forestType);
}

void GPUArrayRepresentationSerializer::InitializeBuffersImpl() {
    InitializeLengthsArray();
    InitializeOffsetsArray();
    InitializeModelArray();
    InitializeClassInformation();
}

std::shared_ptr<IModelSerializer> ConstructGPUArrayRepresentation(const std::string& jsonFilename) {
  return std::make_shared<GPUArrayRepresentationSerializer>(jsonFilename);
}

void GPUArrayRepresentationSerializer::CallPredictionMethod(void* predictFuncPtr,
                                                            Memref<double, 2> inputs,
                                                            Memref<double, 1> results) {
    using InputElementType = double;
    using ReturnType = double;

    auto tileSize = m_inferenceRunner->GetTileSize();
    if (tileSize == 1) {
      GPUArraySparseSerializerBase::CallPredictionMethod(predictFuncPtr, inputs, results);
      return;
    }
    
    using InferenceFunc_t = Memref<ReturnType, 1> (*)(
        InputElementType *, InputElementType *, int64_t, int64_t, int64_t, int64_t, int64_t, // Input data
        ReturnType *, ReturnType *, int64_t, int64_t, int64_t, // Return values
        Tile *, Tile *, int64_t, int64_t, int64_t, // Model memref
        int64_t *, int64_t *, int64_t, int64_t, int64_t, // Tree offsets
        int64_t *, int64_t *, int64_t, int64_t, int64_t, // Tree lengths
        int8_t *, int8_t *, int64_t, int64_t, int64_t,    // class IDs
        int8_t *, int8_t *, int64_t, int64_t, int64_t, int64_t, int64_t // LUT
        );
    auto lutMemref = m_inferenceRunner->GetLUTMemref();
    auto inferenceFuncPtr = reinterpret_cast<InferenceFunc_t>(predictFuncPtr);
    inferenceFuncPtr(inputs.bufferPtr, inputs.alignedPtr, inputs.offset, inputs.lengths[0], inputs.lengths[1], inputs.strides[0], inputs.strides[1],
                    results.bufferPtr, results.alignedPtr, results.offset, results.lengths[0], results.strides[0],
                    m_modelMemref.bufferPtr, m_modelMemref.alignedPtr, m_modelMemref.offset, m_modelMemref.lengths[0], m_modelMemref.strides[0],
                    m_offsetsMemref.bufferPtr, m_offsetsMemref.alignedPtr, m_offsetsMemref.offset, m_offsetsMemref.lengths[0], m_offsetsMemref.strides[0],
                    m_lengthsMemref.bufferPtr, m_lengthsMemref.alignedPtr, m_lengthsMemref.offset, m_lengthsMemref.lengths[0], m_lengthsMemref.strides[0],
                    m_classIDMemref.bufferPtr, m_classIDMemref.alignedPtr, m_classIDMemref.offset, m_classIDMemref.lengths[0], m_classIDMemref.strides[0],
                    lutMemref.bufferPtr, lutMemref.alignedPtr, lutMemref.offset, lutMemref.lengths[0], lutMemref.lengths[1], lutMemref.strides[0], lutMemref.strides[1]);
}


REGISTER_SERIALIZER(gpu_array, ConstructGPUArrayRepresentation)

}
}

#endif // TREEBEARD_GPU_SUPPORT