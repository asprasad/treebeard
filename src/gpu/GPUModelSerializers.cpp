#include "TreeTilingDescriptor.h"
#include "TreeTilingUtils.h"
#include "TiledTree.h"
#include "Logger.h"
#include "ModelSerializers.h"
#include "GPUModelSerializers.h"

namespace mlir
{
namespace decisionforest
{

// ===---------------------------------------------------=== //
// GPUArraySparseSerializerBase Methods
// ===---------------------------------------------------=== //

int32_t GPUArraySparseSerializerBase::InitializeLengthsArray() {
  typedef LengthMemrefType (*InitLengthFunc_t)(int64_t*, int64_t*, int64_t, int64_t, int64_t);
  auto initLengthPtr = GetFunctionAddress<InitLengthFunc_t>("Init_Lengths");
  auto m_tileSize = m_inferenceRunner->GetTileSize();
  auto m_thresholdSize = m_inferenceRunner->GetThresholdWidth();
  auto m_featureIndexSize = m_inferenceRunner->GetFeatureIndexWidth();
  std::vector<int64_t> lengths(mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfTrees(), -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengths.data(), 
                                                                               m_tileSize,
                                                                               m_thresholdSize,
                                                                               m_featureIndexSize);
  LengthMemrefType lengthsMemref{lengths.data(), lengths.data(), 0, {static_cast<int64_t>(lengths.size())}, {1}};
  m_lengthsMemref = initLengthPtr(lengthsMemref.bufferPtr, lengthsMemref.alignedPtr, lengthsMemref.offset, lengthsMemref.lengths[0], lengthsMemref.strides[0]);

  return 0;
}

// TODO All the initialize methods are doing the same thing, except that the getter they're calling are different. 
// Refactor them into a shared method.
int32_t GPUArraySparseSerializerBase::InitializeOffsetsArray() {
  typedef LengthMemrefType (*InitLengthFunc_t)(int64_t*, int64_t*, int64_t, int64_t, int64_t);
  auto initOffsetPtr = GetFunctionAddress<InitLengthFunc_t>("Init_Offsets");
  auto m_tileSize = m_inferenceRunner->GetTileSize();
  auto m_thresholdSize = m_inferenceRunner->GetThresholdWidth();
  auto m_featureIndexSize = m_inferenceRunner->GetFeatureIndexWidth();
  
  std::vector<int64_t> offsets(mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfTrees(), -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsets.data(),
                                                                               m_tileSize,
                                                                               m_thresholdSize,
                                                                               m_featureIndexSize);
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
    typedef ModelMemrefType (*InitModelPtr_t)(ThresholdType*, ThresholdType*, int64_t, int64_t, int64_t, 
                                              FeatureIndexType*, FeatureIndexType*, int64_t, int64_t, int64_t,
                                              TileShapeType*, TileShapeType*, int64_t, int64_t, int64_t);
    auto initModelPtr = GetFunctionAddress<InitModelPtr_t>("Init_Model");

    m_modelMemref = initModelPtr(thresholdsMemref.bufferPtr, thresholdsMemref.alignedPtr, thresholdsMemref.offset, thresholdsMemref.lengths[0], thresholdsMemref.strides[0],
                featureIndexMemref.bufferPtr, featureIndexMemref.alignedPtr, featureIndexMemref.offset, featureIndexMemref.lengths[0], featureIndexMemref.strides[0],
                tileShapeIDMemref.bufferPtr, tileShapeIDMemref.alignedPtr, tileShapeIDMemref.offset, tileShapeIDMemref.lengths[0], tileShapeIDMemref.strides[0]);
  }
  else {
    assert (m_sparseRepresentation);
    Memref<ChildIndexType, 1> childIndexMemref{childIndices.data(), childIndices.data(), 0, {(int64_t)childIndices.size()}, 1};
    typedef ModelMemrefType (*InitModelPtr_t)(ThresholdType*, ThresholdType*, int64_t, int64_t, int64_t, 
                                              FeatureIndexType*, FeatureIndexType*, int64_t, int64_t, int64_t,
                                              TileShapeType*, TileShapeType*, int64_t, int64_t, int64_t,
                                              ChildIndexType*, ChildIndexType*, int64_t, int64_t, int64_t);
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
    else if (featureIndexSize == 16) {
      return ResolveTileShapeType<double, int16_t>();
    }
    else if (featureIndexSize == 8) {
      return ResolveTileShapeType<double, int8_t>();
    }
    else {
      assert (false);
    }
  }
  else if (thresholdSize == 32) {
    if (featureIndexSize == 32) {
      return ResolveTileShapeType<float, int32_t>();
    }
    else if (featureIndexSize == 16) {
      return ResolveTileShapeType<float, int16_t>();
    }
    else if (featureIndexSize == 8) {
      return ResolveTileShapeType<float, int8_t>();
    }
    else {
      assert (false);
    }
  }
  else {
    assert (false);
  }
  return 0;
}

void GPUArraySparseSerializerBase::InitializeClassInformation() {
}

void GPUArraySparseSerializerBase::CallPredictionMethod(void* predictFuncPtr,
                                                        Memref<double, 2> inputs,
                                                        Memref<double, 1> results) {
    using InputElementType = double;
    using ReturnType = double;

    typedef Memref<ReturnType, 1> (*InferenceFunc_t)(InputElementType*, InputElementType*, int64_t, int64_t, int64_t, int64_t, int64_t, 
                                                     ReturnType*, ReturnType*, int64_t, int64_t, int64_t,
                                                     Tile*, Tile*, int64_t, int64_t, int64_t,
                                                     int64_t*, int64_t*, int64_t, int64_t, int64_t,
                                                     int64_t*, int64_t*, int64_t, int64_t, int64_t,
                                                     int8_t*, int8_t*, int64_t, int64_t, int64_t);
    auto inferenceFuncPtr = reinterpret_cast<InferenceFunc_t>(predictFuncPtr);
    inferenceFuncPtr(inputs.bufferPtr, inputs.alignedPtr, inputs.offset, inputs.lengths[0], inputs.lengths[1], inputs.strides[0], inputs.strides[1],
                     results.bufferPtr, results.alignedPtr, results.offset, results.lengths[0], results.strides[0],
                     m_modelMemref.bufferPtr, m_modelMemref.alignedPtr, m_modelMemref.offset, m_modelMemref.lengths[0], m_modelMemref.strides[0],
                     m_offsetsMemref.bufferPtr, m_offsetsMemref.alignedPtr, m_offsetsMemref.offset, m_offsetsMemref.lengths[0], m_offsetsMemref.strides[0],
                     m_lengthsMemref.bufferPtr, m_lengthsMemref.alignedPtr, m_lengthsMemref.offset, m_lengthsMemref.lengths[0], m_lengthsMemref.strides[0],
                     nullptr, nullptr, 0, 0, 0);
    return;
}    

void GPUArraySparseSerializerBase::ReadData() {
    decisionforest::ForestJSONReader::GetInstance().SetFilePath(m_filepath);
    decisionforest::ForestJSONReader::GetInstance().ParseJSONFile();
    // TODO read the thresholdSize and featureIndexSize from the JSON!
    m_batchSize = decisionforest::ForestJSONReader::GetInstance().GetBatchSize();
    m_rowSize = decisionforest::ForestJSONReader::GetInstance().GetRowSize();
    m_inputTypeBitWidth = decisionforest::ForestJSONReader::GetInstance().GetInputElementBitWidth();
    m_returnTypeBitwidth = decisionforest::ForestJSONReader::GetInstance().GetReturnTypeBitWidth();
}

void GPUArraySparseSerializerBase::SetBatchSize(int32_t value){
    m_batchSize = value;
}

void GPUArraySparseSerializerBase::SetRowSize(int32_t value) {
    m_rowSize = value;
}

void GPUArraySparseSerializerBase::SetInputTypeBitWidth(int32_t value){
    m_inputTypeBitWidth = value;
}

void GPUArraySparseSerializerBase::SetReturnTypeBitWidth(int32_t value){
    m_returnTypeBitwidth = value;
}

// ===---------------------------------------------------=== //
// Persistence Helper Methods
// ===---------------------------------------------------=== //

void PersistDecisionForestArrayBased(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType);
void PersistDecisionForestSparse(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType);

// ===---------------------------------------------------=== //
// GPUSparseRepresentationSerializer Methods
// ===---------------------------------------------------=== //

void GPUSparseRepresentationSerializer::Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) {
    mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(m_filepath);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetBatchSize(m_batchSize);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetRowSize(m_rowSize);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetInputElementBitWidth(m_inputTypeBitWidth);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetReturnTypeBitWidth(m_returnTypeBitwidth);
    PersistDecisionForestSparse(forest, forestType);
}

int32_t GPUSparseRepresentationSerializer::InitializeLeafArrays() {
  return 0;
}

void GPUSparseRepresentationSerializer::InitializeBuffersImpl() {
    InitializeLengthsArray();
    InitializeOffsetsArray();
    InitializeModelArray();
    InitializeClassInformation();
    InitializeLeafArrays();
}

std::shared_ptr<IModelSerializer> ConstructGPUSparseRepresentation(const std::string& jsonFilename) {
  return std::make_shared<GPUSparseRepresentationSerializer>(jsonFilename);
}

REGISTER_SERIALIZER(gpu_sparse, ConstructGPUSparseRepresentation)

// ===---------------------------------------------------=== //
// GPUArrayRepresentationSerializer Methods
// ===---------------------------------------------------=== //

void GPUArrayRepresentationSerializer::Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) {
    mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(m_filepath);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetBatchSize(m_batchSize);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetRowSize(m_rowSize);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetInputElementBitWidth(m_inputTypeBitWidth);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetReturnTypeBitWidth(m_returnTypeBitwidth);
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

REGISTER_SERIALIZER(gpu_array, ConstructGPUArrayRepresentation)

}
}