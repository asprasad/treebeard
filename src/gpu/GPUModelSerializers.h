#ifdef TREEBEARD_GPU_SUPPORT

#ifndef _GPUMODELSERIALIZERS_H_
#define _GPUMODELSERIALIZERS_H_

#include <map>
#include <set>
#include "ExecutionHelpers.h"
#include "TreebeardContext.h"

namespace mlir
{
namespace decisionforest
{

class GPUArraySparseSerializerBase : public IModelSerializer {
protected:
  // GPU buffers
  LengthMemrefType m_lengthsMemref;
  LengthMemrefType m_offsetsMemref;
  ModelMemrefType m_modelMemref;
  ClassMemrefType m_classIDMemref;
  // TODO we should have a way to store the class IDs memref here
  
  bool m_sparseRepresentation;
  template<typename ThresholdType, typename FeatureIndexType, typename TileShapeType, typename ChildIndexType>
  int32_t CallInitMethod();
  
  template<typename ThresholdType, typename FeatureIndexType>
  int32_t ResolveTileShapeType();

  template<typename ThresholdType, typename FeatureIndexType, typename TileShapeType>
  int32_t ResolveChildIndexType();

  int32_t InitializeLengthsArray();
  int32_t InitializeOffsetsArray();
  int32_t InitializeModelArray();
  void InitializeClassInformation();
public:
  GPUArraySparseSerializerBase(const std::string& modelGlobalsJSONPath, bool sparseRep)
    :IModelSerializer(modelGlobalsJSONPath), m_sparseRepresentation(sparseRep)
  { }
  ~GPUArraySparseSerializerBase() { }
  void ReadData() override;

  void CallPredictionMethod(void* predictFuncPtr,
                            Memref<double, 2> inputs,
                            Memref<double, 1> results) override;
  bool HasCustomPredictionMethod() override { return true; }
  void CleanupBuffers() override;
  
  ModelMemrefType GetModelMemref() { return m_modelMemref; }    
};

class GPUArrayRepresentationSerializer : public GPUArraySparseSerializerBase {
protected:
  void InitializeBuffersImpl() override;
public:
  GPUArrayRepresentationSerializer(const std::string& modelGlobalsJSONPath)
    :GPUArraySparseSerializerBase(modelGlobalsJSONPath, false)
  { }
  ~GPUArrayRepresentationSerializer() {}
  void Persist(mlir::decisionforest::DecisionForest& forest, mlir::decisionforest::TreeEnsembleType forestType) override;
  void CallPredictionMethod(void* predictFuncPtr,
                            Memref<double, 2> inputs,
                            Memref<double, 1> results) override;

};

using LeafValueMemref = Memref<double, 1>;

class GPUSparseRepresentationSerializer : public GPUArraySparseSerializerBase {
protected:
  LeafValueMemref m_leafValues;
  OffsetMemrefType m_leafLengthsMemref;
  OffsetMemrefType m_leafOffsetsMemref;
  // TODO This should also have a way to store the memref
  // for the leaves array
  int32_t InitializeLeafArrays();
  int32_t InitializeLeafValues();
  int32_t InitializeLeafLengths();
  int32_t InitializeLeafOffsets();
  void InitializeBuffersImpl() override;

  template<typename ThresholdType>
  LeafValueMemref InitLeafValues(int32_t tileSize, int32_t thresholdBitWidth, int32_t featureIndexBitWidth);
public:
  GPUSparseRepresentationSerializer(const std::string& modelGlobalsJSONPath)
    :GPUArraySparseSerializerBase(modelGlobalsJSONPath, true)
  { }
  ~GPUSparseRepresentationSerializer() {}
  void Persist(mlir::decisionforest::DecisionForest& forest, mlir::decisionforest::TreeEnsembleType forestType) override;
  void CallPredictionMethod(void* predictFuncPtr,
                            Memref<double, 2> inputs,
                            Memref<double, 1> results) override;
  void CleanupBuffers() override;
};

} // decisionforest
} // mlir

#endif // _GPUMODELSERIALIZERS_H_

#endif // TREEBEARD_GPU_SUPPORT