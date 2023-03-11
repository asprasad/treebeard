#ifndef _GPUMODELSERIALIZERS_H_
#define _GPUMODELSERIALIZERS_H_

#include <map>
#include <set>
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

  void SetBatchSize(int32_t value) override;
  void SetRowSize(int32_t value) override;
  void SetInputTypeBitWidth(int32_t value) override;
  void SetReturnTypeBitWidth(int32_t value) override;

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
  void Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) override;
};

class GPUSparseRepresentationSerializer : public GPUArraySparseSerializerBase {
protected:
  // TODO This should also have a way to store the memref
  // for the leaves array
  int32_t InitializeLeafArrays();
  void InitializeBuffersImpl() override;
public:
  GPUSparseRepresentationSerializer(const std::string& modelGlobalsJSONPath)
    :GPUArraySparseSerializerBase(modelGlobalsJSONPath, true)
  { }
  ~GPUSparseRepresentationSerializer() {}
  void Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) override;
};

} // decisionforest
} // mlir

#endif // _GPUMODELSERIALIZERS_H_