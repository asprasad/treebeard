#ifndef _MODELSERIALIZERS_H_
#define _MODELSERIALIZERS_H_

#include <map>
#include <set>
#include "TreebeardContext.h"

namespace mlir
{
namespace decisionforest
{

class ArraySparseSerializerBase : public IModelSerializer {
protected:
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
  ArraySparseSerializerBase(const std::string& modelGlobalsJSONPath, bool sparseRep)
    :IModelSerializer(modelGlobalsJSONPath), m_sparseRepresentation(sparseRep)
  { }
  ~ArraySparseSerializerBase() { }
};

class ArrayRepresentationSerializer : public ArraySparseSerializerBase {
protected:
  void InitializeBuffersImpl() override;
public:
  ArrayRepresentationSerializer(const std::string& modelGlobalsJSONPath)
    :ArraySparseSerializerBase(modelGlobalsJSONPath, false)
  { }
  ~ArrayRepresentationSerializer() {}
  void Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) override;
  void ReadData() override;

  void SetBatchSize(int32_t value) override;
  void SetRowSize(int32_t value) override;
  void SetInputTypeBitWidth(int32_t value) override;
  void SetReturnTypeBitWidth(int32_t value) override;
};

class SparseRepresentationSerializer : public ArraySparseSerializerBase {
protected:
  int32_t InitializeLeafArrays();
  void InitializeBuffersImpl() override;
public:
  SparseRepresentationSerializer(const std::string& modelGlobalsJSONPath)
    :ArraySparseSerializerBase(modelGlobalsJSONPath, true)
  { }
  ~SparseRepresentationSerializer() {}
  void Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) override;
  void ReadData() override;
  
  void SetBatchSize(int32_t value) override;
  void SetRowSize(int32_t value) override;
  void SetInputTypeBitWidth(int32_t value) override;
  void SetReturnTypeBitWidth(int32_t value) override;
};

class ModelSerializerFactory {
public:
  static std::shared_ptr<IModelSerializer> GetModelSerializer(const std::string& name, const std::string& modelGlobalsJSONPath);
};

// TODO This function needs to be removed
// Helper to construct the right serializer to work around the 
// global "UseSparseRepresentation"
std::shared_ptr<IModelSerializer> ConstructModelSerializer(const std::string& modelGlobalsJSONPath);

} // decisionforest
} // mlir

#endif // _MODELSERIALIZERS_H_