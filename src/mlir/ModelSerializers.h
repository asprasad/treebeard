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
  int32_t CallInitMethod();
  int32_t InitializeModelArray();
public:
  ArraySparseSerializerBase(const std::string& modelGlobalsJSONPath, bool sparseRep)
    :IModelSerializer(modelGlobalsJSONPath), m_sparseRepresentation(sparseRep)
  { }
  ~ArraySparseSerializerBase() { }

  void ReadData() override;

  void SetBatchSize(int32_t value) override;
  void SetRowSize(int32_t value) override;
  void SetInputTypeBitWidth(int32_t value) override;
  void SetReturnTypeBitWidth(int32_t value) override;
};

class ArrayRepresentationSerializer : public ArraySparseSerializerBase {
protected:
  void InitializeBuffersImpl() override;
public:
  ArrayRepresentationSerializer(const std::string& modelGlobalsJSONPath)
    :ArraySparseSerializerBase(modelGlobalsJSONPath, false)
  { }
  ~ArrayRepresentationSerializer() {}
  void Persist(mlir::decisionforest::DecisionForest& forest, mlir::decisionforest::TreeEnsembleType forestType) override;
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
  void Persist(mlir::decisionforest::DecisionForest& forest, mlir::decisionforest::TreeEnsembleType forestType) override;
};

class ModelSerializerFactory {
public:
  typedef std::shared_ptr<IModelSerializer> 
               (*SerializerConstructor_t)(const std::string& modelGlobalsJSONPath);
private:
  std::map<std::string, SerializerConstructor_t> m_constructionMap;
public:
  static ModelSerializerFactory& Get();
  bool RegisterSerializer(const std::string& name,
                          SerializerConstructor_t constructionFunc);
  std::shared_ptr<IModelSerializer> GetModelSerializer(const std::string& name,
                                                       const std::string& modelGlobalsJSONPath);
};

#define REGISTER_SERIALIZER(name, func) __attribute__((unused)) static bool UNIQUE_NAME(register_serializer_) = ModelSerializerFactory::Get().RegisterSerializer(#name, func);

// TODO This function needs to be removed
// Helper to construct the right serializer to work around the 
// global "UseSparseRepresentation"
std::shared_ptr<IModelSerializer> ConstructModelSerializer(const std::string& modelGlobalsJSONPath);
std::shared_ptr<IModelSerializer> ConstructGPUModelSerializer(const std::string& modelGlobalsJSONPath);

} // decisionforest
} // mlir

#endif // _MODELSERIALIZERS_H_