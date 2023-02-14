#ifndef _MODELSERIALIZERS_H_
#define _MODELSERIALIZERS_H_

#include <map>
#include <set>
#include "TreebeardContext.h"

namespace mlir
{
namespace decisionforest
{

class ArrayRepresentationSerializer : public IModelSerializer {
public:
  ArrayRepresentationSerializer(const std::string& modelGlobalsJSONPath)
    :IModelSerializer(modelGlobalsJSONPath)
  { }
  ~ArrayRepresentationSerializer() {}
  void Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) override;

  void SetBatchSize(int32_t value) override;
  void SetRowSize(int32_t value) override;
  void SetInputTypeBitWidth(int32_t value) override;
  void SetReturnTypeBitWidth(int32_t value) override;
};

class SparseRepresentationSerializer : public IModelSerializer {
public:
  SparseRepresentationSerializer(const std::string& modelGlobalsJSONPath)
    :IModelSerializer(modelGlobalsJSONPath)
  { }
  ~SparseRepresentationSerializer() {}
  void Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) override;
  
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