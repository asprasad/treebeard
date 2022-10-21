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
  ~ArrayRepresentationSerializer() {}
  void Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) override;
};

class SparseRepresentationSerializer : public IModelSerializer {
public:
  ~SparseRepresentationSerializer() {}
  void Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) override;
};

class ModelSerializerFactory {
public:
  static std::shared_ptr<IModelSerializer> GetModelSerializer(const std::string& name);
};

// TODO This function needs to be removed
// Helper to construct the right serializer to work around the 
// global "UseSparseRepresentation"
std::shared_ptr<IModelSerializer> ConstructModelSerializer();

} // decisionforest
} // mlir

#endif // _MODELSERIALIZERS_H_