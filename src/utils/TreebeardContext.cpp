#include "TreebeardContext.h"
#include "ForestCreatorFactory.h"
#include "Representations.h"

namespace TreeBeard
{

void TreebeardContext::SetForestCreatorType(const std::string& creatorName) {
  this->forestConstructor = ForestCreatorFactory::Get().GetForestCreator(creatorName, *this);
}

void TreebeardContext::SetRepresentationAndSerializer(const std::string& repName) {
  using namespace mlir::decisionforest;
  this->serializer = ModelSerializerFactory::Get().GetModelSerializer(repName, this->modelGlobalsJSONPath);
  this->representation = RepresentationFactory::Get().GetRepresentation(repName);
}

}