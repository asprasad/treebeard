#include "ForestCreatorFactory.h"
#include "xgboostparser.h"
#include "onnxmodelparser.h"

namespace TreeBeard
{

REGISTER_FOREST_CREATOR(xgboost_json, ConstructXGBoostJSONParser)
REGISTER_FOREST_CREATOR(onnx_file, ConstructONNXFileParser)

// ===---------------------------------------------------=== //
// ForestCreatorFactory Methods
// ===---------------------------------------------------=== //

std::shared_ptr<ForestCreator> ForestCreatorFactory::GetForestCreator(const std::string& name,
                                                                      TreeBeard::TreebeardContext& tbContext) {
  auto mapIter = m_constructionMap.find(name);
  assert (mapIter != m_constructionMap.end() && "Unknown forest creator name!");
  return mapIter->second(tbContext);
}

ForestCreatorFactory& ForestCreatorFactory::Get() {
  static std::unique_ptr<ForestCreatorFactory> sInstancePtr = nullptr;
  if (sInstancePtr == nullptr)
      sInstancePtr = std::make_unique<ForestCreatorFactory>();
  return *sInstancePtr;
}

bool ForestCreatorFactory::RegisterForestCreator(const std::string& name,
                                                 ForestCreatorConstructor_t constructionFunc) {
  assert (m_constructionMap.find(name) == m_constructionMap.end());
  m_constructionMap[name] = constructionFunc;
  return true;
}

} // TreeBeard