#include "ForestCreatorFactory.h"
#include "onnxmodelparser.h"
#include "xgboostparser.h"

namespace TreeBeard {

REGISTER_FOREST_CREATOR(xgboost_json, ConstructXGBoostJSONParser)

#if ENABLE_ONNX_PARSER
REGISTER_FOREST_CREATOR(onnx_file, ConstructONNXFileParser)
#endif // ENABLE_ONNX_PARSER

// ===---------------------------------------------------=== //
// ForestCreatorFactory Methods
// ===---------------------------------------------------=== //

std::shared_ptr<ForestCreator>
ForestCreatorFactory::GetForestCreator(const std::string &name,
                                       TreeBeard::TreebeardContext &tbContext) {
  auto mapIter = m_constructionMap.find(name);
  assert(mapIter != m_constructionMap.end() && "Unknown forest creator name!");
  return mapIter->second(tbContext);
}

ForestCreatorFactory &ForestCreatorFactory::Get() {
  static std::unique_ptr<ForestCreatorFactory> sInstancePtr = nullptr;
  if (sInstancePtr == nullptr)
    sInstancePtr = std::make_unique<ForestCreatorFactory>();
  return *sInstancePtr;
}

bool ForestCreatorFactory::RegisterForestCreator(
    const std::string &name, ForestCreatorConstructor_t constructionFunc) {
  assert(m_constructionMap.find(name) == m_constructionMap.end());
  m_constructionMap[name] = constructionFunc;
  return true;
}

} // namespace TreeBeard