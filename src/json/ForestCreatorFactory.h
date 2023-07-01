#ifndef _FORESTCREATORFACTORY_H_
#define _FORESTCREATORFACTORY_H_

#include "forestcreator.h"
#include <map>

namespace TreeBeard
{

class ForestCreatorFactory {
public:
  typedef std::shared_ptr<TreeBeard::ForestCreator> (*ForestCreatorConstructor_t)(TreeBeard::TreebeardContext& tbContext);
private:
  std::map<std::string, ForestCreatorConstructor_t> m_constructionMap;
public:
  static ForestCreatorFactory& Get();
  bool RegisterForestCreator(const std::string& name,
                             ForestCreatorConstructor_t constructionFunc);

  std::shared_ptr<ForestCreator> GetForestCreator(const std::string& name, TreeBeard::TreebeardContext& tbContext);
};

#define REGISTER_FOREST_CREATOR(name, func) __attribute__((unused)) static bool UNIQUE_NAME(register_rep_) = ForestCreatorFactory::Get().RegisterForestCreator(#name, func);

} // TreeBeard
#endif // ifndef _FORESTCREATORFACTORY_H_

