#include "json.hpp"

#ifndef _JSONHELPERS_H_
#define _JSONHELPERS_H_

using json = nlohmann::json;

template<typename T>
json WriteListOfVectorsToJSON(std::list<std::vector<T>>& values) {
    json retJSON;
    for (auto& val : values) {
        json currJSON = val;
        retJSON.push_back(currJSON);
    }
    return retJSON;
}

template<typename ContainerType, typename ElemType>
void ParseJSONList(ContainerType& container, json& jsonList) {
    for (auto& elemJSON : jsonList) {
        ElemType elem = elemJSON.get<ElemType>();
        container.push_back(elem);
    }
}

template<typename ContainerType, typename ElemType>
void ParseJSONListOfLists(std::list<ContainerType>& container, json& jsonListOfLists) {
    for (auto& jsonList : jsonListOfLists) {
        ContainerType elem;
        ParseJSONList<ContainerType, ElemType>(elem, jsonList);
        container.push_back(elem);
    }
}


#endif // _JSONHELPERS_H_