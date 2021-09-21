#ifndef _TREETILINGUTILS_H_
#define _TREETILINGUTILS_H_

#include <string>
#include <fstream>
#include <list>
#include <vector>

#include "DecisionTreeTypes.h"

#include "json.hpp"
#include "DecisionForest.h"

using json = nlohmann::json;

// *. Utilities to verify that a tiling is valid
//      - All nodes are covered, tiles are connected, # of nodes are within the limit, tiles are disjoint
// *. Utilities to manage tiles -- for example, number of tile shapes with a given tile size and thier IDs, 
//    child indices based on comparison outcomes etc.

// *. Given the serialization of a forest, write it into a file
// *. Read from a serialized file into a buffer 

namespace mlir
{
namespace decisionforest
{
using ThresholdType = double;
using FeatureIndexType = int32_t;
using IndexType = int64_t;

class ForestJSONBuilder
{
    json m_json;
public:
    ForestJSONBuilder() { }
    void AddTreesToJSON(std::list<int32_t>& treeOffsets, std::list<std::vector<ThresholdType>>& serializedThresholds, 
                        std::list<std::vector<FeatureIndexType>>& serializedFetureIndices,
                        const int32_t tileSize, const int32_t thresholdBitWidth, const int32_t indexBitWidth);
    void WriteToFile(const std::string& filename){
        std::ofstream fout(filename);
        fout << m_json;
    }
};


// For now, we're making the compiler directly populate the value of the forest into the 
// ForestJSON reader. It will (at some point in the future), read this value from 
// the JSON that is written as part of compilation.
class ForestJSONReader
{
    struct SingleTileSizeEntry {
        int32_t tileSize;
        int32_t thresholdBitWidth;
        int32_t indexBitWidth;
        std::list<int32_t> treeIndices;
        std::list<int32_t> numberOfTiles;
        std::list<std::vector<ThresholdType>> serializedThresholds;
        std::list<std::vector<FeatureIndexType>> serializedFetureIndices;
    };
    std::list<SingleTileSizeEntry> m_tileSizeEntries;
    json m_json;
    int32_t m_numberOfTrees;
    void ParseJSONFile();
    std::list<SingleTileSizeEntry>::iterator FindEntry(int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth);
    static ForestJSONReader m_instance;
    ForestJSONReader() { }
public:
    ForestJSONReader(const std::string& filename) {
        std::ifstream fin;
        fin >> m_json;
        ParseJSONFile();
    }
    void AddSingleTree(int32_t treeIndex, int32_t numTiles, std::vector<ThresholdType>& serializedThresholds, std::vector<FeatureIndexType>& serializedFetureIndices,
                       const int32_t tileSize, const int32_t thresholdBitWidth, const int32_t indexBitWidth);
    void AddSingleTileSizeEntry(std::list<int32_t>& treeIndices, std::list<int32_t>& numTilesList, std::list<std::vector<ThresholdType>>& serializedThresholds, 
                                std::list<std::vector<FeatureIndexType>>& serializedFetureIndices,
                                const int32_t tileSize, const int32_t thresholdBitWidth, const int32_t indexBitWidth);
    void InitializeBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth, std::vector<int32_t>& treeOffsets);
    void InitializeOffsetBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth);
    void InitializeLengthBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth);
    void ClearAllData();
    void SetNumberOfTrees(int32_t val) { m_numberOfTrees = val; }
    int32_t GetNumberOfTrees() { return m_numberOfTrees; }
    static ForestJSONReader& GetInstance() {
        return m_instance;
    }
    static int32_t GetLengthOfTree(std::vector<int32_t>& offsets, int32_t treeIndex);
};

void PersistDecisionForest(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType);
void ClearPersistedForest();

} // decisionforest
} // mlir

#endif // _TREETILINGUTILS_H_
