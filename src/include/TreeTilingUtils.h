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
        std::list<std::vector<int32_t>> serializedTileShapeIDs;
        std::list<std::vector<int32_t>> serializedChildIndices;
        std::list<std::vector<ThresholdType>> serializedLeaves;

        bool operator==(const SingleTileSizeEntry& that) const {
            return tileSize==that.tileSize && thresholdBitWidth==that.thresholdBitWidth && indexBitWidth==that.indexBitWidth &&
                   treeIndices==that.treeIndices && numberOfTiles==that.numberOfTiles && 
                   serializedThresholds==that.serializedThresholds && serializedFetureIndices == that.serializedFetureIndices &&
                   serializedTileShapeIDs==that.serializedTileShapeIDs && serializedChildIndices==that.serializedChildIndices &&
                   serializedLeaves==that.serializedLeaves;
        };
    };
    int32_t m_rowSize;
    int32_t m_batchSize;
    int32_t m_tileShapeBitWidth;
    int32_t m_childIndexBitWidth;
    std::list<SingleTileSizeEntry> m_tileSizeEntries;
    json m_json;
    int32_t m_numberOfTrees;
    int8_t m_numberOfClasses;
    std::vector<int8_t> m_classIds;

    std::string m_jsonFilePath;

    std::list<SingleTileSizeEntry>::iterator FindEntry(int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth);
    static ForestJSONReader m_instance;
    ForestJSONReader() { }
    
    void WriteSingleTileSizeEntryToJSON(json& tileSizeEntryJSON, ForestJSONReader::SingleTileSizeEntry& tileSizeEntry);
    void ParseSingleTileSizeEntry(json& tileSizeEntryJSON, ForestJSONReader::SingleTileSizeEntry& tileSizeEntry);

    void AddSingleTileSizeEntry(std::list<int32_t>& treeIndices, std::list<int32_t>& numTilesList, std::list<std::vector<ThresholdType>>& serializedThresholds, 
                            std::list<std::vector<FeatureIndexType>>& serializedFetureIndices, std::list<std::vector<int32_t>>& serializedTileShapeIDs,
                            std::list<std::vector<int32_t>>& serializedChildIndices, std::list<std::vector<ThresholdType>>& serializedLeaves,
                            const int32_t tileSize, const int32_t thresholdBitWidth, const int32_t indexBitWidth);

    template<typename LeafType>
    void InitializeLeavesImpl(LeafType *bufPtr, std::list<ForestJSONReader::SingleTileSizeEntry>::iterator listIter);

public:

    //===----------------------------------------===/
    // Persist routines
    //===----------------------------------------===/
    void AddSingleTree(int32_t treeIndex, int32_t numTiles, std::vector<ThresholdType>& serializedThresholds, 
                       std::vector<FeatureIndexType>& serializedFetureIndices, std::vector<int32_t>& tileShapeIDs,
                       const int32_t tileSize, const int32_t thresholdBitWidth, const int32_t indexBitWidth, const int8_t classId);
    void AddSingleSparseTree(int32_t treeIndex, int32_t numTiles, std::vector<ThresholdType>& serializedThresholds,
                             std::vector<FeatureIndexType>& serializedFetureIndices, std::vector<int32_t>& tileShapeIDs, 
                             std::vector<int32_t>& childIndices, std::vector<ThresholdType>& leaves,
                             const int32_t tileSize, const int32_t thresholdBitWidth, const int32_t indexBitWidth);

    //===----------------------------------------===/
    // JSON routines
    //===----------------------------------------===/
    void SetFilePath(const std::string& jsonFilePath) { m_jsonFilePath = jsonFilePath; }
    void ParseJSONFile();
    void WriteJSONFile();

    //===----------------------------------------===/
    // Initialization routines
    //===----------------------------------------===/
    void InitializeBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth, std::vector<int32_t>& treeOffsets);
    void InitializeOffsetBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth);
    void InitializeLengthBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth);
    void InitializeLookUpTable(void* bufPtr, int32_t tileSize, int32_t entryBitWidth);
    void InitializeClassInformation(void *classInfoBuf);
    
    void InitializeLeaves(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth);
    void InitializeLeavesOffsetBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth);
    void InitializeLeavesLengthBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth);
    void ClearAllData();

    //===----------------------------------------===/
    // Configuration helpers routines
    //===----------------------------------------===/
    void SetNumberOfTrees(int32_t val) { m_numberOfTrees = val; }
    int32_t GetNumberOfTrees() { return m_numberOfTrees; }
    int32_t GetTotalNumberOfTiles();
    int32_t GetTotalNumberOfLeaves();
    
    int32_t GetTileShapeBitWidth() { return m_tileShapeBitWidth; }
    void SetTileShapeBitWidth(int32_t val) { m_tileShapeBitWidth=val; }

    int32_t GetChildIndexBitWidth() { return m_childIndexBitWidth; }
    void SetChildIndexBitWidth(int32_t val) { m_childIndexBitWidth=val; }

    void SetNumberOfClasses(int8_t nclasses) { m_numberOfClasses = nclasses; }

    void SetRowSize(int32_t val) { m_rowSize = val; }
    int32_t GetRowSize() { return m_rowSize; }

    void SetBatchSize(int32_t val) { m_batchSize = val; }
    int32_t GetBatchSize() { return m_batchSize; }
    
    static ForestJSONReader& GetInstance() {
        return m_instance;
    }
    static int32_t GetLengthOfTree(std::vector<int32_t>& offsets, int32_t treeIndex);
    
    template<typename ThresholdType, typename FeatureIndexType, typename TileShapeType>
    void GetModelValues(int32_t tileSize, int32_t thresholdSize, int32_t featureIndexSize, 
                        std::vector<ThresholdType>& thresholds, std::vector<FeatureIndexType>& featureIndices, std::vector<TileShapeType>& tileShapeIDs);
    
    template<typename ThresholdType, typename FeatureIndexType, typename TileShapeType, typename ChildIndexType>
    void GetModelValues(int32_t tileSize, int32_t thresholdSize, int32_t featureIndexSize, 
                        std::vector<ThresholdType>& thresholds, std::vector<FeatureIndexType>& featureIndices, std::vector<TileShapeType>& tileShapeIDs,
                        std::vector<ChildIndexType>& childIndices);

};

template<typename ThresholdType, typename FeatureIndexType, typename TileShapeType>
void ForestJSONReader::GetModelValues(int32_t tileSize, int32_t thresholdSize, int32_t featureIndexSize, 
                                      std::vector<ThresholdType>& thresholds, std::vector<FeatureIndexType>& featureIndices, std::vector<TileShapeType>& tileShapeIDs) {
    auto iter = FindEntry(tileSize, thresholdSize, featureIndexSize);
    for (auto& thresholdVec : iter->serializedThresholds)
        thresholds.insert(thresholds.end(), thresholdVec.begin(), thresholdVec.end());
    for (auto& indexVec : iter->serializedFetureIndices)
        featureIndices.insert(featureIndices.end(), indexVec.begin(), indexVec.end());
    for (auto& tileShapeIDVec : iter->serializedTileShapeIDs)
        tileShapeIDs.insert(tileShapeIDs.end(), tileShapeIDVec.begin(), tileShapeIDVec.end());
    if (tileSize > 1)
        assert ((int32_t)tileShapeIDs.size() == GetTotalNumberOfTiles());
}

template<typename ThresholdType, typename FeatureIndexType, typename TileShapeType, typename ChildIndexType>
void ForestJSONReader::GetModelValues(int32_t tileSize, int32_t thresholdSize, int32_t featureIndexSize, 
                                      std::vector<ThresholdType>& thresholds, std::vector<FeatureIndexType>& featureIndices, std::vector<TileShapeType>& tileShapeIDs,
                                      std::vector<ChildIndexType>& childIndices) {
    auto iter = FindEntry(tileSize, thresholdSize, featureIndexSize);
    for (auto& thresholdVec : iter->serializedThresholds)
        thresholds.insert(thresholds.end(), thresholdVec.begin(), thresholdVec.end());
    for (auto& indexVec : iter->serializedFetureIndices)
        featureIndices.insert(featureIndices.end(), indexVec.begin(), indexVec.end());
    for (auto& tileShapeIDVec : iter->serializedTileShapeIDs)
        tileShapeIDs.insert(tileShapeIDs.end(), tileShapeIDVec.begin(), tileShapeIDVec.end());
    for (auto& childIndicesVec : iter->serializedChildIndices)
        childIndices.insert(childIndices.end(), childIndicesVec.begin(), childIndicesVec.end());
    if (tileSize > 1) {
        assert ((int32_t)tileShapeIDs.size() == GetTotalNumberOfTiles());
        assert (tileShapeIDs.size() == childIndices.size());
    }

}
void PersistDecisionForest(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType);
void ClearPersistedForest();
int32_t GetTotalNumberOfTiles();
int32_t GetTotalNumberOfLeaves();

} // decisionforest
} // mlir

#endif // _TREETILINGUTILS_H_
