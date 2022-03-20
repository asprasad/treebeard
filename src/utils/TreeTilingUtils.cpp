#include <queue>
#include <iostream>
#include "TreeTilingDescriptor.h"
#include "TreeTilingUtils.h"
#include "TiledTree.h"
#include "llvm/ADT/STLExtras.h"
#include "Dialect.h"
#include "Logger.h"

namespace mlir
{
namespace decisionforest
{

ForestJSONReader ForestJSONReader::m_instance;

// TODO How does this work for the last tree? We need to also know the length of the full serialization to compute the length
// of the last tree
int32_t ForestJSONReader::GetLengthOfTree(std::vector<int32_t>& offsets, int32_t treeIndex) {
  auto startOffset = offsets[treeIndex];
  ++treeIndex;
  while (offsets[treeIndex] == -1)
    ++treeIndex;
  auto endOffset = offsets[treeIndex];
  return endOffset - startOffset;
}

// Since this a tree compile time function, we don't really care about the actual 
// type of threshold and index. We'll just write the widths to the file and make
// sure we initialize the runtime buffer (model memrefs) with the correct types
void ForestJSONBuilder::AddTreesToJSON(std::list<int32_t>& treeIndices, std::list<std::vector<ThresholdType>>& serializedThresholds,
                                       std::list<std::vector<FeatureIndexType>>& serializedFetureIndices,
                                       const int32_t tileSize, const int32_t thresholdBitWidth, const int32_t indexBitWidth) {
    json treeSet;
    treeSet["tileSize"] = tileSize;
    treeSet["thresholdBitWidth"] = thresholdBitWidth;
    treeSet["indexBitWidth"] = indexBitWidth;

    // list of lists that represent the node values
    json treeValues;
    assert (serializedThresholds.size() == serializedFetureIndices.size());
    assert (treeIndices.size() == serializedFetureIndices.size());
    auto thresholdIter = serializedThresholds.begin();
    auto featureIndexIter = serializedFetureIndices.begin();
    auto treeIndexIter = treeIndices.begin();
    while(thresholdIter != serializedThresholds.end()) {
        json singleTree;
        auto& thresholds = *thresholdIter;
        auto& featureIndices = *featureIndexIter;
        assert (thresholds.size() == featureIndices.size());
        singleTree["treeIndex"] = *treeIndexIter;
        singleTree["numberOfNodes"] = thresholds.size();

        json nodeVals;
        for (size_t i=0 ; i<thresholds.size() ; ++i) {
            json node;
            node["threshold"] = thresholds[i];
            node["featureIndex"] = featureIndices[i];
            nodeVals.push_back(node);
        }
        singleTree["nodes"] = nodeVals;
        treeValues.push_back(singleTree);

        ++thresholdIter;
        ++featureIndexIter;
        ++treeIndexIter;
    }
    treeSet["trees"] = treeValues;
    m_json.push_back(treeSet);
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

void ForestJSONReader::ParseSingleTileSizeEntry(json& tileSizeEntryJSON, ForestJSONReader::SingleTileSizeEntry& tileSizeEntry) {
    tileSizeEntry.tileSize = tileSizeEntryJSON["TileSize"].get<int32_t>();
    tileSizeEntry.thresholdBitWidth = tileSizeEntryJSON["ThresholdBitWidth"].get<int32_t>();
    tileSizeEntry.indexBitWidth = tileSizeEntryJSON["FeatureIndexBitWidth"].get<int32_t>();
    // std::list<int32_t> treeIndices;
    ParseJSONList<std::list<int32_t>, int32_t>(tileSizeEntry.treeIndices, tileSizeEntryJSON["TreeIndices"]);
    // std::list<int32_t> numberOfTiles;
    ParseJSONList<std::list<int32_t>, int32_t>(tileSizeEntry.numberOfTiles, tileSizeEntryJSON["NumberOfTiles"]);
    // std::list<std::vector<ThresholdType>> serializedThresholds;
    ParseJSONListOfLists<std::vector<ThresholdType>, ThresholdType>(tileSizeEntry.serializedThresholds, tileSizeEntryJSON["SerializedThresholds"]);
    // std::list<std::vector<FeatureIndexType>> serializedFetureIndices;
    ParseJSONListOfLists<std::vector<FeatureIndexType>, FeatureIndexType>(tileSizeEntry.serializedFetureIndices, tileSizeEntryJSON["SerializedFeatureIndices"]);
    // std::list<std::vector<int32_t>> serializedTileShapeIDs;
    ParseJSONListOfLists<std::vector<int32_t>, int32_t>(tileSizeEntry.serializedTileShapeIDs, tileSizeEntryJSON["SerializedTileShapeIDs"]);
    // std::list<std::vector<int32_t>> serializedLeafBitMasks;
    ParseJSONListOfLists<std::vector<int32_t>, int32_t>(tileSizeEntry.serializedLeafBitMasks, tileSizeEntryJSON["SerializedLeafBitMasks"]);
    // std::list<std::vector<int32_t>> serializedChildIndices;
    ParseJSONListOfLists<std::vector<int32_t>, int32_t>(tileSizeEntry.serializedChildIndices, tileSizeEntryJSON["SerializedChildIndices"]);
    // std::list<std::vector<int32_t>> serializedLeafIndices;
    ParseJSONListOfLists<std::vector<int32_t>, int32_t>(tileSizeEntry.serializedLeafIndices, tileSizeEntryJSON["SerializedLeafIndices"]);
    // std::list<std::vector<ThresholdType>> serializedLeaves;
    ParseJSONListOfLists<std::vector<ThresholdType>, ThresholdType>(tileSizeEntry.serializedLeaves, tileSizeEntryJSON["SerializedLeaves"]);
    // std::list<int8_t> classIDs;
    ParseJSONList<std::list<int8_t>, int8_t>(tileSizeEntry.classIDs, tileSizeEntryJSON["TreeClassIDs"]);
}

void ForestJSONReader::ParseJSONFile() {
    ClearAllData();
    assert (m_jsonFilePath != "");
    m_json.clear();
    std::ifstream fin(m_jsonFilePath);
    fin >> m_json;

    m_inputElementBitwidth = m_json["InputElementBitWidth"];
    m_returnTypeBitWidth = m_json["ReturnTypeBitWidth"];
    m_rowSize = m_json["RowSize"];
    m_batchSize = m_json["BatchSize"];
    m_numberOfTrees = m_json["NumberOfTrees"];
    m_childIndexBitWidth = m_json["ChildIndexBitWidth"];
    m_tileShapeBitWidth = m_json["TileShapeBitWidth"];
    m_numberOfClasses = m_json["NumberOfClasses"];
    decisionforest::UseSparseTreeRepresentation = m_json["SparseRepresentation"];

    std::list<SingleTileSizeEntry> newEntries;
    for (auto& tileSizeEntryJSON : m_json["TileSizeEntries"]) {
        SingleTileSizeEntry tileSizeEntry;
        ParseSingleTileSizeEntry(tileSizeEntryJSON, tileSizeEntry);
        newEntries.push_back(tileSizeEntry);
    }
    m_tileSizeEntries = newEntries;
}

template<typename T>
json WriteListOfVectorsToJSON(std::list<std::vector<T>>& values) {
    json retJSON;
    for (auto& val : values) {
        json currJSON = val;
        retJSON.push_back(currJSON);
    }
    return retJSON;
}

void ForestJSONReader::WriteSingleTileSizeEntryToJSON(json& tileSizeEntryJSON, ForestJSONReader::SingleTileSizeEntry& tileSizeEntry) {
    // int32_t tileSize;
    tileSizeEntryJSON["TileSize"] = tileSizeEntry.tileSize;
    // int32_t thresholdBitWidth;
    tileSizeEntryJSON["ThresholdBitWidth"] = tileSizeEntry.thresholdBitWidth;
    // int32_t indexBitWidth;
    tileSizeEntryJSON["FeatureIndexBitWidth"] = tileSizeEntry.indexBitWidth;
    // std::list<int32_t> treeIndices;
    tileSizeEntryJSON["TreeIndices"] = tileSizeEntry.treeIndices;
    // std::list<int32_t> numberOfTiles;
    tileSizeEntryJSON["NumberOfTiles"] = tileSizeEntry.numberOfTiles;
    // std::list<std::vector<ThresholdType>> serializedThresholds;
    tileSizeEntryJSON["SerializedThresholds"] = WriteListOfVectorsToJSON(tileSizeEntry.serializedThresholds);
    // std::list<std::vector<FeatureIndexType>> serializedFetureIndices;
    tileSizeEntryJSON["SerializedFeatureIndices"] = WriteListOfVectorsToJSON(tileSizeEntry.serializedFetureIndices);
    // std::list<std::vector<int32_t>> serializedTileShapeIDs;
    tileSizeEntryJSON["SerializedTileShapeIDs"] = WriteListOfVectorsToJSON(tileSizeEntry.serializedTileShapeIDs);
    // std::list<std::vector<int32_t>> serializedLeafBitMasks;
    tileSizeEntryJSON["SerializedLeafBitMasks"] = WriteListOfVectorsToJSON(tileSizeEntry.serializedLeafBitMasks);
    // std::list<std::vector<int32_t>> serializedChildIndices;
    tileSizeEntryJSON["SerializedChildIndices"] = WriteListOfVectorsToJSON(tileSizeEntry.serializedChildIndices);
    // std::list<std::vector<int32_t>> serializedLeafIndices;
    tileSizeEntryJSON["SerializedLeafIndices"] = WriteListOfVectorsToJSON(tileSizeEntry.serializedLeafIndices);
    // std::list<std::vector<ThresholdType>> serializedLeaves;
    tileSizeEntryJSON["SerializedLeaves"] = WriteListOfVectorsToJSON(tileSizeEntry.serializedLeaves);
    // #TODO - Persist the bit width of the class ID type as well.
    tileSizeEntryJSON["TreeClassIDs"] = tileSizeEntry.classIDs;
}

void ForestJSONReader::WriteJSONFile() {
    // m_jsonFilePath = "/home/ashwin/temp/modelValues.json";
    assert (m_jsonFilePath != "");
    m_json.clear();

    m_json["InputElementBitWidth"] = m_inputElementBitwidth;
    m_json["ReturnTypeBitWidth"] = m_returnTypeBitWidth;
    m_json["RowSize"] = m_rowSize;
    m_json["BatchSize"] = m_batchSize;
    m_json["NumberOfTrees"] = m_numberOfTrees;
    m_json["ChildIndexBitWidth"] = m_childIndexBitWidth;
    m_json["TileShapeBitWidth"] = m_tileShapeBitWidth;
    m_json["SparseRepresentation"] = decisionforest::UseSparseTreeRepresentation;
    m_json["NumberOfClasses"] = m_numberOfClasses;

    for (auto& tileSizeEntry : m_tileSizeEntries) {
        json tileSizeJSON;
        WriteSingleTileSizeEntryToJSON(tileSizeJSON, tileSizeEntry);
        m_json["TileSizeEntries"].push_back(tileSizeJSON);
    }

    std::ofstream fout(m_jsonFilePath);
    assert(fout);
    fout << m_json;
    fout.close();
}

template<typename T>
void AppendAtEndOfList(std::list<T>& l, std::list<T>& newElements) {
    l.insert(std::end(l), std::begin(newElements), std::end(newElements));
}

void ForestJSONReader::AddSingleTileSizeEntry(std::list<int32_t>& treeIndices, std::list<int32_t>& numTilesList, std::list<std::vector<ThresholdType>>& serializedThresholds, 
                                              std::list<std::vector<FeatureIndexType>>& serializedFetureIndices, std::list<std::vector<int32_t>>& serializedTileShapeIDs,
                                              std::list<std::vector<int32_t>> serializedLeafBitMasks, 
                                              std::list<std::vector<int32_t>>& serializedChildIndices,
                                              std::list<std::vector<int32_t>> serializedLeafIndices,
                                              std::list<std::vector<ThresholdType>>& serializedLeaves,
                                              std::list<int8_t>& classIDs,
                                              const int32_t tileSize, const int32_t thresholdBitWidth, const int32_t indexBitWidth) {
    // Find if there is already an entry with the given tileSize, thresholdWidth and indexWidth.
    auto listIter = this->m_tileSizeEntries.begin();
    while (listIter != m_tileSizeEntries.end()) {
        if (listIter->tileSize == tileSize && listIter->thresholdBitWidth==thresholdBitWidth && listIter->indexBitWidth==indexBitWidth)
            break;
        ++listIter;
    }
    if (listIter == m_tileSizeEntries.end()) {
        SingleTileSizeEntry entry {tileSize, thresholdBitWidth, indexBitWidth, treeIndices, numTilesList, 
                                   serializedThresholds, serializedFetureIndices, serializedTileShapeIDs, serializedLeafBitMasks,
                                   serializedChildIndices, serializedLeafIndices, serializedLeaves, classIDs};
        m_tileSizeEntries.push_back(entry);
    }
    else {
        AppendAtEndOfList(listIter->treeIndices, treeIndices);
        AppendAtEndOfList(listIter->numberOfTiles, numTilesList);
        AppendAtEndOfList(listIter->serializedThresholds, serializedThresholds);
        AppendAtEndOfList(listIter->serializedFetureIndices, serializedFetureIndices);
        AppendAtEndOfList(listIter->serializedTileShapeIDs, serializedTileShapeIDs);
        AppendAtEndOfList(listIter->serializedLeafBitMasks, serializedLeafBitMasks);
        AppendAtEndOfList(listIter->serializedChildIndices, serializedChildIndices);
        AppendAtEndOfList(listIter->serializedLeafIndices, serializedLeafIndices);
        AppendAtEndOfList(listIter->serializedLeaves, serializedLeaves);
        AppendAtEndOfList(listIter->classIDs, classIDs);
    }
}

void ForestJSONReader::AddSingleTree(int32_t treeIndex, int32_t numTiles, std::vector<ThresholdType>& serializedThresholds,
                                     std::vector<FeatureIndexType>& serializedFetureIndices, std::vector<int32_t>& tileShapeIDs,
                                     const int32_t tileSize, const int32_t thresholdBitWidth, const int32_t indexBitWidth,
                                     // #TODO Tree-Beard#19
                                     const int8_t classId) {
    std::list<int32_t> treeIndices = { treeIndex };
    std::list<int32_t> numTilesList = { numTiles };
    std::list<std::vector<ThresholdType>> serializedThresholdsList = { serializedThresholds }, serializedLeaves;
    std::list<std::vector<FeatureIndexType>> serializedFetureIndicesList = { serializedFetureIndices };
    std::list<std::vector<int32_t>> serializedTileShapeIDs = { tileShapeIDs };
    std::list<std::vector<int32_t>> serializedChildIndices, serializedLeafIndices, serializedLeafBitMasks;
    std::list<int8_t> classIDs = { classId };

    AddSingleTileSizeEntry(treeIndices, numTilesList, serializedThresholdsList, serializedFetureIndicesList, serializedTileShapeIDs, serializedLeafBitMasks, 
                           serializedChildIndices, serializedLeafIndices, serializedLeaves, classIDs, tileSize, thresholdBitWidth, indexBitWidth);
}

void ForestJSONReader::AddSingleSparseTree(int32_t treeIndex, int32_t numTiles, std::vector<ThresholdType>& serializedThresholds,
                                           std::vector<FeatureIndexType>& serializedFetureIndices, std::vector<int32_t>& tileShapeIDs, 
                                           std::vector<int32_t>& childIndices, std::vector<ThresholdType>& leaves,
                                           const int32_t tileSize, const int32_t thresholdBitWidth, const int32_t indexBitWidth,
                                           // #TODO Tree-Beard#19
                                           const int8_t classId) {
    std::list<int32_t> treeIndices = { treeIndex };
    std::list<int32_t> numTilesList = { numTiles };
    std::list<std::vector<ThresholdType>> serializedThresholdsList = { serializedThresholds }, serializedLeaves = { leaves };
    std::list<std::vector<FeatureIndexType>> serializedFetureIndicesList = { serializedFetureIndices };
    std::list<std::vector<int32_t>> serializedTileShapeIDs = { tileShapeIDs };
    std::list<std::vector<int32_t>> serializedChildIndices = { childIndices };
    std::list<int8_t> classIDs = { classId };
    std::list<std::vector<int32_t>> serializedLeafIndices, serializedLeafBitMasks;

    AddSingleTileSizeEntry(treeIndices, numTilesList, serializedThresholdsList, serializedFetureIndicesList, serializedTileShapeIDs, serializedLeafBitMasks,
                           serializedChildIndices, serializedLeafIndices, serializedLeaves, classIDs, tileSize, thresholdBitWidth, indexBitWidth);
}

void ForestJSONReader::AddSingleSparseTree(int32_t treeIndex, int32_t numTiles, std::vector<ThresholdType>& serializedThresholds,
                            std::vector<FeatureIndexType>& serializedFetureIndices, std::vector<int32_t>& tileShapeIDs, std::vector<int32_t>& leafBitMasks, 
                            std::vector<int32_t>& childIndices, std::vector<int32_t>& leafIndices, std::vector<ThresholdType>& leaves,
                            const int32_t tileSize, const int32_t thresholdBitWidth, const int32_t indexBitWidth, const int8_t classId) {
    std::list<int32_t> treeIndices = { treeIndex };
    std::list<int32_t> numTilesList = { numTiles };
    std::list<std::vector<ThresholdType>> serializedThresholdsList = { serializedThresholds }, serializedLeaves = { leaves };
    std::list<std::vector<FeatureIndexType>> serializedFetureIndicesList = { serializedFetureIndices };
    std::list<std::vector<int32_t>> serializedTileShapeIDs = { tileShapeIDs };
    std::list<std::vector<int32_t>> serializedChildIndices = { childIndices };
    std::list<int8_t> classIDs = { classId };
    std::list<std::vector<int32_t>> serializedLeafIndices = { leafIndices }, serializedLeafBitMasks = { leafBitMasks };

    AddSingleTileSizeEntry(treeIndices, numTilesList, serializedThresholdsList, serializedFetureIndicesList, serializedTileShapeIDs, serializedLeafBitMasks,
                           serializedChildIndices, serializedLeafIndices, serializedLeaves, classIDs, tileSize, thresholdBitWidth, indexBitWidth);
    
}

void ForestJSONReader::ClearAllData() {
    m_tileSizeEntries.clear();
    // Can't clear the json path here because PersistForest calls Clear!
    // m_jsonFilePath.clear();
    m_tileShapeBitWidth = -1;
    m_childIndexBitWidth = -1;
    m_numberOfTrees = -1;
}

template<typename DestType, typename SourceType>
struct ElementCopier {
    void copyElement(char *buf, SourceType& val) {
        *reinterpret_cast<DestType*>(buf) = static_cast<DestType>(val);
    }

    void incrementPtr(char* &buf){
        buf += sizeof(DestType);
    }
};

class CopyModelValuesIntoBufferInterface {
    void IncrementPointer(char*& ptr, int32_t bytesToIncrement) {
        ptr += bytesToIncrement;
    }
public:
    virtual void CopyElements(char* bufPtr, std::vector<int32_t>& offsets, std::list<int32_t>& numberOfTiles, std::list<std::vector<ThresholdType>>& thresholdVals, 
                              std::list<std::vector<FeatureIndexType>>& featureIndices, std::list<std::vector<int32_t>>& tileShapeIDs,
                              std::list<int32_t>& treeIndices) = 0;
};

template<typename CopyThreshold, typename CopyFeatureIndex>
class CopyModelValuesIntoBuffer : public CopyModelValuesIntoBufferInterface {
    int32_t m_tileSize;
    // int32_t m_thresholdSizeInBytes;
    // int32_t m_featureIndexSizeInBytes;
    int32_t m_featureIndexStartOffsetInBytes;
    int32_t m_tileShapeIDOffsetInBytes;
    int32_t m_tileSizeInBytes;
    CopyThreshold m_copyThreshold;
    CopyFeatureIndex m_copyFeatureIndex;
    bool m_writeTileShapeID;

    int32_t CopySingleTree(char* &bufPtr, std::vector<ThresholdType>& thresholds, std::vector<FeatureIndexType>& featureIndices, 
                           std::vector<int32_t>& tileShapeIDs) {
        int32_t numTilesWritten = 0;
        assert (thresholds.size() == featureIndices.size());
        
        for (size_t i=0, tileIndex=0 ; i<thresholds.size() ; i+=m_tileSize, ++tileIndex) {
            // Copy a single tile
            // First the thresholds
            char *currPtr = bufPtr;
            for (size_t j=0 ; j<(size_t)m_tileSize ; ++j) {
                m_copyThreshold.copyElement(currPtr, thresholds[i+j]);
                m_copyThreshold.incrementPtr(currPtr);
            }
            // Then copy the feature indices
            currPtr = bufPtr + m_featureIndexStartOffsetInBytes;
            for (size_t j=0 ; j<(size_t)m_tileSize ; ++j) {
                m_copyFeatureIndex.copyElement(currPtr, featureIndices[i+j]);
                m_copyFeatureIndex.incrementPtr(currPtr);
            }

            if (m_writeTileShapeID) {
                currPtr = bufPtr + m_tileShapeIDOffsetInBytes;
                *reinterpret_cast<int32_t*>(currPtr) = static_cast<int32_t>(tileShapeIDs.at(tileIndex));
            }
            bufPtr += m_tileSizeInBytes;
            numTilesWritten += 1;
        }
        return numTilesWritten;
    }
public:
    CopyModelValuesIntoBuffer(int32_t tileSize, int32_t featureIndexStart, int32_t tileSizeInBytes, int32_t tileShapeIDStart, bool writeTileShapeID)
        :m_tileSize(tileSize), m_featureIndexStartOffsetInBytes(featureIndexStart), m_tileShapeIDOffsetInBytes(tileShapeIDStart), 
        m_tileSizeInBytes(tileSizeInBytes), m_writeTileShapeID(writeTileShapeID)
         // m_thresholdSizeInBytes(thresholdSize), m_featureIndexSizeInBytes(featureIndexSize)
    {}

    void CopyElements(char* bufPtr, std::vector<int32_t>& offsets, std::list<int32_t>& numberOfTiles, std::list<std::vector<ThresholdType>>& thresholdVals, 
                      std::list<std::vector<FeatureIndexType>>& featureIndices, std::list<std::vector<int32_t>>& tileShapeIDs, 
                      std::list<int32_t>& treeIndices) override {
        
        // TODO this function assumes that all trees have non zero elements to write into the output buffer. 
        // This may not be the case when we have multiple tile sizes. This function needs to check and 
        // return early when a tree has no tiles to write for the current tile size (maybe offset[i] == offset[i+1]
        // -- but how will this work for the last tree?)
        // Actually, this is only going over tree indices that are non-empty. So maybe not a problem?
        assert (thresholdVals.size() == featureIndices.size());
        assert (treeIndices.size() == thresholdVals.size());
        assert (tileShapeIDs.size() == thresholdVals.size());

        auto thresholdIter = thresholdVals.begin();
        auto featureIndexIter = featureIndices.begin();
        auto tileShapeIDsIter = tileShapeIDs.begin();
        auto treeIndexIter = treeIndices.begin();
        auto numTilesIter = numberOfTiles.begin();
        // This is the offset into the buffer in terms of tiles
        int32_t currTileOffset = 0;
        while (thresholdIter != thresholdVals.end())
        {
            // Write the offset of the current tree -- this is the start index (in tiles)
            // of the current tree. 
            auto treeIndex = *treeIndexIter;
            assert (offsets[treeIndex] == -1 && "Tree start offset can only be written once");
            offsets[treeIndex] = currTileOffset;

            // Copy the tiles of the current tree into the buffer
            auto tilesWritten = CopySingleTree(bufPtr, *thresholdIter, *featureIndexIter, *tileShapeIDsIter);

            currTileOffset += tilesWritten;
            assert(*numTilesIter == tilesWritten && "Number of tiles copied should match");
            ++thresholdIter;
            ++featureIndexIter;
            ++tileShapeIDsIter;
            ++treeIndexIter;
            ++numTilesIter;
        }
    }
};

CopyModelValuesIntoBufferInterface* GetModelCopier(int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth){
    // TODO we need a better way to allocate these copiers. Maybe add a base class for the copiers so we can allocate them separately?
    
    // TODO We need to take care of alignment here. For now just copying things with 1 byte packing
    // TODO We're always copying the tileShapeID when tile size != 1. This should be a more configurable decision.
    // For example, we shouldn't copy it if the tiles in a forest are all the same shape. 
    int32_t featureIndexStart = (tileSize * thresholdBitWidth/8); // Tile is written as <thresholds X tileSize, featureIndices X tileSize>
    int32_t tileSizeInBytes = ((thresholdBitWidth + indexBitWidth) * tileSize)/8;
    int32_t tileShapeIDOffsetInBytes = -1;
    bool copyTileShapeID = false;
    if (tileSize != 1) {
        tileShapeIDOffsetInBytes = tileSizeInBytes;
        tileSizeInBytes += sizeof(int32_t);
        copyTileShapeID = true;
    }

    if (thresholdBitWidth == 32) {
        using ThresholdCopier = ElementCopier<float, ThresholdType>;
        if (indexBitWidth == 8)
            return new CopyModelValuesIntoBuffer<ThresholdCopier, ElementCopier<int8_t, FeatureIndexType>>(tileSize, featureIndexStart, tileSizeInBytes, 
                                                                                                           tileShapeIDOffsetInBytes, copyTileShapeID);
        else if(indexBitWidth == 16)
            return new CopyModelValuesIntoBuffer<ThresholdCopier, ElementCopier<int16_t, FeatureIndexType>>(tileSize, featureIndexStart, tileSizeInBytes,
                                                                                                            tileShapeIDOffsetInBytes, copyTileShapeID);
        else if(indexBitWidth == 32)
            return new CopyModelValuesIntoBuffer<ThresholdCopier, ElementCopier<int32_t, FeatureIndexType>>(tileSize, featureIndexStart, tileSizeInBytes,
                                                                                                            tileShapeIDOffsetInBytes, copyTileShapeID);
        else {
            assert (false && "unsupported feature index bitwidth");
            return nullptr;
        }

    }
    else if (thresholdBitWidth == 64) {
        using ThresholdCopier = ElementCopier<double, ThresholdType>;
        if (indexBitWidth == 8)
            return new CopyModelValuesIntoBuffer<ThresholdCopier, ElementCopier<int8_t, FeatureIndexType>>(tileSize, featureIndexStart, tileSizeInBytes,
                                                                                                           tileShapeIDOffsetInBytes, copyTileShapeID);
        else if(indexBitWidth == 16)
            return new CopyModelValuesIntoBuffer<ThresholdCopier, ElementCopier<int16_t, FeatureIndexType>>(tileSize, featureIndexStart, tileSizeInBytes,
                                                                                                            tileShapeIDOffsetInBytes, copyTileShapeID);
        else if(indexBitWidth == 32)
            return new CopyModelValuesIntoBuffer<ThresholdCopier, ElementCopier<int32_t, FeatureIndexType>>(tileSize, featureIndexStart, tileSizeInBytes,
                                                                                                            tileShapeIDOffsetInBytes, copyTileShapeID);
        else {
            assert (false && "unsupported feature index bitwidth");
            return nullptr;
        }
    }
    return nullptr;
}

std::list<ForestJSONReader::SingleTileSizeEntry>::iterator ForestJSONReader::FindEntry(int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth) {
    // Find if there is already an entry with the given tileSize, thresholdWidth and indexWidth.
    auto listIter = this->m_tileSizeEntries.begin();
    while (listIter != m_tileSizeEntries.end()) {
        if (listIter->tileSize == tileSize && listIter->thresholdBitWidth==thresholdBitWidth && listIter->indexBitWidth==indexBitWidth)
            break;
        ++listIter;
    }
    assert (listIter != m_tileSizeEntries.end() && "Given tileSize and bit width entry must be present!");
    return listIter;
}

void ForestJSONReader::InitializeBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth, std::vector<int32_t>& treeOffsets) {
    auto listIter = FindEntry(tileSize, thresholdBitWidth, indexBitWidth);
    auto modelCopier = GetModelCopier(tileSize, thresholdBitWidth, indexBitWidth);
    modelCopier->CopyElements(reinterpret_cast<char*>(bufPtr), treeOffsets, listIter->numberOfTiles, listIter->serializedThresholds, 
                              listIter->serializedFetureIndices, listIter->serializedTileShapeIDs, listIter->treeIndices);
}

void ForestJSONReader::InitializeOffsetBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth) {
    IndexType *offsetBuffer = reinterpret_cast<IndexType*>(bufPtr);
    auto listIter = FindEntry(tileSize, thresholdBitWidth, indexBitWidth);
    assert (listIter->numberOfTiles.size() == listIter->treeIndices.size());
    auto currentOffset = 0;
    auto treeIndexIter = listIter->treeIndices.begin();
    std::vector<bool> treeIndexPresent(m_numberOfTrees, false);
    for (auto numTilesIter=listIter->numberOfTiles.begin() ; numTilesIter!=listIter->numberOfTiles.end() ; ++numTilesIter, ++treeIndexIter) {
        offsetBuffer[*treeIndexIter] = currentOffset;
        treeIndexPresent[*treeIndexIter] = true;
        currentOffset += *numTilesIter;
    }
    for (size_t index=0 ; index<treeIndexPresent.size() ; ++index) {
        if (treeIndexPresent[index] == false) {
            offsetBuffer[index] = -1;
        }
    }
}

void ForestJSONReader::InitializeLengthBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth) {
    IndexType *lengthBuffer = reinterpret_cast<IndexType*>(bufPtr);
    auto listIter = FindEntry(tileSize, thresholdBitWidth, indexBitWidth);
    assert (listIter->numberOfTiles.size() == listIter->treeIndices.size());
    auto treeIndexIter = listIter->treeIndices.begin();
    std::vector<bool> treeIndexPresent(m_numberOfTrees, false);
    for (auto numTilesIter=listIter->numberOfTiles.begin() ; numTilesIter!=listIter->numberOfTiles.end() ; ++numTilesIter, ++treeIndexIter) {
        lengthBuffer[*treeIndexIter] = *numTilesIter;
        treeIndexPresent[*treeIndexIter] = true;
    }
    for (size_t index=0 ; index<treeIndexPresent.size() ; ++index) {
        if (treeIndexPresent[index] == false) {
            lengthBuffer[index] = 0;
        }
    }
}

void ForestJSONReader::InitializeLookUpTable(void* bufPtr, int32_t tileSize, int32_t entryBitWidth) {
    assert (entryBitWidth == 8 && "LUT entry must be i8");
    int8_t* lutBufferPtr = reinterpret_cast<int8_t*>(bufPtr);
    TileShapeToTileIDMap tileShapeToTileIDMap(tileSize);
    auto lut = tileShapeToTileIDMap.ComputeTileLookUpTable();
    for (size_t tileShapeID=0 ; tileShapeID<lut.size() ; ++tileShapeID) {
        for (size_t outcome=0 ; outcome<lut.at(tileShapeID).size() ; ++outcome) {
            *lutBufferPtr = lut.at(tileShapeID).at(outcome);
            lutBufferPtr += 1;
        }
    }
}

void ForestJSONReader::InitializeClassInformation(void *classInfoBuf, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth) {
    if (m_numberOfClasses == 0) return;
    
    auto listIter = FindEntry(tileSize, thresholdBitWidth, indexBitWidth);
    assert (listIter->numberOfTiles.size() == listIter->treeIndices.size());

    // #TODO Tree-Beard#19
    int8_t *classInfoBufferPtr = reinterpret_cast<int8_t*>(classInfoBuf);
    int32_t i = 0;
    for (auto x : listIter->classIDs) {
        classInfoBufferPtr[i++] = x;
    }
}


template<typename LeafType>
void ForestJSONReader::InitializeLeavesImpl(LeafType *bufPtr, std::list<ForestJSONReader::SingleTileSizeEntry>::iterator listIter) {
    auto treeIndexIter = listIter->treeIndices.begin();
    std::vector<bool> treeIndexPresent(m_numberOfTrees, false);
    for (auto leavesIter=listIter->serializedLeaves.begin() ; leavesIter!=listIter->serializedLeaves.end() ; ++leavesIter, ++treeIndexIter) {
        auto& leaves = *leavesIter;
        std::copy(leaves.begin(), leaves.end(), bufPtr);
        bufPtr += leaves.size();
    }
}

void ForestJSONReader::InitializeLeaves(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth) {
    auto listIter = FindEntry(tileSize, thresholdBitWidth, indexBitWidth);
    if (thresholdBitWidth == 32) {
        InitializeLeavesImpl(reinterpret_cast<float*>(bufPtr), listIter);
    }
    else if (thresholdBitWidth == 64) {
        InitializeLeavesImpl(reinterpret_cast<double*>(bufPtr), listIter);
    }
    else {
        assert (false && "Unknown threshold type");
    }
}

void ForestJSONReader::InitializeLeavesOffsetBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth) {
    IndexType *offsetBuffer = reinterpret_cast<IndexType*>(bufPtr);
    auto listIter = FindEntry(tileSize, thresholdBitWidth, indexBitWidth);
    auto currentOffset = 0;
    auto treeIndexIter = listIter->treeIndices.begin();
    std::vector<bool> treeIndexPresent(m_numberOfTrees, false);
    for (auto leaveIter=listIter->serializedLeaves.begin() ; leaveIter!=listIter->serializedLeaves.end() ; ++leaveIter, ++treeIndexIter) {
        offsetBuffer[*treeIndexIter] = currentOffset;
        treeIndexPresent[*treeIndexIter] = true;
        currentOffset += (*leaveIter).size();
    }
    for (size_t index=0 ; index<treeIndexPresent.size() ; ++index) {
        if (treeIndexPresent[index] == false) {
            assert (false && "All trees should be present");
            offsetBuffer[index] = -1;
        }
    }
}

void ForestJSONReader::InitializeLeavesLengthBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth) {
    IndexType *lengthBuffer = reinterpret_cast<IndexType*>(bufPtr);
    auto listIter = FindEntry(tileSize, thresholdBitWidth, indexBitWidth);
    auto treeIndexIter = listIter->treeIndices.begin();
    std::vector<bool> treeIndexPresent(m_numberOfTrees, false);
    for (auto leaveIter=listIter->serializedLeaves.begin() ; leaveIter!=listIter->serializedLeaves.end() ; ++leaveIter, ++treeIndexIter) {
        lengthBuffer[*treeIndexIter] = leaveIter->size();
        treeIndexPresent[*treeIndexIter] = true;
    }
    for (size_t index=0 ; index<treeIndexPresent.size() ; ++index) {
        if (treeIndexPresent[index] == false) {
            assert (false && "All trees should be present");
            lengthBuffer[index] = 0;
        }
    }
}


int32_t ForestJSONReader::GetTotalNumberOfTiles() {
    assert (this->m_tileSizeEntries.size() == 1 && "Only a single (tile size, threshold type, feature index type) configuration is supported");
    auto& tileSizeEntry = m_tileSizeEntries.front();
    int32_t numTiles = 0;
    for (auto singleTreeNumTiles : tileSizeEntry.numberOfTiles)
        numTiles += singleTreeNumTiles;
    return numTiles;
}

int32_t ForestJSONReader::GetTotalNumberOfLeaves() {
    assert (this->m_tileSizeEntries.size() == 1 && "Only a single (tile size, threshold type, feature index type) configuration is supported");
    auto& tileSizeEntry = m_tileSizeEntries.front();
    int32_t numLeaves = 0;
    for (auto& singleTreeLeaves : tileSizeEntry.serializedLeaves)
        numLeaves += singleTreeLeaves.size();
    return numLeaves;
}

void LogLeafDepths(const std::vector<int32_t>& depths) {
    std::string out;
    for (auto depth : depths) {
        out += std::to_string(depth) + ", ";
    }
    TreeBeard::Logging::Log(out);
}

void LogTreeStats(const std::vector<TiledTreeStats>& tiledTreeStats) {
    int32_t numDummyNodes=0, numEmptyTiles=0, numLeafNodesInTiledTree=0, numLeavesInOrigModel=0, numTiles=0, numUniqueTiles=0;
    int32_t numNodesInOrigModel=0, numLeavesWithAllSiblingsLeaves=0;
    double_t averageDepth=0.0, avgOrigTreeDepth=0.0;
    for (auto& treeStats : tiledTreeStats) {
        // Total number of dummy nodes
        numDummyNodes += treeStats.numAddedNodes;

        // Extra nodes due to leaves being stored as tiles
        numLeafNodesInTiledTree += treeStats.tiledTreeNumberOfLeafTiles;

        numLeavesInOrigModel += treeStats.originalTreeNumberOfLeaves;

        // Number of "empty tiles"
        int32_t tiledTreeDepth = treeStats.tiledTreeDepth;
        int32_t numChildrenPerTile = treeStats.tileSize + 1;
        int32_t numberOfTiles = (std::pow(numChildrenPerTile, tiledTreeDepth) - 1)/(numChildrenPerTile - 1);

        numEmptyTiles += numberOfTiles - treeStats.numberOfTiles;

        numTiles += treeStats.numberOfTiles;
        numUniqueTiles += treeStats.numberOfUniqueTiles;
        averageDepth += treeStats.tiledTreeDepth;
        avgOrigTreeDepth += treeStats.originalTreeDepth;
        numNodesInOrigModel += treeStats.originalTreeNumNodes;
        numLeavesWithAllSiblingsLeaves += treeStats.numLeavesWithAllLeafSiblings;
        // TreeBeard::Logging::Log(std::to_string(treeStats.numberOfFeatures));
        // LogLeafDepths(treeStats.leafDepths);
    }
    averageDepth /= tiledTreeStats.size();
    avgOrigTreeDepth /= tiledTreeStats.size();

    // TreeBeard::Logging::Log("Number of nodes in original model : " + std::to_string(numNodesInOrigModel));
    // TreeBeard::Logging::Log("Number of leaves in original model : " + std::to_string(numLeavesInOrigModel));
    // TreeBeard::Logging::Log("Number of inserted dummy nodes : " + std::to_string(numDummyNodes));
    // TreeBeard::Logging::Log("Number of leaf tiles : " + std::to_string(numLeafNodesInTiledTree));
    // TreeBeard::Logging::Log("Number of empty tiles : " + std::to_string(numEmptyTiles));
    // TreeBeard::Logging::Log("Number of tiles (assuming duplicated nodes unique) : " + std::to_string(numTiles));
    // TreeBeard::Logging::Log("Number of unique tiles : " + std::to_string(numUniqueTiles));
    // TreeBeard::Logging::Log("Avg tiled tree depth : " + std::to_string(averageDepth));
    // TreeBeard::Logging::Log("Avg original tree depth : " + std::to_string(avgOrigTreeDepth));
    TreeBeard::Logging::Log(std::to_string(numNodesInOrigModel) + ", " + 
                            std::to_string(numLeavesInOrigModel) + ", " +
                            std::to_string(numDummyNodes)+ ", " +
                            std::to_string(numLeafNodesInTiledTree)+ ", " +
                            std::to_string(numEmptyTiles)+ ", " +
                            std::to_string(numTiles)+ ", " +
                            std::to_string(numUniqueTiles)+ ", " +
                            std::to_string(averageDepth)+ ", " +
                            std::to_string(avgOrigTreeDepth)+ ", " +
                            std::to_string(numLeavesWithAllSiblingsLeaves));

}

template<typename T>
void LogTileShapeStats(std::vector<T>& numberOfTileShapes, const std::string& message) {
    // mean, max, min and median
    std::sort(numberOfTileShapes.begin(), numberOfTileShapes.end());
    auto max = numberOfTileShapes.back();
    auto min = numberOfTileShapes.front();
    auto median = numberOfTileShapes.at(numberOfTileShapes.size()/2);
    auto sum = std::accumulate(numberOfTileShapes.begin(), numberOfTileShapes.end(), (T)0);
    auto mean = (double)sum/numberOfTileShapes.size();

    std::string logString = message + " - Min:" + std::to_string(min) + ", Max:" + std::to_string(max) + ", Median:" + std::to_string(median) + ", Mean:" + std::to_string(mean);
    TreeBeard::Logging::Log(logString);
}

template<typename PersistTreeScalarType, typename PersistTreeTiledType>
void PersistDecisionForestImpl(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType,
                               PersistTreeScalarType persistTreeScalar, PersistTreeTiledType persistTreeTiled) {
    
    mlir::decisionforest::ForestJSONReader::GetInstance().ClearAllData();

    auto numTrees = forest.NumTrees();
    mlir::decisionforest::ForestJSONReader::GetInstance().SetNumberOfTrees(numTrees);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetNumberOfClasses(forest.GetNumClasses());

    std::vector<TiledTreeStats> treeStats;
    std::vector<int32_t> numberOfTileShapes, numberOfOriginalTileShapes, numberOfNonSubsetTiles;
    std::vector<double> expectedNumberOfHops, idealExpectedNumberOfHops;
    uint tileShapeBitWidth = 0;
    int32_t childIndexBitWidth = -1;
    for (size_t i=0; i<numTrees ; ++i) {
        auto treeType = forestType.getTreeType(0).cast<decisionforest::TreeType>();
        
        auto currentTreeTileShapeBitWidth = treeType.getTileShapeType().getIntOrFloatBitWidth();
        assert (tileShapeBitWidth==0 || currentTreeTileShapeBitWidth==tileShapeBitWidth);
        tileShapeBitWidth=currentTreeTileShapeBitWidth;

        if (decisionforest::UseSparseTreeRepresentation) {
            int32_t currentChildIndexBitWidth = (int32_t)treeType.getChildIndexType().getIntOrFloatBitWidth();
            assert (childIndexBitWidth==-1 || currentChildIndexBitWidth==childIndexBitWidth);
            childIndexBitWidth=currentChildIndexBitWidth;
        }

        // TODO We're assuming that the threshold type is a float type and index type 
        // is an integer. This is just to get the size. Can we get the size differently?
        // auto thresholdType = treeType.getThresholdType().cast<FloatType>();
        // auto featureIndexType = treeType.getFeatureIndexType().cast<IntegerType>(); 

        auto& tree = forest.GetTree(static_cast<int64_t>(i));
        if (tree.TilingDescriptor().MaxTileSize() == 1) {
            persistTreeScalar(tree, i, treeType);
        }
        else {
            TiledTree& tiledTree = *tree.GetTiledTree();
            std::string dotFile = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/temp/tiledTree_" + std::to_string(i) + ".dot";
            tiledTree.WriteDOTFile(dotFile);
            persistTreeTiled(tiledTree, i, treeType);
            if (TreeBeard::Logging::loggingOptions.logTreeStats) {
                auto tiledTreeStats=tiledTree.GetTreeStats();
                treeStats.push_back(tiledTreeStats);
                numberOfTileShapes.push_back(tiledTree.GetNumberOfTileShapes());
                numberOfOriginalTileShapes.push_back(tiledTree.GetNumberOfOriginalTileShapes());
                auto expectedHops = tiledTree.ComputeExpectedNumberOfTileEvaluations();
                expectedNumberOfHops.push_back(std::get<0>(expectedHops));
                // std::cout << std::get<1>(expectedHops) << " ";
                idealExpectedNumberOfHops.push_back(std::get<1>(expectedHops));
                numberOfNonSubsetTiles.push_back(tiledTree.GetNumberOfTilesThatAreNotSubsets());
            }
        }
    }
    assert (tileShapeBitWidth != 0);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetTileShapeBitWidth(tileShapeBitWidth);
    
    if (decisionforest::UseSparseTreeRepresentation) {
        mlir::decisionforest::ForestJSONReader::GetInstance().SetChildIndexBitWidth(childIndexBitWidth);
    }

    if (TreeBeard::Logging::loggingOptions.logTreeStats) {
        LogTreeStats(treeStats);
        LogTileShapeStats(numberOfTileShapes, "Tile shapes");
        LogTileShapeStats(numberOfOriginalTileShapes, "Original tile shapes");
        LogTileShapeStats(expectedNumberOfHops, "Expected number of hops");
        LogTileShapeStats(idealExpectedNumberOfHops, "Ideal expected number of hops");
        LogTileShapeStats(numberOfNonSubsetTiles, "Non subset tiles");
    }

    mlir::decisionforest::ForestJSONReader::GetInstance().WriteJSONFile();
    // mlir::decisionforest::ForestJSONReader::GetInstance().ParseJSONFile();
    
    // The below lines clear all state that is currently persisted. This is to ensure that 
    // all state is correctly being read back from the model globals json file.
    mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath("");
}

// Ultimately, this will write a JSON file. For now, we're just 
// storing it in memory assuming the compiler and inference 
// will run in the same process. 
void PersistDecisionForestArrayBased(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) {
    PersistDecisionForestImpl(forest, forestType,
            [](DecisionTree<>& tree, int32_t treeNumber, decisionforest::TreeType treeType) {
                std::vector<ThresholdType> thresholds = tree.GetThresholdArray();
                std::vector<FeatureIndexType> featureIndices = tree.GetFeatureIndexArray();
                std::vector<int32_t> tileShapeIDs = { };
                int32_t numTiles = tree.GetNumberOfTiles();
                int32_t tileSize = tree.TilingDescriptor().MaxTileSize();
                mlir::decisionforest::ForestJSONReader::GetInstance().AddSingleTree(
                    treeNumber,
                    numTiles,
                    thresholds,
                    featureIndices,
                    tileShapeIDs,
                    tileSize,
                    treeType.getThresholdType().getIntOrFloatBitWidth(),
                    treeType.getFeatureIndexType().getIntOrFloatBitWidth(),
                    (int8_t)tree.GetClassId());
            },
            [](TiledTree& tiledTree, int32_t treeNumber, decisionforest::TreeType treeType) {
                std::vector<ThresholdType> thresholds = tiledTree.SerializeThresholds();
                std::vector<FeatureIndexType> featureIndices = tiledTree.SerializeFeatureIndices();
                std::vector<int32_t> tileShapeIDs = tiledTree.SerializeTileShapeIDs();
                int32_t numTiles = tiledTree.GetNumberOfTiles();
                int32_t tileSize = tiledTree.TileSize();
                mlir::decisionforest::ForestJSONReader::GetInstance().AddSingleTree(
                    treeNumber,
                    numTiles,
                    thresholds,
                    featureIndices,tileShapeIDs, tileSize,
                    treeType.getThresholdType().getIntOrFloatBitWidth(),
                    treeType.getFeatureIndexType().getIntOrFloatBitWidth(),
                    (int8_t)tiledTree.GetClassId()); // TODO - Support tiled trees.
            }
    );
}

void PersistDecisionForestSparse(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) {
    PersistDecisionForestImpl(forest, forestType,
            [](DecisionTree<>& tree, int32_t treeNumber, decisionforest::TreeType treeType) {
                std::vector<ThresholdType> thresholds = tree.GetSparseThresholdArray(), leaves;
                std::vector<FeatureIndexType> featureIndices = tree.GetSparseFeatureIndexArray();
                std::vector<int32_t> childIndices = tree.GetChildIndexArray();
                std::vector<int32_t> tileShapeIDs = { };
                int32_t numTiles = childIndices.size();
                int32_t tileSize = tree.TilingDescriptor().MaxTileSize();
                int32_t classId = tree.GetClassId();
                mlir::decisionforest::ForestJSONReader::GetInstance().AddSingleSparseTree(treeNumber, numTiles, thresholds, featureIndices, tileShapeIDs, childIndices, 
                                                                                    leaves, tileSize, treeType.getThresholdType().getIntOrFloatBitWidth(), 
                                                                                    treeType.getFeatureIndexType().getIntOrFloatBitWidth(), classId);
            },
            [](TiledTree& tiledTree, int32_t treeNumber, decisionforest::TreeType treeType) {
                std::vector<ThresholdType> thresholds;
                std::vector<FeatureIndexType> featureIndices;
                std::vector<int32_t> tileShapeIDs, leafBitMasks;
                std::vector<int32_t> childIndices, leafIndices;
                std::vector<double> leaves;
                if (decisionforest::RemoveExtraHopInSparseRepresentation)
                    tiledTree.GetSparseSerialization(thresholds, featureIndices, leafBitMasks, tileShapeIDs, childIndices, leafIndices, leaves);
                else    
                    tiledTree.GetSparseSerialization(thresholds, featureIndices, tileShapeIDs, childIndices, leaves);
                int32_t numTiles = tileShapeIDs.size();
                int32_t tileSize = tiledTree.TileSize();
                int32_t classId = tiledTree.GetClassId();
                if (decisionforest::RemoveExtraHopInSparseRepresentation)
                    mlir::decisionforest::ForestJSONReader::GetInstance().AddSingleSparseTree(treeNumber, numTiles, thresholds, 
                                                                    featureIndices, tileShapeIDs, leafBitMasks, childIndices, leafIndices,
                                                                    leaves, tileSize, treeType.getThresholdType().getIntOrFloatBitWidth(), 
                                                                    treeType.getFeatureIndexType().getIntOrFloatBitWidth(), classId);
                else
                    mlir::decisionforest::ForestJSONReader::GetInstance().AddSingleSparseTree(treeNumber, numTiles, thresholds, featureIndices, tileShapeIDs, childIndices,
                                                                                        leaves, tileSize, treeType.getThresholdType().getIntOrFloatBitWidth(), 
                                                                                        treeType.getFeatureIndexType().getIntOrFloatBitWidth(), classId);
            }
    );
}

void PersistDecisionForest(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) {
    if (decisionforest::UseSparseTreeRepresentation)
        PersistDecisionForestSparse(forest, forestType);
    else
        PersistDecisionForestArrayBased(forest, forestType);
}

void ClearPersistedForest() {
    mlir::decisionforest::ForestJSONReader::GetInstance().ClearAllData();
}

int32_t GetTotalNumberOfTiles() {
    return mlir::decisionforest::ForestJSONReader::GetInstance().GetTotalNumberOfTiles();
}

int32_t GetTotalNumberOfLeaves() {
    return mlir::decisionforest::ForestJSONReader::GetInstance().GetTotalNumberOfLeaves();
}

// -----------------------------------------------
// Construction of Tiled Tree
// -----------------------------------------------

// -----------------------------------------------
// Methods for class TiledTreeNode
// -----------------------------------------------
DecisionTree<>& TiledTreeNode::GetTree() { 
    return m_tiledTree.m_modifiedTree;
}

const DecisionTree<>::Node& TiledTreeNode::GetNode(int32_t index) const { 
    return m_tiledTree.m_modifiedTree.GetNodes().at(index);
}

bool TiledTreeNode::AreNodesInSameTile(int32_t node1, int32_t node2) {
    auto& tilingDescriptor = m_tiledTree.m_modifiedTree.TilingDescriptor();
    if(tilingDescriptor.TileIDs().at(node1) == tilingDescriptor.TileIDs().at(node2)) {
        assert (tilingDescriptor.TileIDs().at(node1) == m_tileID);
        return true;
    }
    return false;
}

int32_t TiledTreeNode::FindTileEntryNode() {
    int32_t entryNode = DecisionTree<>::INVALID_NODE_INDEX;
    for (auto nodeIdx : m_nodeIndices) {
        auto& node = GetNode(nodeIdx);
        auto parentIdx = node.parent;
        // Figure out if the parent is in this tile
        bool isRoot = (parentIdx == DecisionTree<>::INVALID_NODE_INDEX);
        bool parentInTile = !isRoot && AreNodesInSameTile(parentIdx, nodeIdx);
        if (!parentInTile) {
            assert (entryNode == DecisionTree<>::INVALID_NODE_INDEX);
            entryNode = nodeIdx;
            // NOT breaking here so we check the rest of the nodes. We should never get back in here.
        }
    }
    assert (entryNode != DecisionTree<>::INVALID_NODE_INDEX);
    return entryNode;
}

void TiledTreeNode::SortTileNodes() {
    std::vector<int32_t> levelOrderSorted;
    levelOrderSorted.reserve(m_nodeIndices.size());
    std::queue<int32_t> traversalQ;
    auto entryNodeIndex = FindTileEntryNode();
    traversalQ.push(entryNodeIndex);
    while (!traversalQ.empty()) {
        int32_t nodeIndex = traversalQ.front();
        auto& node = GetNode(nodeIndex);
        traversalQ.pop();
        assert (AreNodesInSameTile(entryNodeIndex, nodeIndex));
        levelOrderSorted.push_back(nodeIndex);
        // If the node is not a leaf, then it must have two valid children
        if (node.IsLeaf()) 
            continue;
        if (AreNodesInSameTile(node.leftChild, nodeIndex)) {
            assert (std::find(m_nodeIndices.begin(), m_nodeIndices.end(), node.leftChild) != m_nodeIndices.end());
            traversalQ.push(node.leftChild);
        }
        if (AreNodesInSameTile(node.rightChild, nodeIndex)) {
            assert (std::find(m_nodeIndices.begin(), m_nodeIndices.end(), node.rightChild) != m_nodeIndices.end());
            traversalQ.push(node.rightChild);
        }
    }
    assert(m_nodeIndices.size() == levelOrderSorted.size());
    m_nodeIndices = levelOrderSorted;
}

void TiledTreeNode::AddExtraNodesIfNeeded() {
    // How do we add the extra nodes in the right places in the vector? We need
    // to maintain level order!
    if (static_cast<int32_t>(m_nodeIndices.size()) == m_tiledTree.m_modifiedTree.TilingDescriptor().MaxTileSize())
        return;
    // If the only node in the tile is a leaf, then just return
    if (static_cast<int32_t>(m_nodeIndices.size()) == 1 && 
        GetNode(m_nodeIndices.at(0)).IsLeaf()) {
        return;
    }
    m_hasExtraNodes = true;
    assert (static_cast<int32_t>(m_nodeIndices.size()) < m_tiledTree.m_modifiedTree.TilingDescriptor().MaxTileSize());
    do {
      int32_t numberOfNodesToAdd = m_tiledTree.m_modifiedTree.TilingDescriptor().MaxTileSize() - static_cast<int32_t>(m_nodeIndices.size());
      // TODO Must there be at least one node where both children are leaves?
      // This must be true
      // 1. The child of any node must either be in the same tile or must be a leaf (if it were not, the tile could have been larger)
      // 2. Since the tile can't grow indefinitely, #1 => there is at least one node with both children being leaves
      std::list<int32_t> candidateNodes;
      for (auto nodeIndex : m_nodeIndices) {
          auto& node = GetNode(nodeIndex);
          auto leftChildIndex = node.leftChild;
          auto rightChildIndex = node.rightChild;
          auto& leftChild = GetNode(leftChildIndex);
          auto& rightChild = GetNode(rightChildIndex);
          assert (AreNodesInSameTile(nodeIndex, leftChildIndex) || leftChild.IsLeaf());
          assert (AreNodesInSameTile(nodeIndex, rightChildIndex) || rightChild.IsLeaf());
          if (leftChild.IsLeaf() || rightChild.IsLeaf())
              candidateNodes.push_back(nodeIndex);
      }
      assert (candidateNodes.size() > 0);
      // TODO How do we determine the shape of this tile once we add new nodes? Maybe some kind of look up based on the positions of the nodes in the 
      // full dense serialization?
      // TODO How do we decide which of the candidate nodes to use as the parent of the new node(s)? For now, picking from the first candidate, which will 
      // be the right most node on the bottom most level. 
      auto candidateIter = candidateNodes.begin();
      for (int32_t i=0 ; i<numberOfNodesToAdd && candidateIter!=candidateNodes.end(); ) { // We can add two dummy nodes for every candidate node
          // TODO How do we decide where to add the new nodes? Maybe just add them somewhere and call sort again?
          assert (candidateIter != candidateNodes.end());
          auto candidateIndex = *candidateIter;
          auto candidateNode = GetNode(candidateIndex);
          if (GetNode(candidateNode.leftChild).IsLeaf()){
            auto leafIndex = candidateNode.leftChild;
            auto &leafNode = GetNode(leafIndex);
            assert(leafNode.IsLeaf());
            // Add the dummy node as the left child of the candidate
            auto dummyNode = m_tiledTree.m_modifiedTree.NewNode(candidateNode.threshold, candidateNode.featureIndex, m_tileID);
            m_tiledTree.m_modifiedTree.SetNodeLeftChild(dummyNode, leafIndex);
            m_tiledTree.m_modifiedTree.SetNodeRightChild(dummyNode, leafIndex);
            m_tiledTree.m_modifiedTree.SetNodeParent(leafIndex, dummyNode);

            m_tiledTree.m_modifiedTree.SetNodeParent(dummyNode, candidateIndex);
            m_tiledTree.m_modifiedTree.SetNodeLeftChild(candidateIndex, dummyNode);
            
            m_tiledTree.AddNodeToTile(m_tileIndex, dummyNode);
            const_cast<mlir::decisionforest::DecisionTree<>::Node&>(GetNode(dummyNode)).hitCount = -1;
            ++i;
          }

          if (i == numberOfNodesToAdd)
              break;

          if (GetNode(candidateNode.rightChild).IsLeaf()) {
            auto leafIndex = candidateNode.rightChild;
            auto &leafNode = GetNode(leafIndex);
            assert(leafNode.IsLeaf());
            // Add the dummy node as the right child of the candidate
            auto dummyNode = m_tiledTree.m_modifiedTree.NewNode(candidateNode.threshold, candidateNode.featureIndex, m_tileID);
            
            m_tiledTree.m_modifiedTree.SetNodeLeftChild(dummyNode, leafIndex);
            m_tiledTree.m_modifiedTree.SetNodeRightChild(dummyNode, leafIndex);
            m_tiledTree.m_modifiedTree.SetNodeParent(leafIndex, dummyNode);
            
            m_tiledTree.m_modifiedTree.SetNodeParent(dummyNode, candidateIndex);
            m_tiledTree.m_modifiedTree.SetNodeRightChild(candidateIndex, dummyNode);

            m_tiledTree.AddNodeToTile(m_tileIndex, dummyNode);
            const_cast<mlir::decisionforest::DecisionTree<>::Node&>(GetNode(dummyNode)).hitCount = -1;
            ++i;
          }
          ++candidateIter;
      }
    } while (static_cast<int32_t>(m_nodeIndices.size()) < m_tiledTree.m_modifiedTree.TilingDescriptor().MaxTileSize());
    assert (static_cast<int32_t>(m_nodeIndices.size()) == m_tiledTree.m_modifiedTree.TilingDescriptor().MaxTileSize());
    SortTileNodes();
    m_tiledTree.SetChildrenForTile(*this);
}

void TiledTreeNode::GetThresholds(std::vector<double>::iterator beginIter) {
    if (m_nodeIndices.size() == 1) {
        // This is a leaf tile
        auto& node = GetNode(m_nodeIndices.front());
        assert (node.IsLeaf() && "A tile with a single node can only contain a leaf");
        int32_t tileSize = m_tiledTree.m_modifiedTree.TilingDescriptor().MaxTileSize();
        auto threshold = node.threshold;
        for (int32_t i=0; i<tileSize ; ++i) {
            *beginIter = threshold;
            ++beginIter;
        }
        return;
    }
    assert (static_cast<int32_t>(m_nodeIndices.size()) == m_tiledTree.m_modifiedTree.TilingDescriptor().MaxTileSize());
    for (auto nodeIndex : m_nodeIndices) {
        auto threshold = GetNode(nodeIndex).threshold;
        *beginIter = threshold;
        ++beginIter;
    }
}

void TiledTreeNode::GetFeatureIndices(std::vector<int32_t>::iterator beginIter) {
    if (m_nodeIndices.size() == 1) {
        // This is a leaf tile
        auto& node = GetNode(m_nodeIndices.front());
        assert (node.IsLeaf() && "A tile with a single node can only contain a leaf");
        int32_t tileSize = m_tiledTree.m_modifiedTree.TilingDescriptor().MaxTileSize();
        auto featureIndex = -1; // node.featureIndex;
        for (int32_t i=0; i<tileSize ; ++i) {
            *beginIter = featureIndex;
            ++beginIter;
        }
        return;
    }
    assert (static_cast<int32_t>(m_nodeIndices.size()) == m_tiledTree.m_modifiedTree.TilingDescriptor().MaxTileSize());
    for (auto nodeIndex : m_nodeIndices) {
        auto featureIndex = GetNode(nodeIndex).featureIndex;
        *beginIter = featureIndex;
        ++beginIter;
    }
}

void TiledTreeNode::ComputeTileShapeString(std::string& str, int32_t tileNodeIndex, int32_t stringIndex) {
    str.at(stringIndex) = '1';
    auto& node = GetNode(tileNodeIndex);
    if (node.leftChild!=DecisionTree<>::INVALID_NODE_INDEX && AreNodesInSameTile(tileNodeIndex, node.leftChild)) {
        ComputeTileShapeString(str, node.leftChild, 2*stringIndex + 1);
    }
    if (node.rightChild!=DecisionTree<>::INVALID_NODE_INDEX && AreNodesInSameTile(tileNodeIndex, node.rightChild)) {
        ComputeTileShapeString(str, node.rightChild, 2*stringIndex + 2);
    }
    return;
}

std::string TiledTreeNode::GetTileShapeString() {
    auto tileSize = m_tiledTree.m_modifiedTree.TilingDescriptor().MaxTileSize();
    auto stringSize = std::pow(2, tileSize) - 1;
    std::string tileShapeStr(stringSize, '0');
    ComputeTileShapeString(tileShapeStr, GetEntryNode(), 0);
    return tileShapeStr;
}

void TiledTreeNode::WriteDOTSubGraph(std::ofstream& fout) {
    std::vector<std::string> colors = { "aquamarine3", "darkolivegreen4", "deepskyblue", "firebrick", "grey80", "teal"};
    std::string& color = colors[m_tileID % colors.size()];

    fout << "subgraph tile_" << m_tileID << " {\n";
    fout << "\tnode [style=filled, ";
    fout << "color=" << color << "];\n";
    for (size_t i=0 ; i<m_nodeIndices.size() ; ++i) {
        auto nodeIndex = m_nodeIndices.at(i);
        auto& node = GetNode(nodeIndex);
        int64_t parentIndex = node.parent;
        auto& parentNode = GetNode(parentIndex);
        fout << "\t\"node" << nodeIndex << "\" [ label = \"Id:" << nodeIndex << ", Thres:" << node.threshold << ", FeatIdx:" << node.featureIndex << "\"];\n";
        if (parentIndex != DecisionTree<>::INVALID_NODE_INDEX) {
            std::string edgeColor = parentNode.leftChild == nodeIndex ? "green" : "red";
            if (parentNode.leftChild == parentNode.rightChild)
                edgeColor = "black";
            fout << "\t\"node" << parentIndex << "\" -> \"node" << nodeIndex << "\"[style=bold,color=" << edgeColor << "];\n";
        }
    }
    fout << "}\n";
}

// -----------------------------------------------
// Methods for class TiledTree
// -----------------------------------------------

TiledTree::TiledTree(DecisionTree<>& owningTree)
 : m_numberOfDummyTiles(0), m_owningTree(owningTree), m_modifiedTree(owningTree), 
   m_tileShapeToTileIDMap(m_owningTree.TilingDescriptor().MaxTileSize())
{
    ConstructTiledTree();
}

void TiledTree::SetChildrenHelper(TiledTreeNode& tile, int32_t nodeIndex, std::vector<int32_t>& children) {
    auto &node = m_modifiedTree.GetNodes().at(nodeIndex);
    if (node.IsLeaf()) return;
    if (tile.AreNodesInSameTile(nodeIndex, node.leftChild))
        SetChildrenHelper(tile, node.leftChild, children);
    else
        children.push_back(GetNodeTileIndex(node.leftChild));
    if (tile.AreNodesInSameTile(nodeIndex, node.rightChild))
        SetChildrenHelper(tile, node.rightChild, children);
    else
        children.push_back(GetNodeTileIndex(node.rightChild));
}

void TiledTree::SetChildrenForTile(TiledTreeNode& tile) {
    tile.m_children.clear();
    SetChildrenHelper(tile, tile.GetEntryNode(), tile.m_children);
}

bool IsTileASubset(const std::string& subtile, const std::string& tile) {
    assert (subtile.size() == tile.size());
    for (size_t i=0 ; i<subtile.size() ; ++i) {
        if (subtile.at(i) == '1') {
            if (tile.at(i) == '0')
                return false;
        }
    }
    return true;
}

void TiledTree::ConstructTiledTree() {
    const TreeTilingDescriptor& tilingDescriptor = m_modifiedTree.TilingDescriptor();
    auto& tileIDs = tilingDescriptor.TileIDs();
    std::map<int32_t, int32_t> tileIDToTileIndexMap;
    assert(m_modifiedTree.GetNodes().size() == tileIDs.size());
    // First, split the nodes among tiles
    int32_t nodeIndex=0;
    for (auto tileID : tileIDs) {
        auto tileIDMapIter = tileIDToTileIndexMap.find(tileID);
        int32_t tileIndex = -1;
        if (tileIDMapIter == tileIDToTileIndexMap.end()) {
            tileIndex = NewTiledTreeNode(tileID);
            tileIDToTileIndexMap[tileID] = tileIndex;
        }
        else {
            tileIndex = tileIDMapIter->second;
        }
        AddNodeToTile(tileIndex, nodeIndex);
        ++nodeIndex;
    }
    // Sort the nodes in each tile
    for (auto& tile : m_tiles)
        tile.SortTileNodes();
    // Set the parents and the children of each tile
    for (auto& tile : m_tiles) {
        auto entryNode = tile.GetEntryNode();
        auto parentNode = m_modifiedTree.GetNodes().at(entryNode).parent;
        auto parentTileIndex = GetNodeTileIndex(parentNode);
        tile.SetParent(parentTileIndex);

        SetChildrenForTile(tile);
    }
    
    std::set<std::string> tileShapeStrings;
    for (auto& tile : m_tiles) {
        if (tile.IsLeafTile() || (int32_t)tile.m_nodeIndices.size() != m_modifiedTree.TilingDescriptor().MaxTileSize())
            continue;
        std::string tileString = tile.GetTileShapeString();
        tileShapeStrings.insert(tileString);
    }

    m_numTilesThatAreNotSubsets=0;
    for (auto& tile : m_tiles) {
        // Only iterate through incomplete tiles
        if (tile.IsLeafTile() || (int32_t)tile.m_nodeIndices.size() == m_modifiedTree.TilingDescriptor().MaxTileSize())
            continue;
        std::string tileString = tile.GetTileShapeString();
        bool isSubset = false;
        for (auto& fullTileString : tileShapeStrings) {
            if (IsTileASubset(tileString, fullTileString)) {
                isSubset = true;
                break;
            }
        }
        m_numTilesThatAreNotSubsets += isSubset ? 0 : 1;
    }

    // Expand the tiles that aren't full with dummy nodes
    for (auto& tile : m_tiles) {
        tile.AddExtraNodesIfNeeded();
    }
    
    // Set the shape ID of all the tiles in the tree
    std::set<int32_t> tileShapeIDs, originalTileShapeIDs;
    for (auto& tile : m_tiles) {
        tile.m_tileShapeID = m_tileShapeToTileIDMap.GetTileID(tile);
        tileShapeIDs.insert(tile.m_tileShapeID);
        if (!tile.m_hasExtraNodes)
            originalTileShapeIDs.insert(tile.m_tileShapeID);
    }
    m_numberOfTileShapes = tileShapeIDs.size();
    m_originalNumberOfTileShapes = originalTileShapeIDs.size();
    assert(Validate());
}

bool TiledTree::Validate() {
    // Checks
    //  - All nodes in the owning tree are covered and contained in exactly one tile
    //  - All tiles are the same size or are leaves (with size == 1)
    //  - Leaves are not part of tiles
    //  - TODO Tiles are connected (traverse from entry tile and assert all nodes are reachable)
    std::vector<int32_t> nodeCounts(m_modifiedTree.GetNodes().size(), 0);
    for (auto& tile: m_tiles) {
        auto& tileNodeIndices = tile.GetNodeIndices();
        for (auto nodeIndex : tileNodeIndices) {
            if (nodeIndex >= static_cast<int32_t>(nodeCounts.size()))
                continue;
            nodeCounts.at(nodeIndex)++;
        }
        int32_t maxTileSize = m_modifiedTree.TilingDescriptor().MaxTileSize();
        if (tileNodeIndices.size() == 1) {
            if (!m_modifiedTree.GetNodes().at(tileNodeIndices.front()).IsLeaf()) {
                assert (false && "Node in tile with a single node must be a leaf");
                return false;
            }
        }
        else {
            if (static_cast<int32_t>(tileNodeIndices.size()) != maxTileSize) {
                assert (false && "Tile sizes must be equal except for leaf nodes");
                return false;
            }
            for (auto nodeIndex : tileNodeIndices) {
                // A node that is a non-unit tile must not be a leaf
                if (m_modifiedTree.GetNodes().at(nodeIndex).IsLeaf()) {
                    assert(false && "A node that is a non-unit tile must not be a leaf");
                    return false;
                }
            }
        }
        assert (tile.m_children.size() == 0 || (int32_t)(tile.m_children.size()) == (maxTileSize+1));
    }
    for (auto nodeCount : nodeCounts) {
        if (nodeCount!=1) {
            assert (false && "Node must be in exactly one tile");
            return false;
        }
    }
    return true;
}

void TiledTree::EmitNodeDOT(std::ofstream& fout, int32_t nodeIndex) {
    std::vector<std::string> colors = { "aquamarine3", "darkolivegreen4", "deepskyblue", "firebrick", "grey80", "teal"};

    auto& node = m_modifiedTree.GetNodes().at(nodeIndex);
    int32_t tileID = this->GetNodeTileIndex(nodeIndex);
    std::string& color = colors[tileID % colors.size()];

    int64_t parentIndex = node.parent;
    fout << "\t\"node" << nodeIndex << "\" [ label = \"Id:" << nodeIndex
        << ", Thres:" << node.threshold
        << ", FeatIdx:" << node.featureIndex 
        << ", Hits:" << node.hitCount
        << ", TileID:" << tileID
        << "\", style=bold, color=" << color << "];\n";
    if (parentIndex != decisionforest::DecisionTree<>::INVALID_NODE_INDEX) {
        auto& parentNode = m_modifiedTree.GetNodes().at(parentIndex);
        std::string edgeColor = parentNode.leftChild == nodeIndex ? "green" : "red";
        if (parentNode.leftChild == parentNode.rightChild)
            edgeColor = "black";
        fout << "\t\"node" << parentIndex << "\" -> \"node" << nodeIndex << "\"[style=bold,color=" << edgeColor << "];\n";
        // fout << "\t\"node" << parentIndex << "\" -> \"node" << nodeIndex << "\";\n";
    }
    if (node.leftChild != decisionforest::DecisionTree<>::INVALID_NODE_INDEX)
        EmitNodeDOT(fout, node.leftChild);
    if (node.rightChild != decisionforest::DecisionTree<>::INVALID_NODE_INDEX)
        EmitNodeDOT(fout, node.rightChild);
}

// Routines to output DOT files for the tiled tree
void TiledTree::WriteDOTFile(const std::string& filename) {
    std::ofstream fout(filename);
#ifdef EMIT_TILES_AS_SUBGRAPHS
    fout << "digraph {\n";
    for (size_t i=0 ; i<m_tiles.size() ; ++i) {
        m_tiles[i].WriteDOTSubGraph(fout);
    }
    fout << "}\n";
#else // EMIT_TILES_AS_SUBGRAPHS
    fout << "digraph {\n";
    // TODO This assumes the root is the first node
    EmitNodeDOT(fout, 0);
    fout << "}\n";

#endif // EMIT_TILES_AS_SUBGRAPHS
}

int32_t TiledTree::GetTreeDepthHelper(int32_t tileIndex) {
    auto& tile = m_tiles.at(tileIndex);
    int32_t depth = 0;
    for (auto child : tile.GetChildren()) {
        depth = std::max(depth, GetTreeDepthHelper(child));
    }
    return 1+depth;
}

template <typename AttribType, typename GetterType>
void TiledTree::GetTileAttributeArray(std::vector<AttribType>& attributeVec,
                                      size_t vecIndex, size_t tileIndex, GetterType get, bool singleAttribute)
{
    auto& tile = m_tiles.at(tileIndex);
    assert(vecIndex < attributeVec.size());
    // TODO What is the type we set on leaf nodes?
    // assert(node.featureType == FeatureType::kNumerical || node.IsLeaf());
    auto tileSize = m_owningTree.TilingDescriptor().MaxTileSize();
    get(tile, attributeVec.begin() + vecIndex * (singleAttribute ? 1 : tileSize));
    const auto& children = tile.GetChildren();
    int32_t numChildren = tileSize + 1;
    for (size_t i=0 ; i<children.size() ; ++i) {
        int32_t childIndex = children[i];
        GetTileAttributeArray<AttribType, GetterType>(attributeVec, numChildren*vecIndex+i+1, childIndex, get, singleAttribute);
    }
}

std::vector<double> TiledTree::SerializeThresholds() {
    int32_t tiledTreeDepth = GetTreeDepth();
    int32_t numChildrenPerTile = m_owningTree.TilingDescriptor().MaxTileSize() + 1;
    int32_t numberOfTiles = (std::pow(numChildrenPerTile, tiledTreeDepth) - 1)/(numChildrenPerTile - 1);
    int32_t vectorLength = numberOfTiles * m_owningTree.TilingDescriptor().MaxTileSize();
    std::vector<double> thresholds(vectorLength, -1);
    GetTileAttributeArray(thresholds, 0, 0, [&](TiledTreeNode& t, std::vector<double>::iterator iter){ t.GetThresholds(iter); }, false /*singleAttribute*/ );
    return thresholds;
}

std::vector<int32_t> TiledTree::SerializeFeatureIndices() {
    int32_t tiledTreeDepth = GetTreeDepth();
    int32_t numChildrenPerTile = m_owningTree.TilingDescriptor().MaxTileSize() + 1;
    int32_t numberOfTiles = (std::pow(numChildrenPerTile, tiledTreeDepth) - 1)/(numChildrenPerTile - 1);
    int32_t vectorLength = numberOfTiles * m_owningTree.TilingDescriptor().MaxTileSize();
    std::vector<int32_t> featureIndices(vectorLength, -1);
    GetTileAttributeArray(featureIndices, 0, 0, [&](TiledTreeNode& t, std::vector<int32_t>::iterator iter){ t.GetFeatureIndices(iter); }, false /*singleAttribute*/ );
    return featureIndices;
}

std::vector<int32_t> TiledTree::SerializeTileShapeIDs() {
    int32_t tiledTreeDepth = GetTreeDepth();
    int32_t numChildrenPerTile = m_owningTree.TilingDescriptor().MaxTileSize() + 1;
    int32_t numberOfTiles = (std::pow(numChildrenPerTile, tiledTreeDepth) - 1)/(numChildrenPerTile - 1);
    std::vector<int32_t> tileShapeIDs(numberOfTiles, -1);
    GetTileAttributeArray(tileShapeIDs, 0, 0, [&](TiledTreeNode& t, std::vector<int32_t>::iterator iter){ *iter=t.GetTileShapeID(); }, true /*singleAttribute*/ );
    return tileShapeIDs;
}

void GetSparseSerialization(std::vector<double>& thresholds, std::vector<int32_t>& featureIndices, 
                            std::vector<int32_t>& tileShapeIDs, std::vector<int32_t>& childIndices);

int32_t TiledTree::NumberOfLeafTilesHelper(int32_t tileIndex) {
    if (m_tiles.at(tileIndex).IsLeafTile())
        return 1;
    auto& tile = m_tiles.at(tileIndex);
    int32_t numLeafTiles = 0;
    for (auto childIndex : tile.m_children) {
        numLeafTiles += NumberOfLeafTilesHelper(childIndex);
    }
    return numLeafTiles;
}

int32_t TiledTree::NumberOfLeafTiles() {
    if (m_tiles.size() == 0)
        return 0;
    assert(m_tiles.at(0).m_parent == DecisionTree<>::INVALID_NODE_INDEX);
    int32_t numLeafTiles = NumberOfLeafTilesHelper(0);
    return numLeafTiles;
}

bool TiledTree::AreAllSiblingsLeaves(TiledTreeNode& tile, const std::vector<TiledTreeNode>& tiles) {
    if (!tile.IsLeafTile())
        return false;
    auto parentIdx = tile.GetParent();
    if (parentIdx == DecisionTree<>::INVALID_NODE_INDEX)
        return true; // This is the only node in the tree
    auto& parent = tiles.at(parentIdx);
    for (auto childIdx : parent.GetChildren()) {
        auto& child = tiles.at(childIdx);
        if (!child.IsLeafTile())
            return false;
    }
    return true;
}

int32_t TiledTree::NumberOfLeavesWithAllLeafSiblings(int32_t tileIndex) {
    if (m_tiles.at(tileIndex).IsLeafTile()) {
        auto tile = m_tiles.at(tileIndex);
        if (AreAllSiblingsLeaves(tile, m_tiles))
            return 1;
        else
            return 0;
    }
    auto& tile = m_tiles.at(tileIndex);
    int32_t numLeafTiles = 0;
    for (auto childIndex : tile.m_children) {
        numLeafTiles += NumberOfLeavesWithAllLeafSiblings(childIndex);
    }
    return numLeafTiles;
}

int32_t TiledTree::NumberOfTiles() {
    int32_t numTiles = 0;
    for (auto& tile : m_tiles){
        if (!tile.IsLeafTile())
            ++numTiles;
    }
    numTiles += NumberOfLeafTiles();
    return numTiles;
}

std::vector<int32_t> TiledTree::GetLeafDepths() {
    std::vector<int32_t> depths;
    for (auto& tile : m_tiles) {
        if (!tile.IsLeafTile())
            continue;
        int32_t depth = 0;
        auto currentTile = &tile;
        while (currentTile->m_parent != DecisionTree<>::INVALID_NODE_INDEX) {
            currentTile = &(m_tiles.at(currentTile->m_parent));
            ++depth;
        }
        depths.push_back(depth);
    }
    return depths;
}

TiledTreeStats TiledTree::GetTreeStats() {
    TiledTreeStats treeStats;
    treeStats.tileSize = TileSize();
    treeStats.originalTreeDepth = m_owningTree.GetTreeDepth();
    treeStats.originalTreeNumNodes = m_owningTree.GetNodes().size();
    treeStats.tiledTreeDepth = GetTreeDepth();
    treeStats.tiledTreeNumNodes = m_tiles.size() * m_owningTree.TilingDescriptor().MaxTileSize();
    treeStats.numAddedNodes = m_modifiedTree.GetNodes().size() - m_owningTree.GetNodes().size();
    treeStats.numberOfUniqueTiles = m_tiles.size();
    treeStats.numberOfTiles = NumberOfTiles();
    treeStats.originalTreeNumberOfLeaves = m_owningTree.NumLeaves();
    treeStats.tiledTreeNumberOfLeafTiles = NumberOfLeafTiles();
    treeStats.numLeavesWithAllLeafSiblings = NumberOfLeavesWithAllLeafSiblings(0 /*root tile index*/ );
    treeStats.numberOfFeatures = m_owningTree.NumFeatures();
    treeStats.leafDepths = GetLeafDepths();
    return treeStats;
}

// TODO this is not valid for the following reasons
// 1. The parent index of leaves is not valid because of the way we're adding dummy nodes to tiles and so AreAllSiblingsLeaves is not correct
//      - Since the parent index of the tile containing the leaf is correct, it shouldn't matter
// 2. If we get to a duplicated leaf, the duplicates should have a zero hit count so that we don't count its contribution multiple times
void TiledTree::ExpectedNumberOfTileEvaluations(double& actualVal, double& idealVal, int32_t currentTile, int32_t depth, std::set<int32_t>& visitedLeaves) {
    auto& tile = m_tiles.at(currentTile);
    if (tile.IsLeafTile() && visitedLeaves.find(currentTile)==visitedLeaves.end()) {
        int32_t programmaticDepth = AreAllSiblingsLeaves(tile, m_tiles) ? depth+1 : depth+2;
        auto totalCount = tile.m_owningTree.GetNodes().at(0).hitCount;
        actualVal += (double)tile.GetNode(tile.m_nodeIndices.front()).hitCount/(double)totalCount * programmaticDepth;
        idealVal += (double)tile.GetNode(tile.m_nodeIndices.front()).hitCount/(double)totalCount * (depth+1);
        visitedLeaves.insert(currentTile);
        return;
    }
    for (auto child : tile.m_children) {
        ExpectedNumberOfTileEvaluations(actualVal, idealVal, child, depth+1, visitedLeaves);
    }
}

std::tuple<double, double> TiledTree::ComputeExpectedNumberOfTileEvaluations() {
    double actualValue = 0.0, idealValue=0.0;
    std::set<int32_t> visitedLeaves;
    ExpectedNumberOfTileEvaluations(actualValue, idealValue, 0, 0, visitedLeaves);
    return std::make_tuple(actualValue, idealValue);
}

void TiledTree::GetSparseSerialization(std::vector<double>& thresholds, std::vector<int32_t>& featureIndices, 
                                       std::vector<int32_t>& tileShapeIDs, std::vector<int32_t>& childIndices,
                                       std::vector<double>& leaves) {
    thresholds.clear(); featureIndices.clear(); tileShapeIDs.clear(); childIndices.clear(); leaves.clear();

    TiledTree::LevelOrderTraversal levelOrder(m_tiles);
    auto& sortedTiles = levelOrder.LevelOrderNodes();
    std::map<int32_t, int32_t> tileIndexMap;
    int32_t numberOfTilesInLeafArray=0, currentTileIndex=0;
    std::vector<double> tileThresholds(TileSize());
    std::vector<int32_t> tileFeatureIndices(TileSize());
    if (!decisionforest::OptimizedSparseRepresentation)
        tileIndexMap[-1] = -1;
    std::list<int32_t> leafTileIndices, leafArrayIndex;
    for (auto& tile : sortedTiles) {
        // if tile is a leaf and all siblings are leaves, put it into the leaf array
        if (tile.IsLeafTile() && AreAllSiblingsLeaves(tile, sortedTiles)) {
            int32_t leafArrayIndex = static_cast<int32_t>(leaves.size());
            leaves.push_back(tile.GetNode(tile.GetNodeIndices().front()).threshold);
            tileIndexMap[currentTileIndex] = -(leafArrayIndex+1); // HACK!
            numberOfTilesInLeafArray += 1;
        }
        else {
            if (!decisionforest::OptimizedSparseRepresentation) {
                tile.GetThresholds(tileThresholds.begin());
                thresholds.insert(thresholds.end(), tileThresholds.begin(), tileThresholds.end());
                tile.GetFeatureIndices(tileFeatureIndices.begin());
                featureIndices.insert(featureIndices.end(), tileFeatureIndices.begin(), tileFeatureIndices.end());
                tileShapeIDs.push_back(tile.GetTileShapeID());
                if (!tile.IsLeafTile())
                    childIndices.push_back(tile.GetChildren().front());
                else
                    childIndices.push_back(-1);
                tileIndexMap[currentTileIndex] = currentTileIndex - numberOfTilesInLeafArray;
            }
            else {
                if (!tile.IsLeafTile()) {
                    tile.GetThresholds(tileThresholds.begin());
                    thresholds.insert(thresholds.end(), tileThresholds.begin(), tileThresholds.end());
                    tile.GetFeatureIndices(tileFeatureIndices.begin());
                    featureIndices.insert(featureIndices.end(), tileFeatureIndices.begin(), tileFeatureIndices.end());
                    tileShapeIDs.push_back(tile.GetTileShapeID());
                    childIndices.push_back(tile.GetChildren().front());
                    tileIndexMap[currentTileIndex] = currentTileIndex - numberOfTilesInLeafArray;
                }
                else {
                    // Make this tile a dummy tile with all child leaves having the same value
                    tile.GetThresholds(tileThresholds.begin());
                    thresholds.insert(thresholds.end(), tileThresholds.begin(), tileThresholds.end());
                    std::vector<int32_t> leafFeatureIndices(TileSize(), 0);
                    featureIndices.insert(featureIndices.end(), leafFeatureIndices.begin(), leafFeatureIndices.end());
                    tileShapeIDs.push_back(0); //tile.GetTileShapeID());
                    
                    // Change this to some unique ID (maybe negative of the current length of the leaf array?)
                    auto childIndex = -static_cast<int32_t>(leaves.size())-1;
                    assert (tileIndexMap.find(childIndex) == tileIndexMap.end());
                    childIndices.push_back(childIndex);
                    tileIndexMap[childIndex] = childIndex; // HACK
                    std::vector<double> leafValVec(TileSize()+1, tileThresholds.front());
                    leaves.insert(leaves.end(), leafValVec.begin(), leafValVec.end());

                    // We also need to insert a look up for this leaf
                    tileIndexMap[currentTileIndex] = currentTileIndex - numberOfTilesInLeafArray;

                    leafTileIndices.push_back(childIndices.size() - 1);
                }
            }
        }
        ++currentTileIndex;
    }

    assert ((sortedTiles.size() - numberOfTilesInLeafArray) == childIndices.size());
    for (size_t i=0 ; i<childIndices.size() ; ++i) {
        // if (childIndices.at(i) == -1)
        //     continue;
        auto mapIter = tileIndexMap.find(childIndices.at(i));
        assert (mapIter != tileIndexMap.end());
        auto newChildIndex = mapIter->second;
        if (newChildIndex < 0) {
            newChildIndex = static_cast<int32_t>(childIndices.size()) + (-newChildIndex-1); // HACK
        }
        assert (newChildIndex >= 0);
        childIndices.at(i) = newChildIndex;
    }

    // Error check for optimized sparse representation
    if (decisionforest::OptimizedSparseRepresentation)
        for (auto leafIndex : leafTileIndices) {
            int32_t thresholdIndex = leafIndex * TileSize();
            int32_t childIndex = childIndices.at(leafIndex);

            auto threshold = thresholds.at(thresholdIndex);
            auto leafArrayVal = leaves.at(childIndex - childIndices.size());
            assert (threshold == leafArrayVal);
        }
}

bool TiledTree::HasLeafSiblings(TiledTreeNode& tile, std::vector<TiledTreeNode>& sortedTiles) {
    if (tile.GetParent() == decisionforest::DecisionTree<>::INVALID_NODE_INDEX) {
        // This is the root. So it doesn't have siblings.
        return false;
    }
    auto& parent = sortedTiles.at(tile.GetParent());
    for (auto& childIndex : parent.GetChildren()) {
        if (sortedTiles.at(childIndex).IsLeafTile()) {
            return true;
        }
    }
    return false;
}

// Compute a bitmask where there is one bit for each child of the tile. 
// If the child is a leaf, its bit is set to 1, 0 otherwise. 
int32_t TiledTree::GetChildrenBitMask(TiledTreeNode& tile, std::vector<TiledTreeNode>& sortedTiles) {
    int32_t bitmask = 0;
    int32_t childNumber = 0;
    for (auto& child : tile.GetChildren()) {
        auto& childTile = sortedTiles.at(child);
        if (childTile.IsLeafTile())
            bitmask |= (1 << childNumber);
        ++childNumber;
    }
    return bitmask;
}

void TiledTree::GetSparseSerialization(std::vector<double>& thresholds, std::vector<int32_t>& featureIndices, std::vector<int32_t>& leafBitMasks,
                                       std::vector<int32_t>& tileShapeIDs, std::vector<int32_t>& childIndices, std::vector<int32_t>& leafIndices,
                                       std::vector<double>& leaves) {
    thresholds.clear(); featureIndices.clear(); tileShapeIDs.clear(); childIndices.clear(); 
    leaves.clear(); leafIndices.clear(); leafBitMasks.clear();

    TiledTree::LevelOrderTraversal levelOrder(m_tiles);
    auto& sortedTiles = levelOrder.LevelOrderNodes();
    
    // leafIndexMap maps the original tile ID to an index in the 
    // leaves array for any node that has a representation in the 
    // leaves array
    std::map<int32_t, int32_t> tileIndexMap, leafIndexMap;

    int32_t numberOfTilesInLeafArray=0, numberOfTilesInTilesArray=0, currentTileIndex=0;
    std::vector<double> tileThresholds(TileSize());
    std::vector<int32_t> tileFeatureIndices(TileSize());
    std::list<int32_t> leafTileIndices, leafArrayIndex;
    for (auto& tile : sortedTiles) {
        assert (numberOfTilesInTilesArray == (int32_t)childIndices.size());
        assert (childIndices.size() == leafIndices.size());
        assert (childIndices.size() == tileShapeIDs.size());
        assert (childIndices.size() == leafBitMasks.size());
        assert (thresholds.size() == featureIndices.size());
        assert (numberOfTilesInLeafArray == (int32_t)leaves.size());

        // if tile is a leaf and all siblings are leaves, put it into the leaf array
        // if not all siblings are leaves, put it in both arrays
        if (tile.IsLeafTile()) {
            
            // Put all leaves in to the leaves array
            int32_t leafArrayIndex = static_cast<int32_t>(leaves.size());
            leaves.push_back(tile.GetNode(tile.GetNodeIndices().front()).threshold);
            leafIndexMap[currentTileIndex] = leafArrayIndex;
            numberOfTilesInLeafArray += 1;

            if (!AreAllSiblingsLeaves(tile, sortedTiles)) {
                // The values that are pushed here are just place holders for leaf tiles. They
                // should never be accessed during execution.
                tile.GetThresholds(tileThresholds.begin());
                thresholds.insert(thresholds.end(), tileThresholds.begin(), tileThresholds.end());
                tile.GetFeatureIndices(tileFeatureIndices.begin());
                featureIndices.insert(featureIndices.end(), tileFeatureIndices.begin(), tileFeatureIndices.end());
                tileShapeIDs.push_back(tile.GetTileShapeID());
                childIndices.push_back(-1);
                leafIndices.push_back(-1);
                leafBitMasks.push_back(0);
                // This is not the index to which other tiles should point, they should
                // point into the leaves array
                // tileIndexMap[currentTileIndex] = currentTileIndex - numberOfTilesInLeafArray;
                tileIndexMap[currentTileIndex] = numberOfTilesInTilesArray; // HACK!
                ++numberOfTilesInTilesArray;
            }
        }
        else {
            // If any siblings of this tile are leaves, then it needs to be pushed
            // into both the leaves and tiles arrays, otherwise, just the tiles array
            
            tile.GetThresholds(tileThresholds.begin());
            thresholds.insert(thresholds.end(), tileThresholds.begin(), tileThresholds.end());
            tile.GetFeatureIndices(tileFeatureIndices.begin());
            featureIndices.insert(featureIndices.end(), tileFeatureIndices.begin(), tileFeatureIndices.end());
            tileShapeIDs.push_back(tile.GetTileShapeID());
            childIndices.push_back(tile.GetChildren().front());
            leafIndices.push_back(tile.GetChildren().front());
            // Get the bit mask that indicates which of the children are leaves
            leafBitMasks.push_back(GetChildrenBitMask(tile, sortedTiles));
            tileIndexMap[currentTileIndex] = numberOfTilesInTilesArray;
            numberOfTilesInTilesArray += 1;

            if (HasLeafSiblings(tile, sortedTiles)) {
                int32_t leafArrayIndex = static_cast<int32_t>(leaves.size());
                assert (leafArrayIndex == numberOfTilesInLeafArray);
                leaves.push_back(tile.GetNode(tile.GetNodeIndices().front()).threshold);
                leafIndexMap[currentTileIndex] = leafArrayIndex;
                numberOfTilesInLeafArray += 1;
            }
        }
        ++currentTileIndex;
    }

    assert (leafIndices.size() == childIndices.size());
    for (size_t i=0 ; i<childIndices.size() ; ++i) {
        if (childIndices.at(i) == -1)
            continue;
        auto mapIter = tileIndexMap.find(childIndices.at(i));
        if (mapIter != tileIndexMap.end()) {
            auto newChildIndex = mapIter->second;
            assert (newChildIndex >= 0);
            childIndices.at(i) = newChildIndex;
        }
        else {
            auto leavesMapIter = leafIndexMap.find(childIndices.at(i));
            assert (leavesMapIter != leafIndexMap.end());
            childIndices.at(i) = leavesMapIter->second + numberOfTilesInTilesArray;
        }
    }
    for (size_t i=0 ; i<leafIndices.size() ; ++i) {
        if (leafIndices.at(i) == -1)
            continue;
        auto mapIter = leafIndexMap.find(leafIndices.at(i));
        assert (mapIter != leafIndexMap.end());
        auto newLeafIndex = mapIter->second;
        assert (newLeafIndex >= 0);
        leafIndices.at(i) = newLeafIndex + numberOfTilesInTilesArray;
    }
}

void TiledTree::IncreaseTileDepth(int32_t leafIndex, int32_t leafDepth, int32_t maxDepth) {
    assert (m_tiles.at(leafIndex).IsLeafTile());
    int32_t leafNodeId = m_tiles.at(leafIndex).GetNodeIndices().front();
    auto treeLeafNode = m_modifiedTree.GetNodes().at(leafNodeId);
    ++m_numberOfDummyTiles;
    // Create a new tile with dummy nodes
    std::vector<int32_t> tileNodeIds(TileSize(), -1);
    int32_t newTileIndex = NewTiledTreeNode(-m_numberOfDummyTiles);
    for (int32_t i=0 ; i<TileSize() ; ++i) {
        tileNodeIds.at(i) = m_modifiedTree.NewNode(0.0, 0, -m_numberOfDummyTiles);
    }

    std::vector<int32_t> newLeafNodeIds(TileSize()+1, -1);
    newLeafNodeIds.at(0) = leafNodeId;
    for (int32_t i=1 ; i<TileSize() + 1; ++i) {
        ++m_numberOfDummyTiles;
        newLeafNodeIds.at(i) = m_modifiedTree.NewNode(treeLeafNode.threshold, treeLeafNode.featureIndex, -m_numberOfDummyTiles);
    }

    // TODO maybe duplicate the leaf as well here? We need to change the 
    // parent of the new leaf to point to the correct node in the tree.
    // Otherwise, setting the parent of the root of the new tile will not 
    // work correctly!
    int32_t treeLeafNodesVectorIndex = 0;
    for (int32_t i=0 ; i<TileSize() ; ++i) {
        // If we are at the root of the tile, then set its parent to be 
        // the parent of the leaf we are pushing down.
        // If the node's parent is something we are adding in the dummy tile
        // set it to that node
        if (i!=0)
            m_modifiedTree.SetNodeParent(tileNodeIds.at(i), tileNodeIds.at(i/2));
        else {
            auto& parentNode = m_modifiedTree.GetNodes().at(treeLeafNode.parent);
            m_modifiedTree.SetNodeParent(tileNodeIds.at(i), treeLeafNode.parent);
            if (parentNode.leftChild == leafNodeId)
                m_modifiedTree.SetNodeLeftChild(treeLeafNode.parent, tileNodeIds.at(i));
            else
                m_modifiedTree.SetNodeRightChild(treeLeafNode.parent, tileNodeIds.at(i));
        }

        if (2*i+1 < TileSize()) {
            m_modifiedTree.SetNodeLeftChild(tileNodeIds.at(i), tileNodeIds.at(2*i + 1));
        }
        else {
            auto newTreeLeafId = newLeafNodeIds.at(treeLeafNodesVectorIndex);
            ++treeLeafNodesVectorIndex;
            m_modifiedTree.SetNodeLeftChild(tileNodeIds.at(i), newTreeLeafId);
            m_modifiedTree.SetNodeParent(newTreeLeafId, tileNodeIds.at(i));
        }
        if (2*i+2 < TileSize()) {
            m_modifiedTree.SetNodeRightChild(tileNodeIds.at(i), tileNodeIds.at(2*i + 2));
        }
        else {
            auto newTreeLeafId = newLeafNodeIds.at(treeLeafNodesVectorIndex);
            ++treeLeafNodesVectorIndex;
            m_modifiedTree.SetNodeRightChild(tileNodeIds.at(i), newTreeLeafId);
            m_modifiedTree.SetNodeParent(newTreeLeafId, tileNodeIds.at(i));
        }
    }
    // TODO the leaf tile needs to be replicated. It cannot be reused
    // TODO the parent node of the leaf is going to be invalid because several 
    // newly added tiles are going to point to the same leaf tile as a child
    auto& newTile = m_tiles.at(newTileIndex);
    for (auto dummyTreeNodeId : tileNodeIds)
        AddNodeToTile(newTileIndex, dummyTreeNodeId);
    newTile.m_parent = m_tiles.at(leafIndex).m_parent;
    m_tiles.at(leafIndex).m_parent = newTileIndex;
    newTile.m_children = std::vector<int32_t>(1, leafIndex);
    for (int32_t i=1; i<TileSize()+1 ; ++i) {
        auto newTreeLeafNodeTileId = m_modifiedTree.TilingDescriptor().TileIDs().at(newLeafNodeIds.at(i));
        auto newLeafTile = NewTiledTreeNode(newTreeLeafNodeTileId);
        AddNodeToTile(newLeafTile, newLeafNodeIds.at(i));
        m_tiles.at(newLeafTile).m_parent = newTileIndex;
        assert (m_tiles.at(newLeafTile).IsLeafTile());
        newTile.m_children.push_back(newLeafTile);
    }

    if (leafDepth + 1 == maxDepth)
        return;

    // Making a copy here because the m_tiles vector can be modified by the recursive calls!
    auto children = m_tiles.at(leafIndex).m_children;
    for (auto childLeaf : children)
        IncreaseTileDepth(childLeaf, leafDepth+1, maxDepth);
}

void TiledTree::MakeAllLeavesSameDepth() {
    auto depth = this->GetTreeDepth();
    std::list<std::tuple<int32_t, int32_t>> leavesToPad;
    for (int32_t i=0 ; i<(int32_t)m_tiles.size() ; ++i) {
        auto& tile = m_tiles.at(i);
        if (!tile.IsLeafTile())
            continue;
        auto tilePtr = &tile;
        int32_t leafDepth = 1;
        while (tilePtr->GetParent() != decisionforest::DecisionTree<>::INVALID_NODE_INDEX) {
            tilePtr = &(m_tiles.at(tilePtr->GetParent()));
            ++leafDepth;
        }
        if (leafDepth != depth) {
            leavesToPad.push_back(std::make_tuple(i, leafDepth));
        }
    }
    if (TreeBeard::Logging::loggingOptions.logTreeStats) {
        TreeBeard::Logging::Log("Number of leaves that were padded : " + std::to_string(leavesToPad.size()));
    }
    for (auto leafEntry : leavesToPad) {
        IncreaseTileDepth(std::get<0>(leafEntry), std::get<1>(leafEntry), depth);
    }
}

// -----------------------------------------------
// Methods for class TileShapeToTileIDMap
// -----------------------------------------------

void TileShapeToTileIDMap::InitMap() {
    TileStringGenerator(m_tileSize);
}

struct TileGenState {
    int32_t numberOfNodes = -1;
    // int32_t rootIndex = -1;
    bool updateLeftTreeSize = true;
    // int32_t leftTreeSize = -1;
    bool updateLeftTree = true;
    bool lastLeftSubtreeOfCurrentSize = false;
    bool lastLeftSubtree = false;
};

enum class SubtreeUpdateState { kContinue, kDone };

void CopySubtree(std::string& newStr, std::string& oldStr, int32_t root) {
    if (root >= static_cast<int32_t>(newStr.size()))
        return;
    assert (newStr.size() == oldStr.size());
    newStr.at(root) = oldStr[root];
    CopySubtree(newStr, oldStr, 2*root+1);
    CopySubtree(newStr, oldStr, 2*root+2);
}

void ResetStates(std::vector<TileGenState>& states, int32_t root) {
    if (root >= static_cast<int32_t>(states.size()))
        return;
    states.at(root) = TileGenState();
    ResetStates(states, 2*root+1);
    ResetStates(states, 2*root+2);
}

SubtreeUpdateState UpdateTileString(std::string& str, std::string& oldStr, std::vector<TileGenState>& states, int32_t nodeIndex) {
    TileGenState& state = states.at(nodeIndex);
    assert (state.numberOfNodes > -1 && "Uninitialized state!");
    if (state.numberOfNodes == 0)
        return SubtreeUpdateState::kDone;
    
    str.at(nodeIndex) = '1';
    if (state.numberOfNodes == 1)
        return SubtreeUpdateState::kDone;
    
    int32_t leftChild = 2*nodeIndex+1;
    int32_t rightChild = 2*nodeIndex+2;
    auto& leftSubtreeState = states[leftChild];
    auto& rightSubtreeState = states[rightChild];
    // TODO Return if these are out of bounds (Actually can they be?)
    if (state.updateLeftTreeSize) {
        assert (state.updateLeftTree);
        auto leftTreeSize = leftSubtreeState.numberOfNodes;
        ResetStates(states, leftChild);
        leftSubtreeState.numberOfNodes = leftTreeSize + 1;
        rightSubtreeState.numberOfNodes = state.numberOfNodes - 1 - leftSubtreeState.numberOfNodes;
        state.updateLeftTreeSize = false;
    }
    if (state.updateLeftTree) {
        auto leftSubtreeUpdateState = UpdateTileString(str, oldStr, states, leftChild);
        // We've reached the last left subtree if all the nodes are in the left subtree and we've generate all possible trees of that size
        state.lastLeftSubtree = leftSubtreeUpdateState == SubtreeUpdateState::kDone && leftSubtreeState.numberOfNodes == state.numberOfNodes-1;
        state.lastLeftSubtreeOfCurrentSize = leftSubtreeUpdateState == SubtreeUpdateState::kDone;
        state.updateLeftTree = false;
        // Reset the states of the right tree
        ResetStates(states, rightChild);
        rightSubtreeState.numberOfNodes =  state.numberOfNodes - 1 - leftSubtreeState.numberOfNodes;
    }
    else {
        // TODO How do we restore the old tree? Maybe copy the required parts from the old string?
        CopySubtree(str, oldStr, leftChild);
    }
    auto rightSubtreeUpdateState = UpdateTileString(str, oldStr, states, rightChild);
    if (state.lastLeftSubtree && rightSubtreeUpdateState==SubtreeUpdateState::kDone)
        return SubtreeUpdateState::kDone;
    else if (state.lastLeftSubtreeOfCurrentSize && rightSubtreeUpdateState==SubtreeUpdateState::kDone) {
        state.updateLeftTreeSize = true;
        state.updateLeftTree = true;
    }
    else if (rightSubtreeUpdateState==SubtreeUpdateState::kDone){
        state.updateLeftTree = true;
    }
    return SubtreeUpdateState::kContinue;
}

void TileShapeToTileIDMap::TileStringGenerator(int32_t numNodes) {
    int32_t maxTreeSize = std::pow(2, numNodes) - 1;
    std::string oldStr = std::string(maxTreeSize, '0');;
    std::vector<TileGenState> states(maxTreeSize);
    states.at(0).numberOfNodes = numNodes;
    SubtreeUpdateState updateState;
    do {
        std::string str = std::string(maxTreeSize, '0');
        updateState = UpdateTileString(str, oldStr, states, 0);
        oldStr = str;
        // std::cout << str << std::endl;
        assert(m_tileStringToTileIDMap.find(str) == m_tileStringToTileIDMap.end());
        m_tileStringToTileIDMap[str] = m_currentTileID;
        ++m_currentTileID;
    } while (updateState == SubtreeUpdateState::kContinue);
    
    // Add an entry for tiles with leaves
    std::string str = std::string(maxTreeSize, '0');
    str[0] = '1';
    m_tileStringToTileIDMap[str] = m_currentTileID;
    ++m_currentTileID;

    // std::cout << m_tileStringToTileIDMap.size() << std::endl;
    // The map should have all full tiles plus 1 for the tile shape with a single leaf
    assert (static_cast<int32_t>(m_tileStringToTileIDMap.size()) == (TileShapeToTileIDMap::NumberOfTileShapes(m_tileSize)+1));
}

std::map<int32_t, int32_t> TileShapeToTileIDMap::tileSizeToNumberOfShapesMap;

int32_t TileShapeToTileIDMap::NumberOfTileShapes(int32_t tileSize) {
    assert(tileSize >= 0);
    if (tileSize==0 || tileSize == 1) return 1;
    if (tileSize == 2) return 2;
    
    auto iter = tileSizeToNumberOfShapesMap.find(tileSize);
    if (iter != tileSizeToNumberOfShapesMap.end())
        return iter->second;
    
    int32_t numShapes=0;
    for (int32_t leftSubTreeSize=0 ; leftSubTreeSize<tileSize ; ++leftSubTreeSize) {
        numShapes += NumberOfTileShapes(leftSubTreeSize)*NumberOfTileShapes(tileSize-1-leftSubTreeSize);
    }
    tileSizeToNumberOfShapesMap[tileSize] = numShapes;
    return numShapes;
}

int32_t ConstructTreeForTile(const std::string& tileStr, int32_t root, DecisionTree<>& tree) {
    if (root >= static_cast<int32_t>(tileStr.size()))
        return -1;
    if (tileStr.at(root) == '0')
        return -1;
    auto node = tree.NewNode(-1, -(root+1)); // Adding the node index as the feature index for debugging
    auto leftChild = ConstructTreeForTile(tileStr, 2*root+1, tree);
    auto rightChild = ConstructTreeForTile(tileStr, 2*root+2, tree);

    if (rightChild != -1) {
        tree.SetNodeParent(rightChild, node);
        tree.SetNodeRightChild(node, rightChild);
    }
    if (leftChild != -1) {
        tree.SetNodeParent(leftChild, node);
        tree.SetNodeLeftChild(node, leftChild);
    }
    return node;
}

void AddTileChildren(DecisionTree<>& tree, int32_t nodeNumber, int32_t& childNumber) {
    if (nodeNumber == -1)
        return;
    auto& node = tree.GetNodes().at(nodeNumber);
    if (node.leftChild != -1)
        AddTileChildren(tree, node.leftChild, childNumber);
    else {
        auto childNode = tree.NewNode(-1, childNumber);
        tree.SetNodeParent(childNode, nodeNumber);
        tree.SetNodeLeftChild(nodeNumber, childNode);
        ++childNumber;
    }
    if (node.rightChild != -1)
        AddTileChildren(tree, node.rightChild, childNumber);
    else {
        auto childNode = tree.NewNode(-1, childNumber);
        tree.SetNodeParent(childNode, nodeNumber);
        tree.SetNodeRightChild(nodeNumber, childNode);
        ++childNumber;
    }
}

void ComputeLUTHelper(DecisionTree<>& tree, size_t root, std::vector<int32_t>& outcomes, std::map<int32_t, std::vector<int32_t>>& childIndexToOutcomesMap) {
    assert(root < tree.GetNodes().size());
    auto& node = tree.GetNodes().at(root);
    if (node.featureIndex >= 0) {
        // We've reached a node that corresponds to the a child of the tile
        childIndexToOutcomesMap[node.featureIndex] = outcomes;
        return;
    }
    auto oldNodeOutcome = outcomes.at(root);
    assert (oldNodeOutcome == -1 && "We shouldn't depend on this nodes outcome unless we've gone through it!");
    outcomes.at(root) = 1;
    ComputeLUTHelper(tree, node.leftChild, outcomes, childIndexToOutcomesMap);
    outcomes.at(root) = 0;
    ComputeLUTHelper(tree, node.rightChild, outcomes, childIndexToOutcomesMap);
    outcomes.at(root) = oldNodeOutcome;
}

inline int32_t InvertBits(int32_t val, int32_t numBits) {
    int32_t ret = 0, i, temp;
    for (i = 0; i < numBits; i++)
    {
        temp = (val & (1 << i));
        if(temp)
            ret |= (1 << ((numBits - 1) - i));
    }
    return ret;
}

void SetLUTEntries(std::vector<int32_t>& lut, int32_t childIndex, const std::vector<int32_t>& outcomes,
                   size_t currentIndex, int32_t outcomeBits, int32_t tileSize) {
    if (currentIndex >= outcomes.size()) {
        if (decisionforest::UseBitcastForComparisonOutcome)
            outcomeBits = InvertBits(outcomeBits, tileSize);
        lut.at(outcomeBits) = childIndex;
        return;
    }
    auto currentOutcome = outcomes.at(currentIndex);
    if (currentOutcome == -1 || currentOutcome == 0) {
        // This node's outcome is a don't care.. So it could be either 0 or 1.
        auto newOutcomeBits = outcomeBits << 1;
        SetLUTEntries(lut, childIndex, outcomes, currentIndex+1, newOutcomeBits, tileSize);
    }
    if (currentOutcome == -1 || currentOutcome == 1) {
        auto newOutcomeBits = outcomeBits << 1;
        newOutcomeBits |= 1;
        SetLUTEntries(lut, childIndex, outcomes, currentIndex+1, newOutcomeBits, tileSize); 
    }
}

std::vector<int32_t> ComputeLookUpTableForSingleShape(DecisionTree<>& tree, int32_t tileSize) {
    std::vector<int32_t> outcomes(tileSize, -1);
    std::map<int32_t, std::vector<int32_t>> childIndexToOutcomesMap;
    ComputeLUTHelper(tree, 0, outcomes, childIndexToOutcomesMap);

    int32_t numOutcomes = std::pow(2, tileSize);
    std::vector<int32_t> lut(numOutcomes, -1);
    for (auto indexOutcomePair : childIndexToOutcomesMap) {
        SetLUTEntries(lut, indexOutcomePair.first, indexOutcomePair.second, 0, 0, tileSize);
    }
    return lut;
}

// Assume that the children of the tile are stored left to right
std::vector<std::vector<int32_t>> TileShapeToTileIDMap::ComputeTileLookUpTable() {
    std::vector<std::vector<int32_t>> tileLUT(m_currentTileID-1);
    for (auto mapPair : m_tileStringToTileIDMap) {
        // Skip adding a row in the LUT for the leaf shape. We added this shape 
        // at the end to the map and so its ID is always currentTileID - 1
        if (mapPair.second == m_currentTileID-1)
            continue;

        // First construct a tree that represents the tile
        DecisionTree<> tree;
        ConstructTreeForTile(mapPair.first, 0, tree);
        
        // TODO make this a method on DecisionTree (can't do it now because level order traversal is not templatized)
        LevelOrderTraversal levelOrderTraversal(tree.GetNodes());
        tree.SetNodes(levelOrderTraversal.LevelOrderNodes());
        
        int32_t childNumber = 0;
        AddTileChildren(tree, 0, childNumber);
        // tree.WriteToDOTFile("/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/treeForTile.dot");

        tileLUT.at(mapPair.second) = ComputeLookUpTableForSingleShape(tree, m_tileSize);
    }
    return tileLUT;
}

int32_t TileShapeToTileIDMap::GetTileID(TiledTreeNode& tile) {
    auto shapeString = tile.GetTileShapeString();
    auto mapIter = m_tileStringToTileIDMap.find(shapeString);
    assert (mapIter != std::end(m_tileStringToTileIDMap));
    return mapIter->second;
}

} // decisionforest
} // mlir
