#include <queue>
#include <iostream>
#include "TreeTilingDescriptor.h"
#include "TreeTilingUtils.h"
#include "TiledTree.h"
#include "llvm/ADT/STLExtras.h"

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

void ForestJSONReader::ParseJSONFile() {
    assert (false && "Function not implemented!");
}

template<typename T>
void AppendAtEndOfList(std::list<T>& l, std::list<T>& newElements) {
    l.insert(std::end(l), std::begin(newElements), std::end(newElements));
}

void ForestJSONReader::AddSingleTileSizeEntry(std::list<int32_t>& treeIndices, std::list<int32_t>& numTilesList, std::list<std::vector<ThresholdType>>& serializedThresholds, 
                                              std::list<std::vector<FeatureIndexType>>& serializedFetureIndices, std::list<std::vector<int32_t>>& serializedTileShapeIDs,
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
                                   serializedThresholds, serializedFetureIndices, serializedTileShapeIDs};
        m_tileSizeEntries.push_back(entry);
    }
    else {
        AppendAtEndOfList(listIter->treeIndices, treeIndices);
        AppendAtEndOfList(listIter->numberOfTiles, numTilesList);
        AppendAtEndOfList(listIter->serializedThresholds, serializedThresholds);
        AppendAtEndOfList(listIter->serializedFetureIndices, serializedFetureIndices);
        AppendAtEndOfList(listIter->serializedTileShapeIDs, serializedTileShapeIDs);
    }
}

void ForestJSONReader::AddSingleTree(int32_t treeIndex, int32_t numTiles, std::vector<ThresholdType>& serializedThresholds,
                                     std::vector<FeatureIndexType>& serializedFetureIndices, std::vector<int32_t>& tileShapeIDs,
                                     const int32_t tileSize, const int32_t thresholdBitWidth, const int32_t indexBitWidth) {
    std::list<int32_t> treeIndices = { treeIndex };
    std::list<int32_t> numTilesList = { numTiles };
    std::list<std::vector<ThresholdType>> serializedThresholdsList = { serializedThresholds };
    std::list<std::vector<FeatureIndexType>> serializedFetureIndicesList = { serializedFetureIndices };
    std::list<std::vector<int32_t>> serializedTileShapeIDs = { tileShapeIDs };

    AddSingleTileSizeEntry(treeIndices, numTilesList, serializedThresholdsList, serializedFetureIndicesList, serializedTileShapeIDs, 
                           tileSize, thresholdBitWidth, indexBitWidth);
}

void ForestJSONReader::ClearAllData() {
    m_tileSizeEntries.clear();
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

int32_t ForestJSONReader::GetTotalNumberOfTiles() {
    assert (this->m_tileSizeEntries.size() == 1 && "Only a single (tile size, threshold type, feature index type) configuration is supported");
    auto& tileSizeEntry = m_tileSizeEntries.front();
    int32_t numTiles = 0;
    for (auto singleTreeNumTiles : tileSizeEntry.numberOfTiles)
        numTiles += singleTreeNumTiles;
    return numTiles;
}

// Ultimately, this will write a JSON file. For now, we're just 
// storing it in memory assuming the compiler and inference 
// will run in the same process. 
void PersistDecisionForest(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) {
    mlir::decisionforest::ForestJSONReader::GetInstance().ClearAllData();

    auto numTrees = forest.NumTrees();
    mlir::decisionforest::ForestJSONReader::GetInstance().SetNumberOfTrees(numTrees);
    for (size_t i=0; i<numTrees ; ++i) {
        auto treeType = forestType.getTreeType(0).cast<decisionforest::TreeType>();
        
        // TODO We're assuming that the threshold type is a float type and index type 
        // is an integer. This is just to get the size. Can we get the size differently?
        auto thresholdType = treeType.getThresholdType().cast<FloatType>();
        auto featureIndexType = treeType.getFeatureIndexType().cast<IntegerType>(); 

        auto& tree = forest.GetTree(static_cast<int64_t>(i));
        if (tree.TilingDescriptor().MaxTileSize() == 1) {
            std::vector<ThresholdType> thresholds = tree.GetThresholdArray();
            std::vector<FeatureIndexType> featureIndices = tree.GetFeatureIndexArray();
            std::vector<int32_t> tileShapeIDs = { };
            int32_t numTiles = tree.GetNumberOfTiles();
            int32_t tileSize = tree.TilingDescriptor().MaxTileSize();
            mlir::decisionforest::ForestJSONReader::GetInstance().AddSingleTree(i, numTiles, thresholds, featureIndices, tileShapeIDs, tileSize, 
                                                                                thresholdType.getWidth(), featureIndexType.getWidth());
        }
        else {
            TiledTree tiledTree(tree);
            // tiledTree.WriteDOTFile("/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/TiledTree.dot");
            std::vector<ThresholdType> thresholds = tiledTree.SerializeThresholds();
            std::vector<FeatureIndexType> featureIndices = tiledTree.SerializeFeatureIndices();
            std::vector<int32_t> tileShapeIDs = tiledTree.SerializeTileShapeIDs();
            int32_t numTiles = tiledTree.GetNumberOfTiles();
            int32_t tileSize = tree.TilingDescriptor().MaxTileSize();
            mlir::decisionforest::ForestJSONReader::GetInstance().AddSingleTree(i, numTiles, thresholds, featureIndices, tileShapeIDs, tileSize, 
                                                                                thresholdType.getWidth(), featureIndexType.getWidth());

        }
    }
}

void ClearPersistedForest() {
    mlir::decisionforest::ForestJSONReader::GetInstance().ClearAllData();
}

int32_t GetTotalNumberOfTiles() {
    return mlir::decisionforest::ForestJSONReader::GetInstance().GetTotalNumberOfTiles();
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

const DecisionTree<>::Node& TiledTreeNode::GetNode(int32_t index) { 
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
          if (leftChild.IsLeaf() && rightChild.IsLeaf())
              candidateNodes.push_front(nodeIndex);
      }
      assert (candidateNodes.size() > 0);
      // TODO How do we determine the shape of this tile once we add new nodes? Maybe some kind of look up based on the positions of the nodes in the 
      // full dense serialization?
      // TODO How do we decide which of the candidate nodes to use as the parent of the new node(s)? For now, picking from the first candidate, which will 
      // be the right most node on the bottom most level. 
      auto candidateIter = candidateNodes.begin();
      for (int32_t i=0 ; i<numberOfNodesToAdd && candidateIter!=candidateNodes.end(); i+=2) { // We can add two dummy nodes for every candidate node
          // TODO How do we decide where to add the new nodes? Maybe just add them somewhere and call sort again?
          assert (candidateIter != candidateNodes.end());
          auto candidateIndex = *candidateIter;
          auto& candidateNode = GetNode(candidateIndex);
          {
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
          }
          if (i+1 == numberOfNodesToAdd)
              break;
          {
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
 : m_owningTree(owningTree), m_modifiedTree(owningTree), m_tileShapeToTileIDMap(m_owningTree.TilingDescriptor().MaxTileSize())
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
    // Expand the tiles that aren't full with dummy nodes
    for (auto& tile : m_tiles)
        tile.AddExtraNodesIfNeeded();
    
    // Set the shape ID of all the tiles in the tree
    for (auto& tile : m_tiles)
        tile.m_tileShapeID = m_tileShapeToTileIDMap.GetTileID(tile);
    
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
        << ", FeatIdx:" << node.featureIndex << "\", style=bold, color=" << color << "];\n";
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

void SetLUTEntries(std::vector<int32_t>& lut, int32_t childIndex, const std::vector<int32_t>& outcomes, size_t currentIndex, int32_t outcomeBits) {
    if (currentIndex >= outcomes.size()) {
        lut.at(outcomeBits) = childIndex;
        return;
    }
    auto currentOutcome = outcomes.at(currentIndex);
    if (currentOutcome == -1 || currentOutcome == 0) {
        // This node's outcome is a don't care.. So it could be either 0 or 1.
        auto newOutcomeBits = outcomeBits << 1;
        SetLUTEntries(lut, childIndex, outcomes, currentIndex+1, newOutcomeBits);
    }
    if (currentOutcome == -1 || currentOutcome == 1) {
        auto newOutcomeBits = outcomeBits << 1;
        newOutcomeBits |= 1;
        SetLUTEntries(lut, childIndex, outcomes, currentIndex+1, newOutcomeBits); 
    }
}

std::vector<int32_t> ComputeLookUpTableForSingleShape(DecisionTree<>& tree, int32_t tileSize) {
    std::vector<int32_t> outcomes(tileSize, -1);
    std::map<int32_t, std::vector<int32_t>> childIndexToOutcomesMap;
    ComputeLUTHelper(tree, 0, outcomes, childIndexToOutcomesMap);

    int32_t numOutcomes = std::pow(2, tileSize);
    std::vector<int32_t> lut(numOutcomes, -1);
    for (auto indexOutcomePair : childIndexToOutcomesMap) {
        SetLUTEntries(lut, indexOutcomePair.first, indexOutcomePair.second, 0, 0);
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
