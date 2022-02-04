#ifndef _TILEDTREE_H_
#define _TILEDTREE_H_

#include <map>
#include <string>
#include <cassert>
#include "DecisionForest.h"

namespace mlir
{
namespace decisionforest
{

class TiledTree;

class TiledTreeNode {
private:
    friend class TiledTree;
    DecisionTree<>& m_owningTree;
    TiledTree& m_tiledTree;
    // indices into the contained tree nodes in the owning tree.
    // indices have to be in level order (wrt to the owning tree)
    std::vector<int32_t> m_nodeIndices;
    // This is the integer value stored in the tiling descriptor for nodes in this tile
    int32_t m_tileID;
    // This is the ID that uniquely identifies the shape of this tile
    int32_t m_tileShapeID;
    int32_t m_tileIndex;
    // Index of the parent TiledTreeNode in the tiled tree
    int32_t m_parent;
    // TODO What order do the children need to be in?
    // Indices of the children of this TiledTreeNode in the tiled tree
    std::vector<int32_t> m_children;
    
    // Whether this tile contains any extra nodes that have been added to the modified tree 
    // in the tiled tree that contains it -- basically does the tile contain any nodes
    // not in the owning tree (the original decision tree)
    bool m_hasExtraNodes;

    // Sort the nodes in this tile in level order and left to right on a given level
    void SortTileNodes();

    void SetParent(int32_t parentIndex) { m_parent = parentIndex; }
    void AddChild(int32_t childIndex) { m_children.push_back(childIndex); }

    int32_t FindTileEntryNode();
    bool AreNodesInSameTile(int32_t node1, int32_t node2);

    void AddNodeToTile(int32_t nodeIndex) { m_nodeIndices.push_back(nodeIndex); }

    void AddExtraNodesIfNeeded();
    void WriteDOTSubGraph(std::ofstream& fout);
    
    void GetThresholds(std::vector<double>::iterator beginIter);
    void GetFeatureIndices(std::vector<int32_t>::iterator beginIter);
    void ComputeTileShapeString(std::string& str, int32_t tileNodeIndex, int32_t stringIndex);
public:
    TiledTreeNode(DecisionTree<>& owningTree, TiledTree& tiledTree, int32_t tileID, int32_t tileIndex)
        :m_owningTree(owningTree), m_tiledTree(tiledTree), m_tileID(tileID), m_tileShapeID(-1), m_tileIndex(tileIndex), m_hasExtraNodes(false)
    { }
    std::vector<int32_t>& GetNodeIndices() { return m_nodeIndices; }

    int32_t GetParent() const { return m_parent; }
    const std::vector<int32_t>& GetChildren() const { return m_children; }

    int32_t GetEntryNode() const { return m_nodeIndices[0]; }
    DecisionTree<>& GetTree();
    const DecisionTree<>::Node& GetNode(int32_t index) const;
    std::string GetTileShapeString();
    int32_t GetTileShapeID() const { return m_tileShapeID; }
    bool IsLeafTile() const { return m_nodeIndices.size()==1 && GetNode(m_nodeIndices.at(0)).IsLeaf(); }
};

class TiledTreeNode;

// Routines to compute combinatorial properties of tiles
class TileShapeToTileIDMap
{
    static std::map<int32_t, int32_t> tileSizeToNumberOfShapesMap;

    int32_t m_tileSize;
    std::map<std::string, int32_t> m_tileStringToTileIDMap;
    int32_t m_currentTileID = 0;
    void TileStringGenerator(int32_t numNodes);
    void InitMap();
public:
    TileShapeToTileIDMap(int32_t tileSize) 
     : m_tileSize(tileSize)
    {
        InitMap();
    }
    int32_t GetTileID(TiledTreeNode& tile);
    // Index is (tileShapeID, comparison result)
    std::vector<std::vector<int32_t>> ComputeTileLookUpTable();
    // Number of tile shapes with the given tile size
    static int32_t NumberOfTileShapes(int32_t tileSize);
};

struct TiledTreeStats {
    int32_t tileSize;
    int32_t originalTreeDepth;
    int32_t tiledTreeDepth;
    // The number of tiles if we store only one copy of duplicated leaves
    int32_t numberOfUniqueTiles;
    // The number of tiles assuming even duplicated leaves are distinct
    int32_t numberOfTiles;
    int32_t originalTreeNumNodes;
    // This is the number of thresholds stored in the tiled tree including
    // replicating each leaf
    int32_t tiledTreeNumNodes;
    // The number of nodes that are added to the original tree
    int32_t numAddedNodes;
    int32_t originalTreeNumberOfLeaves;
    int32_t tiledTreeNumberOfLeafTiles;
    // Num leaves that have only leaves for siblings
    int32_t numLeavesWithAllLeafSiblings;
    int32_t numberOfFeatures;
    std::vector<int32_t> leafDepths;
};

class TiledTree {
    friend class TiledTreeNode;

    std::vector<TiledTreeNode> m_tiles;
    DecisionTree<>& m_owningTree;
    // We may need to add nodes to the original tree to make the tiles full sized. This 
    // tree is the modified tree with nodes added if required.
    DecisionTree<> m_modifiedTree;
    std::map<int32_t, int32_t> m_nodeToTileMap;
    TileShapeToTileIDMap m_tileShapeToTileIDMap;

    void ConstructTiledTree();
    void SetChildrenForTile(TiledTreeNode& tile);
    void SetChildrenHelper(TiledTreeNode& tile, int32_t nodeIndex, std::vector<int32_t>& children);
    void SetParent(int32_t index, int32_t parent) { m_tiles[index].SetParent(parent); }
    void AddChild(int32_t index, int32_t child) { m_tiles[index].AddChild(child); }
    void AddNodeToTile(int32_t index, int32_t nodeIndex) {
        assert(m_nodeToTileMap.find(nodeIndex) == m_nodeToTileMap.end());
        m_nodeToTileMap[nodeIndex] = index;
        m_tiles[index].AddNodeToTile(nodeIndex);
    }
    bool Validate();
    int32_t NewTiledTreeNode(int32_t tileID) {
        int32_t tileIndex = m_tiles.size();
        m_tiles.push_back(TiledTreeNode(m_owningTree, *this, tileID, tileIndex));
        return tileIndex;
    }
    void EmitNodeDOT(std::ofstream& fout, int32_t nodeIndex);
    
    template <typename AttribType, typename GetterType>
    void GetTileAttributeArray(std::vector<AttribType>& attributeVec,
                               size_t vecIndex, size_t nodeIndex, GetterType get, bool singleAttribute);
    
    int32_t GetTreeDepthHelper(int32_t tileIndex);
    int32_t NumberOfLeafTilesHelper(int32_t tileIndex);
    int32_t NumberOfLeavesWithAllLeafSiblings(int32_t tileIndex);
    // The number of tiles required to store this tree if duplicated nodes are considered unique
    int32_t NumberOfTiles();
    bool AreAllSiblingsLeaves(TiledTreeNode& tile, const std::vector<TiledTreeNode>& tiles);
    std::vector<int32_t> GetLeafDepths();
public:
    TiledTree(DecisionTree<>& owningTree);
    
    TiledTreeNode& GetTile(int32_t index) {
        return m_tiles[index];
    }
    int32_t GetNumberOfTiles() {
        int32_t depth = GetTreeDepth();
        int32_t tileSize = m_owningTree.TilingDescriptor().MaxTileSize();
        int32_t numChildren = tileSize + 1;
        size_t numTiles = static_cast<size_t>((std::pow(numChildren, depth) - 1)/(numChildren-1));
        return numTiles;
    }
    int32_t TileSize() { return m_owningTree.TilingDescriptor().MaxTileSize(); }
    size_t NumTiles() { return m_tiles.size(); }
    int32_t GetNodeTileIndex(int32_t nodeIndex) {
        if (nodeIndex == DecisionTree<>::INVALID_NODE_INDEX)
          return nodeIndex;
        assert(m_nodeToTileMap.find(nodeIndex) != m_nodeToTileMap.end());
        return m_nodeToTileMap[nodeIndex];
    }
    void WriteDOTFile(const std::string& filename);
    std::vector<double> SerializeThresholds();
    std::vector<int32_t> SerializeFeatureIndices();
    std::vector<int32_t> SerializeTileShapeIDs();

    void GetSparseSerialization(std::vector<double>& thresholds, std::vector<int32_t>& featureIndices, 
                                std::vector<int32_t>& tileShapeIDs, std::vector<int32_t>& childIndices, std::vector<double>& leaves);
    int32_t GetTreeDepth() { 
        // The root of the tiled tree should be the first node
        assert (m_tiles[0].GetParent() == DecisionTree<>::INVALID_NODE_INDEX);
        return GetTreeDepthHelper(0);
    }
    int32_t NumberOfLeafTiles();
    TiledTreeStats GetTreeStats();
    int32_t GetGroupId() { return m_owningTree.GetGroupId(); } 
    using LevelOrderSorterNodeType = TiledTreeNode;

    class LevelOrderTraversal {
      // using QueueEntry = std::pair<int32_t, LevelOrderSorterNodeType>;
      struct QueueEntry {
        int32_t parentNodeIndex;
        int32_t childNodeIndex;
        int32_t childNumber;
        LevelOrderSorterNodeType childNode;
      };
      std::vector<LevelOrderSorterNodeType> m_levelOrder;
      std::queue<QueueEntry> m_queue;
      struct MapKey {
        int32_t parentNodeIndex;
        int32_t childNodeIndex;
        int32_t childNum;
        bool operator<(const MapKey& rhs) const {
          int32_t vals1[] = { parentNodeIndex, childNodeIndex, childNum };
          int32_t vals2[] = { rhs.parentNodeIndex, rhs.childNodeIndex, rhs.childNum };
          for (int32_t i=0 ; i<static_cast<int32_t>(sizeof(vals1)/sizeof(int32_t)) ; ++i) {
            if (vals1[i] < vals2[i])
              return true;
            else if (vals1[i] > vals2[i])
              return false;  
          }
          return false;
        }
      };
      std::map<MapKey, int32_t> m_nodeIndexMap;
      std::map<int32_t, int32_t> m_levelOrderIndexToOriginalIndexMap;
      void DoLevelOrderTraversal(const std::vector<LevelOrderSorterNodeType>& nodes) {
        int32_t invalidIndex = DecisionTree<>::INVALID_NODE_INDEX;
        // MapKey invalidIndicesMapKey{invalidIndex, invalidIndex, 0 };
        // m_nodeIndexMap[invalidIndicesMapKey] = invalidIndex;
        // Assume the root is the first node.
        assert (nodes[0].GetParent() == invalidIndex);
        m_queue.push(QueueEntry{invalidIndex, 0, 0, nodes[0]});
        while(!m_queue.empty()) {
          auto entry = m_queue.front();
          m_queue.pop();
          auto index = entry.childNodeIndex;
          auto& node = entry.childNode;
          m_levelOrder.push_back(node);
          MapKey mapKey{entry.parentNodeIndex, index, entry.childNumber};
          assert (m_nodeIndexMap.find(mapKey) == m_nodeIndexMap.end());
          m_nodeIndexMap[mapKey] = m_levelOrder.size() - 1;
          m_levelOrderIndexToOriginalIndexMap[m_levelOrder.size() - 1] = index;
          if (node.IsLeafTile())
            continue;
          int32_t childNum=0;
          for (auto child : node.GetChildren()) {
            if (child != DecisionTree<>::INVALID_NODE_INDEX)
                m_queue.push(QueueEntry{index, child, childNum, nodes.at(child)});
            ++childNum;
          }
        }
      }

      int32_t GetNewIndex(MapKey& oldIndex) {
        auto iter = m_nodeIndexMap.find(oldIndex);
        assert (iter != m_nodeIndexMap.end());
        return iter->second;
      }

      void RewriteIndices() {
        int32_t currNodeIndex=0;
        for (auto& node : m_levelOrder) {
          // MapKey parentMapKey{node.GetParent(), }
          // node.m_parent = GetNewIndex(node.GetParent());
          assert (m_levelOrderIndexToOriginalIndexMap.find(currNodeIndex) != m_levelOrderIndexToOriginalIndexMap.end());
          int32_t originalIndex = m_levelOrderIndexToOriginalIndexMap[currNodeIndex];
          assert (originalIndex == node.m_tileIndex);
          for (size_t i=0 ; i<node.m_children.size() ; ++i) {
            assert(node.m_children.at(i) > 0);
            MapKey childMapKey{originalIndex, node.m_children.at(i), static_cast<int32_t>(i)};
            node.m_children.at(i) = GetNewIndex(childMapKey);
            assert(m_levelOrder.at(node.m_children.at(i)).GetParent() == originalIndex);
            m_levelOrder.at(node.m_children.at(i)).SetParent(currNodeIndex);
            assert (i == 0 || ( node.m_children.at(i) = (node.m_children.at(i-1)+1) ));
          }
          ++currNodeIndex;
        }
      }
    public:
      LevelOrderTraversal(const std::vector<LevelOrderSorterNodeType>& nodes) {
        DoLevelOrderTraversal(nodes);
        RewriteIndices();
      }
      std::vector<LevelOrderSorterNodeType>& LevelOrderNodes() { return m_levelOrder; }
    };

};


} // decisionforest
} // mlir
#endif // _TILEDTREE_H_