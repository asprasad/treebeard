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
    int32_t m_tileID;
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
public:
    TiledTreeNode(DecisionTree<>& owningTree, TiledTree& tiledTree, int32_t tileID, int32_t tileIndex)
        :m_owningTree(owningTree), m_tiledTree(tiledTree), m_tileID(tileID), m_tileIndex(tileIndex), m_hasExtraNodes(false)
    { }
    std::vector<int32_t>& GetNodeIndices() { return m_nodeIndices; }

    int32_t GetParent() { return m_parent; }
    const std::vector<int32_t>& GetChildren() { return m_children; }

    int32_t GetEntryNode() { return m_nodeIndices[0]; }
};

class TiledTree {
    friend class TiledTreeNode;

    std::vector<TiledTreeNode> m_tiles;
    DecisionTree<>& m_owningTree;
    // We may need to add nodes to the original tree to make the tiles full sized. This 
    // tree is the modified tree with nodes added if required.
    DecisionTree<> m_modifiedTree;
    std::map<int32_t, int32_t> m_nodeToTileMap;
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
                               size_t vecIndex, size_t nodeIndex, GetterType get);
    
    int32_t GetTreeDepthHelper(int32_t tileIndex);

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
    int32_t GetTreeDepth() { 
        // The root of the tiled tree should be the first node
        assert (m_tiles[0].GetParent() == DecisionTree<>::INVALID_NODE_INDEX);
        return GetTreeDepthHelper(0);
    }
};

} // decisionforest
} // mlir
#endif // _TILEDTREE_H_