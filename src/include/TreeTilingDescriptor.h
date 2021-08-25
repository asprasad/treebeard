#ifndef _TREETILINGDESCRIPTOR_H_
#define _TREETILINGDESCRIPTOR_H_

#include <cstdint>
#include <vector>
#include <string>

// *. Enum to denote regular or irregular tiling
// *. Utilities to verify that a tiling is valid
//      - All nodes are covered, tiles are connected, # of nodes are within the limit, tiles are disjoint
// *. Utilities to manage tiles -- for example, number of tile shapes with a given tile size and thier IDs, 
//    child indices based on comparison outcomes etc.
// *. Check if two tilings are equivalent

namespace mlir
{
namespace decisionforest
{

enum class TilingType { kRegular, kIrregular, kUnknown };

class TreeTilingDescriptor 
{
    // Number of nodes in each tile
    int32_t m_maxTileSize;

    // Number of tiles in this tree
    int32_t m_numTiles;

    // Map from node index to tile (each entry is an int in 0...(numTiles-1))
    std::vector<int32_t> m_tileIDs; 
    
    TilingType m_tilingType;
public:
    TreeTilingDescriptor()
        : m_maxTileSize(1), m_numTiles(-1), m_tilingType(TilingType::kRegular)
    { }

    TreeTilingDescriptor(int32_t maxTileSize, int32_t numTiles, const std::vector<int32_t>& ids, TilingType tileType)
        : m_maxTileSize(maxTileSize), m_numTiles(numTiles), m_tileIDs(ids), m_tilingType(tileType)
    { }

    int32_t MaxTileSize() const { return m_maxTileSize; }
    int32_t NumTiles() const { return m_numTiles; }
    TilingType TileType() const { return m_tilingType; }
    const std::vector<int32_t>&  TileIDs() const { return m_tileIDs; }
    std::vector<int32_t>&  TileIDs() { return m_tileIDs; }
    
    std::string ToHashString() const;
    std::string ToPrintString() const;
    
    bool operator==(const TreeTilingDescriptor& other) const;
    TreeTilingDescriptor& operator=(const TreeTilingDescriptor& other) = default;
};

} // decisionforest
} // mlir
#endif // _TREETILINGDESCRIPTOR_H_