#include "TreeTilingDescriptor.h"
#include <sstream>

namespace mlir
{
namespace decisionforest
{

bool TreeTilingDescriptor::operator==(const TreeTilingDescriptor& other) const
{
    return m_tilingType==other.m_tilingType && m_maxTileSize==other.m_maxTileSize && m_tileIDs==other.m_tileIDs
           && m_numTiles==other.m_numTiles;
}

std::string TreeTilingDescriptor::ToHashString() const
{
    std::stringstream strStream;
    strStream << (int32_t)m_tilingType << m_maxTileSize << m_numTiles;
    for (auto id : m_tileIDs)
        strStream << id;
    return strStream.str();
}

std::string TreeTilingDescriptor::ToPrintString() const
{
    std::stringstream strStream;
    strStream << "TilingType = " << (int32_t)m_tilingType << ", MaxTileSize = " << m_maxTileSize << ", NumberOfTiles = " << m_numTiles;
    return strStream.str();
}

} //decisionforest
} //mlir