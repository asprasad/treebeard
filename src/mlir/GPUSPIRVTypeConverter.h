#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "TreebeardContext.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class GPUSPIRVTypeConverter : public SPIRVTypeConverter {
public:
  using TypeConverter::convertType;

  explicit GPUSPIRVTypeConverter(spirv::TargetEnvAttr &targetAttr,
                                 SPIRVConversionOptions &option)
      : SPIRVTypeConverter(targetAttr, option), targetAttr(targetAttr),
        options(option) {
    addConversion([this](MemRefType memRefType) {
      spirv::TargetEnvAttr &localTargetAttr = this->targetAttr;
      SPIRVConversionOptions &localOptions = this->options;
      SPIRVTypeConverter spirvTypeConverter(localTargetAttr, localOptions);
      auto attr = dyn_cast_or_null<spirv::StorageClassAttr>(memRefType.getMemorySpace());
      spirv::StorageClass storageClass = attr.getValue();
      Type elementType = memRefType.getElementType();
      auto TileType =
          elementType
              .dyn_cast_or_null<decisionforest::TiledNumericalNodeType>();
      auto ReorgType =
          elementType
              .dyn_cast_or_null<decisionforest::ReorgMemrefElementType>();
      if (TileType) {
        // Retrieve field types
        auto thresholdType = TileType.getThresholdFieldType();
        auto indexType = TileType.getIndexFieldType();
        auto childIndexType = TileType.getChildIndexType();
        auto tileShapeIDType = TileType.getTileShapeType();

        Type structType = nullptr;
        // SPIR-V type construction logic
        if (TileType.getTileSize() == 1) {
          // Construct a SPIR-V struct type for a single-tile node
          structType = spirv::StructType::get({thresholdType, indexType, childIndexType});
        } else {
          // Retrieve additional type for multi-tile nodes
          auto tileShapeIDType = TileType.getTileShapeType();
          structType = spirv::StructType::get(
              {thresholdType, indexType, tileShapeIDType, childIndexType});
        }

        unsigned int size = 1;
        int rank = memRefType.getRank();
        if (memRefType.hasStaticShape() && rank) {
          size = memRefType.getDimSize(0);
          auto tileSize = TileType.getTileSize();
          auto arrayType = spirv::ArrayType::get(structType, size * tileSize);
          Type convertedMemRefType = spirv::PointerType::get(
              arrayType, storageClass);
          return convertedMemRefType;
        } else {
          Type convertedMemRefType = spirv::PointerType::get(
              structType, storageClass);
          return convertedMemRefType;
        }
      } else if (ReorgType) {
        auto elemType = ReorgType.getElementType();
        unsigned int size = 1;
        int rank = memRefType.getRank();
        if (memRefType.hasStaticShape() && rank) {
          size = memRefType.getDimSize(0);
          auto arrayType = spirv::ArrayType::get(elemType, size);
          Type convertedMemRefType = spirv::PointerType::get(
              arrayType, storageClass);
          return convertedMemRefType;
        } else {
          Type convertedMemRefType = spirv::PointerType::get(
              elemType, storageClass);
          return convertedMemRefType;
        }
      }
      return spirvTypeConverter.convertType(memRefType);
    });
  }

private:
  spirv::TargetEnvAttr &targetAttr;
  SPIRVConversionOptions &options;
};
} // namespace mlir