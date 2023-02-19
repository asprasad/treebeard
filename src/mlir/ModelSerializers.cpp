#include "TreeTilingDescriptor.h"
#include "TreeTilingUtils.h"
#include "TiledTree.h"
#include "Logger.h"
#include "ModelSerializers.h"
#include "../gpu/GPUModelSerializers.h"

namespace 
{

// void LogLeafDepths(const std::vector<int32_t>& depths) {
//     std::string out;
//     for (auto depth : depths) {
//         out += std::to_string(depth) + ", ";
//     }
//     TreeBeard::Logging::Log(out);
// }

void LogTreeStats(const std::vector<mlir::decisionforest::TiledTreeStats>& tiledTreeStats) {
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

}

namespace mlir
{
namespace decisionforest
{

// ===---------------------------------------------------=== //
// ArraySparseSerializerBase Methods
// ===---------------------------------------------------=== //

int32_t ArraySparseSerializerBase::InitializeLengthsArray() {
  // TODO The ForestJSONReader class needs to provide an interface to iterate over tile sizes and bit widths
  // We need to construct the name of the getter function based on those. 
  typedef LengthMemrefType (*GetLengthFunc_t)();
  auto getLengthPtr = GetFunctionAddress<GetLengthFunc_t>("Get_lengths");
  LengthMemrefType lengthMemref = getLengthPtr();
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengthMemref.alignedPtr,
                                                                               m_inferenceRunner->GetTileSize(),
                                                                               m_inferenceRunner->GetThresholdWidth(),
                                                                               m_inferenceRunner->GetFeatureIndexWidth()); 

  return 0;
}

// TODO All the initialize methods are doing the same thing, except that the getter they're calling are different. 
// Refactor them into a shared method.
int32_t ArraySparseSerializerBase::InitializeOffsetsArray() {
  // TODO The ForestJSONReader class needs to provide an interface to iterate over tile sizes and bit widths
  // We need to construct the name of the getter function based on those. 
  typedef OffsetMemrefType (*GetOffsetsFunc_t)();
  auto getOffsetPtr = GetFunctionAddress<GetOffsetsFunc_t>("Get_offsets");
  auto offsetMemref = getOffsetPtr();
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsetMemref.alignedPtr,
                                                                               m_inferenceRunner->GetTileSize(),
                                                                               m_inferenceRunner->GetThresholdWidth(),
                                                                               m_inferenceRunner->GetFeatureIndexWidth()); 
  return 0;
}

template<typename ThresholdType, typename FeatureIndexType, typename TileShapeType>
int32_t ArraySparseSerializerBase::ResolveChildIndexType() {
  if (!m_sparseRepresentation)
    return CallInitMethod<ThresholdType, FeatureIndexType, TileShapeType, int32_t>();
  else {
    auto childIndexBitWidth = mlir::decisionforest::ForestJSONReader::GetInstance().GetChildIndexBitWidth();
    assert (childIndexBitWidth > 0);
    if (childIndexBitWidth == 8)
      return CallInitMethod<ThresholdType, FeatureIndexType, TileShapeType, int8_t>();
    else if (childIndexBitWidth == 16)
      return CallInitMethod<ThresholdType, FeatureIndexType, TileShapeType, int16_t>();
    else if (childIndexBitWidth == 32)
      return CallInitMethod<ThresholdType, FeatureIndexType, TileShapeType, int32_t>();
    else
      assert (false && "Unsupported child index bit width");
    return -1;
  }
}

template<typename ThresholdType, typename FeatureIndexType>
int32_t ArraySparseSerializerBase::ResolveTileShapeType() {
  auto tileShapeBitWidth = mlir::decisionforest::ForestJSONReader::GetInstance().GetTileShapeBitWidth();
  if (tileShapeBitWidth == 8)
    return ResolveChildIndexType<ThresholdType, FeatureIndexType, int8_t>();
  else if (tileShapeBitWidth == 16)
    return ResolveChildIndexType<ThresholdType, FeatureIndexType, int16_t>();
  else if (tileShapeBitWidth == 32)
    return ResolveChildIndexType<ThresholdType, FeatureIndexType, int32_t>();
  else
    assert (false && "Unsupported tile shape bit width");
  return -1;
}


template<typename ThresholdType, typename FeatureIndexType, typename TileShapeType, typename ChildIndexType>
int32_t ArraySparseSerializerBase::CallInitMethod() {
  auto tileSize = m_inferenceRunner->GetTileSize();
  auto thresholdSize = m_inferenceRunner->GetThresholdWidth();
  auto featureIndexSize = m_inferenceRunner->GetFeatureIndexWidth();
  std::vector<ThresholdType> thresholds;
  std::vector<FeatureIndexType> featureIndices;
  std::vector<TileShapeType> tileShapeIDs;
  std::vector<ChildIndexType> childIndices;

  if (!m_sparseRepresentation)
    mlir::decisionforest::ForestJSONReader::GetInstance().GetModelValues(tileSize, thresholdSize, featureIndexSize, thresholds, featureIndices, tileShapeIDs);
  else {
    assert (m_sparseRepresentation);
    mlir::decisionforest::ForestJSONReader::GetInstance().GetModelValues(tileSize, thresholdSize, featureIndexSize, 
                                                                         thresholds, featureIndices, tileShapeIDs, childIndices);
  }

  Memref<ThresholdType, 1> thresholdsMemref{thresholds.data(), thresholds.data(), 0, {(int64_t)thresholds.size()}, 1};
  Memref<FeatureIndexType, 1> featureIndexMemref{featureIndices.data(), featureIndices.data(), 0, {(int64_t)featureIndices.size()}, 1};
  Memref<TileShapeType, 1> tileShapeIDMemref{tileShapeIDs.data(), tileShapeIDs.data(), 0, {(int64_t)tileShapeIDs.size()}, 1};
  int32_t returnValue = -1;
  
  if (!m_sparseRepresentation) {
    typedef int32_t (*InitModelPtr_t)(ThresholdType*, ThresholdType*, int64_t, int64_t, int64_t, FeatureIndexType*, FeatureIndexType*, int64_t, int64_t, int64_t,
                                      TileShapeType*, TileShapeType*, int64_t, int64_t, int64_t);
    auto initModelPtr = GetFunctionAddress<InitModelPtr_t>("Init_model");

    returnValue = initModelPtr(thresholdsMemref.bufferPtr, thresholdsMemref.alignedPtr, thresholdsMemref.offset, thresholdsMemref.lengths[0], thresholdsMemref.strides[0],
                featureIndexMemref.bufferPtr, featureIndexMemref.alignedPtr, featureIndexMemref.offset, featureIndexMemref.lengths[0], featureIndexMemref.strides[0],
                tileShapeIDMemref.bufferPtr, tileShapeIDMemref.alignedPtr, tileShapeIDMemref.offset, tileShapeIDMemref.lengths[0], tileShapeIDMemref.strides[0]);
  }
  else {
    assert (m_sparseRepresentation);
    Memref<ChildIndexType, 1> childIndexMemref{childIndices.data(), childIndices.data(), 0, {(int64_t)childIndices.size()}, 1};
    typedef int32_t (*InitModelPtr_t)(ThresholdType*, ThresholdType*, int64_t, int64_t, int64_t, FeatureIndexType*, FeatureIndexType*, int64_t, int64_t, int64_t,
                                      TileShapeType*, TileShapeType*, int64_t, int64_t, int64_t, ChildIndexType*, ChildIndexType*, int64_t, int64_t, int64_t);
    auto initModelPtr = GetFunctionAddress<InitModelPtr_t>("Init_model");

    returnValue = initModelPtr(thresholdsMemref.bufferPtr, thresholdsMemref.alignedPtr, thresholdsMemref.offset, thresholdsMemref.lengths[0], thresholdsMemref.strides[0],
                featureIndexMemref.bufferPtr, featureIndexMemref.alignedPtr, featureIndexMemref.offset, featureIndexMemref.lengths[0], featureIndexMemref.strides[0],
                tileShapeIDMemref.bufferPtr, tileShapeIDMemref.alignedPtr, tileShapeIDMemref.offset, tileShapeIDMemref.lengths[0], tileShapeIDMemref.strides[0],
                childIndexMemref.bufferPtr, childIndexMemref.alignedPtr, childIndexMemref.offset, childIndexMemref.lengths[0], childIndexMemref.strides[0]);
  }
  if (TreeBeard::Logging::loggingOptions.logGenCodeStats) {
    TreeBeard::Logging::Log("Model memref size : " + std::to_string(returnValue));
    std::set<int32_t> tileShapes(tileShapeIDs.begin(), tileShapeIDs.end());
    TreeBeard::Logging::Log("Number of unique tile shapes : " + std::to_string(tileShapes.size()));
  }
  assert(returnValue != -1);
  return returnValue;
}

int32_t ArraySparseSerializerBase::InitializeModelArray() {
  // TODO The ForestJSONReader class needs to provide an interface to iterate over tile sizes and bit widths
  // We need to construct the name of the getter function based on those.
  auto thresholdSize = m_inferenceRunner->GetThresholdWidth();
  auto featureIndexSize = m_inferenceRunner->GetFeatureIndexWidth();
  if (thresholdSize == 64) {
    if (featureIndexSize == 32) {
      return ResolveTileShapeType<double, int32_t>();
    }
    else if (featureIndexSize == 16) {
      return ResolveTileShapeType<double, int16_t>();
    }
    else if (featureIndexSize == 8) {
      return ResolveTileShapeType<double, int8_t>();
    }
    else {
      assert (false);
    }
  }
  else if (thresholdSize == 32) {
    if (featureIndexSize == 32) {
      return ResolveTileShapeType<float, int32_t>();
    }
    else if (featureIndexSize == 16) {
      return ResolveTileShapeType<float, int16_t>();
    }
    else if (featureIndexSize == 8) {
      return ResolveTileShapeType<float, int8_t>();
    }
    else {
      assert (false);
    }
  }
  else {
    assert (false);
  }
  return 0;
}

void ArraySparseSerializerBase::InitializeClassInformation() {
  
  if (mlir::decisionforest::ForestJSONReader::GetInstance().GetNumberOfClasses() == 0) return;
   
  typedef ClassMemrefType (*GetClassMemref_t)();
  auto getClassInfoPtr = GetFunctionAddress<GetClassMemref_t>("Get_treeClassInfo");
  ClassMemrefType treeClassInfo = getClassInfoPtr();

  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeClassInformation(treeClassInfo.alignedPtr,
                                                                                   m_inferenceRunner->GetTileSize(),
                                                                                   m_inferenceRunner->GetThresholdWidth(),
                                                                                   m_inferenceRunner->GetFeatureIndexWidth()); 
}

void ArraySparseSerializerBase::ReadData() {
    decisionforest::ForestJSONReader::GetInstance().SetFilePath(m_filepath);
    decisionforest::ForestJSONReader::GetInstance().ParseJSONFile();
    // TODO read the thresholdSize and featureIndexSize from the JSON!
    m_batchSize = decisionforest::ForestJSONReader::GetInstance().GetBatchSize();
    m_rowSize = decisionforest::ForestJSONReader::GetInstance().GetRowSize();
    m_inputTypeBitWidth = decisionforest::ForestJSONReader::GetInstance().GetInputElementBitWidth();
    m_returnTypeBitwidth = decisionforest::ForestJSONReader::GetInstance().GetReturnTypeBitWidth();
}

void ArraySparseSerializerBase::SetBatchSize(int32_t value){
    m_batchSize = value;
}

void ArraySparseSerializerBase::SetRowSize(int32_t value) {
    m_rowSize = value;
}

void ArraySparseSerializerBase::SetInputTypeBitWidth(int32_t value){
    m_inputTypeBitWidth = value;
}

void ArraySparseSerializerBase::SetReturnTypeBitWidth(int32_t value){
    m_returnTypeBitwidth = value;
}

// ===---------------------------------------------------=== //
// Persistence Helper Methods
// ===---------------------------------------------------=== //

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
    for (size_t i=0; i<numTrees ; ++i) {
        auto treeType = forestType.getTreeType(0).cast<decisionforest::TreeType>();
        
        auto currentTreeTileShapeBitWidth = treeType.getTileShapeType().getIntOrFloatBitWidth();
        assert (tileShapeBitWidth==0 || currentTreeTileShapeBitWidth==tileShapeBitWidth);
        tileShapeBitWidth=currentTreeTileShapeBitWidth;

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
            // std::string dotFile = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/temp/tiledTree_" + std::to_string(i) + ".dot";
            // tiledTree.WriteDOTFile(dotFile);
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
    
    if (TreeBeard::Logging::loggingOptions.logTreeStats) {
        LogTreeStats(treeStats);
        LogTileShapeStats(numberOfTileShapes, "Tile shapes");
        LogTileShapeStats(numberOfOriginalTileShapes, "Original tile shapes");
        LogTileShapeStats(expectedNumberOfHops, "Expected number of hops");
        LogTileShapeStats(idealExpectedNumberOfHops, "Ideal expected number of hops");
        LogTileShapeStats(numberOfNonSubsetTiles, "Non subset tiles");
    }

    mlir::decisionforest::ForestJSONReader::GetInstance().WriteJSONFile();
    
    // The below lines clear the model globals json file path that is currently persisted. This is to test that 
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
                auto childIndexBitWidth = treeType.getChildIndexType().getIntOrFloatBitWidth();
                mlir::decisionforest::ForestJSONReader::GetInstance().SetChildIndexBitWidth(childIndexBitWidth);
            },
            [](TiledTree& tiledTree, int32_t treeNumber, decisionforest::TreeType treeType) {
                std::vector<ThresholdType> thresholds;
                std::vector<FeatureIndexType> featureIndices;
                std::vector<int32_t> tileShapeIDs, leafBitMasks;
                std::vector<int32_t> childIndices, leafIndices;
                std::vector<double> leaves;
                tiledTree.GetSparseSerialization(thresholds, featureIndices, tileShapeIDs, childIndices, leaves);
                int32_t numTiles = tileShapeIDs.size();
                int32_t tileSize = tiledTree.TileSize();
                int32_t classId = tiledTree.GetClassId();
                mlir::decisionforest::ForestJSONReader::GetInstance().AddSingleSparseTree(treeNumber, numTiles, thresholds, featureIndices, tileShapeIDs, childIndices,
                                                                                    leaves, tileSize, treeType.getThresholdType().getIntOrFloatBitWidth(), 
                                                                                    treeType.getFeatureIndexType().getIntOrFloatBitWidth(), classId);
                
                auto childIndexBitWidth = treeType.getChildIndexType().getIntOrFloatBitWidth();
                mlir::decisionforest::ForestJSONReader::GetInstance().SetChildIndexBitWidth(childIndexBitWidth);
            }
    );
}

// ===---------------------------------------------------=== //
// SparseRepresentationSerializer Methods
// ===---------------------------------------------------=== //

void SparseRepresentationSerializer::Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) {
    mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(m_filepath);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetBatchSize(m_batchSize);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetRowSize(m_rowSize);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetInputElementBitWidth(m_inputTypeBitWidth);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetReturnTypeBitWidth(m_returnTypeBitwidth);
    PersistDecisionForestSparse(forest, forestType);
}

int32_t SparseRepresentationSerializer::InitializeLeafArrays() {
  auto tileSize = m_inferenceRunner->GetTileSize();
  auto thresholdSize = m_inferenceRunner->GetThresholdWidth();
  auto featureIndexSize = m_inferenceRunner->GetFeatureIndexWidth();
  {
    // Initialize the leaf values
    typedef Memref<double, 1> (*GetLeavesFunc_t)();
    auto getLeavesPtr = GetFunctionAddress<GetLeavesFunc_t>("Get_leaves");
    auto leavesMemref = getLeavesPtr();
    mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLeaves(leavesMemref.alignedPtr, tileSize, thresholdSize, featureIndexSize); 
  }
  {
    // Initialize leaf offsets
    typedef OffsetMemrefType (*GetOffsetsFunc_t)();
    auto getOffsetPtr = GetFunctionAddress<GetOffsetsFunc_t>("Get_leavesOffsets");
    auto offsetMemref = getOffsetPtr();
    mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLeavesOffsetBuffer(offsetMemref.alignedPtr, tileSize, thresholdSize, featureIndexSize);
  }
  {
    typedef LengthMemrefType (*GetLengthFunc_t)();
    auto getLengthPtr = GetFunctionAddress<GetLengthFunc_t>("Get_leavesLengths");
    LengthMemrefType lengthMemref = getLengthPtr();
    mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLeavesLengthBuffer(lengthMemref.alignedPtr, tileSize, 
                                                                                       thresholdSize, featureIndexSize); 
  }
  return 0;
}

void SparseRepresentationSerializer::InitializeBuffersImpl() {
    InitializeLengthsArray();
    InitializeOffsetsArray();
    InitializeModelArray();
    InitializeClassInformation();
    InitializeLeafArrays();
}

std::shared_ptr<IModelSerializer> ConstructSparseRepresentation(const std::string& jsonFilename) {
  return std::make_shared<SparseRepresentationSerializer>(jsonFilename);
}

REGISTER_SERIALIZER(sparse, ConstructSparseRepresentation)

// ===---------------------------------------------------=== //
// ArrayRepresentationSerializer Methods
// ===---------------------------------------------------=== //

void ArrayRepresentationSerializer::Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) {
    mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(m_filepath);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetBatchSize(m_batchSize);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetRowSize(m_rowSize);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetInputElementBitWidth(m_inputTypeBitWidth);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetReturnTypeBitWidth(m_returnTypeBitwidth);
    PersistDecisionForestArrayBased(forest, forestType);
}

void ArrayRepresentationSerializer::InitializeBuffersImpl() {
    InitializeLengthsArray();
    InitializeOffsetsArray();
    InitializeModelArray();
    InitializeClassInformation();
}

std::shared_ptr<IModelSerializer> ConstructArrayRepresentation(const std::string& jsonFilename) {
  return std::make_shared<ArrayRepresentationSerializer>(jsonFilename);
}

REGISTER_SERIALIZER(array, ConstructArrayRepresentation)

// ===---------------------------------------------------=== //
// ModelSerializerFactory Methods
// ===---------------------------------------------------=== //

std::shared_ptr<IModelSerializer> ModelSerializerFactory::GetModelSerializer(const std::string& name, 
                                                                             const std::string& modelGlobalsJSONPath) {
  auto mapIter = m_constructionMap.find(name);
  assert (mapIter != m_constructionMap.end() && "Unknown serializer name!");
  return mapIter->second(modelGlobalsJSONPath);
}

ModelSerializerFactory& ModelSerializerFactory::Get() {
  static std::unique_ptr<ModelSerializerFactory> s_instancePtr = nullptr;
  if (s_instancePtr == nullptr)
    s_instancePtr = std::make_unique<ModelSerializerFactory>();
  return *s_instancePtr; 
}

bool ModelSerializerFactory::RegisterSerializer(const std::string& name,
                                                SerializerConstructor_t constructionFunc) {
  assert (m_constructionMap.find(name) == m_constructionMap.end());
  m_constructionMap[name] = constructionFunc;
  return true;
}

std::shared_ptr<IModelSerializer> ConstructModelSerializer(const std::string& modelGlobalsJSONPath) {
  if (decisionforest::UseSparseTreeRepresentation)
    return ModelSerializerFactory::Get().GetModelSerializer("sparse", modelGlobalsJSONPath);
  else
    return ModelSerializerFactory::Get().GetModelSerializer("array", modelGlobalsJSONPath);
}

std::shared_ptr<IModelSerializer> ConstructGPUModelSerializer(const std::string& modelGlobalsJSONPath) {
  if (decisionforest::UseSparseTreeRepresentation)
    return ModelSerializerFactory::Get().GetModelSerializer("gpu_sparse", modelGlobalsJSONPath);
  else
    return ModelSerializerFactory::Get().GetModelSerializer("gpu_array", modelGlobalsJSONPath);
}

}
}