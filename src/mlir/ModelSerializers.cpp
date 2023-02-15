#include "TreeTilingDescriptor.h"
#include "TreeTilingUtils.h"
#include "TiledTree.h"
#include "Logger.h"
#include "ModelSerializers.h"

namespace mlir
{
namespace decisionforest
{

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

void SparseRepresentationSerializer::SetBatchSize(int32_t value){
    m_batchSize = value;
}

void SparseRepresentationSerializer::SetRowSize(int32_t value) {
    m_rowSize = value;
}

void SparseRepresentationSerializer::SetInputTypeBitWidth(int32_t value){
    m_inputTypeBitWidth = value;
}

void SparseRepresentationSerializer::SetReturnTypeBitWidth(int32_t value){
    m_returnTypeBitwidth = value;
}

void SparseRepresentationSerializer::Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) {
    mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(m_filepath);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetBatchSize(m_batchSize);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetRowSize(m_rowSize);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetInputElementBitWidth(m_inputTypeBitWidth);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetReturnTypeBitWidth(m_returnTypeBitwidth);
    PersistDecisionForestSparse(forest, forestType);
}

void SparseRepresentationSerializer::ReadData() {
    decisionforest::ForestJSONReader::GetInstance().SetFilePath(m_filepath);
    decisionforest::ForestJSONReader::GetInstance().ParseJSONFile();
    // TODO read the thresholdSize and featureIndexSize from the JSON!
    m_batchSize = decisionforest::ForestJSONReader::GetInstance().GetBatchSize();
    m_rowSize = decisionforest::ForestJSONReader::GetInstance().GetRowSize();
    m_inputTypeBitWidth = decisionforest::ForestJSONReader::GetInstance().GetInputElementBitWidth();
    m_returnTypeBitwidth = decisionforest::ForestJSONReader::GetInstance().GetReturnTypeBitWidth();
}

void ArrayRepresentationSerializer::SetBatchSize(int32_t value){
    m_batchSize = value;
}

void ArrayRepresentationSerializer::SetRowSize(int32_t value) {
    m_rowSize = value;
}

void ArrayRepresentationSerializer::SetInputTypeBitWidth(int32_t value){
    m_inputTypeBitWidth = value;
}

void ArrayRepresentationSerializer::SetReturnTypeBitWidth(int32_t value){
    m_returnTypeBitwidth = value;
}

void ArrayRepresentationSerializer::Persist(mlir::decisionforest::DecisionForest<>& forest, mlir::decisionforest::TreeEnsembleType forestType) {
    mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(m_filepath);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetBatchSize(m_batchSize);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetRowSize(m_rowSize);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetInputElementBitWidth(m_inputTypeBitWidth);
    mlir::decisionforest::ForestJSONReader::GetInstance().SetReturnTypeBitWidth(m_returnTypeBitwidth);
    PersistDecisionForestArrayBased(forest, forestType);
}

void ArrayRepresentationSerializer::ReadData() {
    decisionforest::ForestJSONReader::GetInstance().SetFilePath(m_filepath);
    decisionforest::ForestJSONReader::GetInstance().ParseJSONFile();
    // TODO read the thresholdSize and featureIndexSize from the JSON!
    m_batchSize = decisionforest::ForestJSONReader::GetInstance().GetBatchSize();
    m_rowSize = decisionforest::ForestJSONReader::GetInstance().GetRowSize();
    m_inputTypeBitWidth = decisionforest::ForestJSONReader::GetInstance().GetInputElementBitWidth();
    m_returnTypeBitwidth = decisionforest::ForestJSONReader::GetInstance().GetReturnTypeBitWidth();
}

// TODO Make this implementation more general by having some kind of registry
std::shared_ptr<IModelSerializer> ModelSerializerFactory::GetModelSerializer(const std::string& name, const std::string& modelGlobalsJSONPath) {
  if (name == "array")
    return std::make_shared<ArrayRepresentationSerializer>(modelGlobalsJSONPath);
  else if (name == "sparse")
    return std::make_shared<SparseRepresentationSerializer>(modelGlobalsJSONPath);
  
  assert(false && "Unknown serialization format");
  return nullptr;
}

std::shared_ptr<IModelSerializer> ConstructModelSerializer(const std::string& modelGlobalsJSONPath) {
  if (decisionforest::UseSparseTreeRepresentation)
    return ModelSerializerFactory::GetModelSerializer("sparse", modelGlobalsJSONPath);
  else
    return ModelSerializerFactory::GetModelSerializer("array", modelGlobalsJSONPath);
}

}
}