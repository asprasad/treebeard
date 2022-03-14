#include <vector>
#include <sstream>
#include <chrono>
#include "TreeTilingUtils.h"
#include "ExecutionHelpers.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "xgboostparser.h"
#include "TiledTree.h"

#include "TestUtilsCommon.h"
#include "ForestTestUtils.h"

namespace TreeBeard
{
namespace test
{

// Codegen tests
bool Test_LoadTileFeatureIndicesOp_DoubleInt32_TileSize1(TestArgs_t& args);
bool Test_LoadTileThresholdOp_DoubleInt32_TileSize1(TestArgs_t& args);
bool Test_LoadTileThresholdOp_Subview_DoubleInt32_TileSize1(TestArgs_t& args);
bool Test_LoadTileFeatureIndicesOp_Subview_DoubleInt32_TileSize1(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize1(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize2(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize4(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize1(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize2(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize4(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize1(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize2(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize4(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize1_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize2_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize4_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize1_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize2_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize4_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize1_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize2_Float(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize4_Float(TestArgs_t& args);

// Tiled Codegen tests
bool Test_TiledCodeGeneration_RightHeavy_BatchSize1(TestArgs_t& args);
bool Test_TiledCodeGeneration_LeftHeavy_BatchSize1(TestArgs_t& args);
bool Test_TiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape(TestArgs_t& args);
bool Test_TiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape(TestArgs_t& args);
bool Test_TiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape(TestArgs_t& args);
bool Test_TiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape(TestArgs_t& args);
bool Test_TiledCodeGeneration_BalancedTree_BatchSize1(TestArgs_t& args);
bool Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1(TestArgs_t& args);
bool Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize(TestArgs_t& args);
bool Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize(TestArgs_t& args);

// Tiled Init Tests
bool Test_ModelInit_LeftHeavy(TestArgs_t& args);
bool Test_ModelInit_RightHeavy(TestArgs_t& args);
bool Test_ModelInit_RightAndLeftHeavy(TestArgs_t& args);
bool Test_ModelInit_Balanced(TestArgs_t& args);
bool Test_ModelInit_LeftHeavy_Int8TileShape(TestArgs_t& args);
bool Test_ModelInit_LeftHeavy_Int16TileShape(TestArgs_t& args);
bool Test_ModelInit_RightHeavy_Int8TileShape(TestArgs_t& args);
bool Test_ModelInit_RightHeavy_Int16TileShape(TestArgs_t& args);
bool Test_ModelInit_Balanced_Int8TileShape(TestArgs_t& args);
bool Test_ModelInit_Balanced_Int16TileShape(TestArgs_t& args);
bool Test_ModelInit_RightAndLeftHeavy_Int8TileShape(TestArgs_t& args);
bool Test_ModelInit_RightAndLeftHeavy_Int16TileShape(TestArgs_t& args);

// Uniform Tiling Tests
bool Test_UniformTiling_LeftHeavy_BatchSize1(TestArgs_t& args);
bool Test_UniformTiling_RightHeavy_BatchSize1(TestArgs_t &args);
bool Test_UniformTiling_Balanced_BatchSize1(TestArgs_t &args);
bool Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1(TestArgs_t &args);
bool Test_UniformTiling_LeftHeavy_BatchSize1_Int8TileShape(TestArgs_t& args);
bool Test_UniformTiling_RightHeavy_BatchSize1_Int8TileShape(TestArgs_t &args);
bool Test_UniformTiling_Balanced_BatchSize1_Int8TileShape(TestArgs_t &args);
bool Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1_Int8TileShape(TestArgs_t &args);
bool Test_UniformTiling_LeftHeavy_BatchSize1_Int16TileShape(TestArgs_t& args);
bool Test_UniformTiling_RightHeavy_BatchSize1_Int16TileShape(TestArgs_t &args);
bool Test_UniformTiling_Balanced_BatchSize1_Int16TileShape(TestArgs_t &args);
bool Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1_Int16TileShape(TestArgs_t &args);

bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize1(TestArgs_t& args);
bool Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize1(TestArgs_t& args);
bool Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize1(TestArgs_t& args);
bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize2(TestArgs_t& args);
bool Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize2(TestArgs_t& args);
bool Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize2(TestArgs_t& args);
bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4(TestArgs_t& args);
bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4_Int8TileShape(TestArgs_t& args);
bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4_Int16TileShape(TestArgs_t& args);
bool Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4(TestArgs_t& args);
bool Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4_Int8TileShape(TestArgs_t& args);
bool Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4_Int16TileShape(TestArgs_t& args);
bool Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4(TestArgs_t& args);
bool Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4_Int8TileShape(TestArgs_t& args);
bool Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4_Int16TileShape(TestArgs_t& args);

// XGBoost benchmark models tests
bool Test_Scalar_Abalone(TestArgs_t &args);
bool Test_TileSize2_Abalone(TestArgs_t &args);
bool Test_TileSize3_Abalone(TestArgs_t &args);
bool Test_TileSize4_Abalone(TestArgs_t &args);
bool Test_TileSize8_Abalone(TestArgs_t &args);

bool Test_Scalar_Airline(TestArgs_t &args);
bool Test_TileSize2_Airline(TestArgs_t &args);
bool Test_TileSize3_Airline(TestArgs_t &args);
bool Test_TileSize4_Airline(TestArgs_t &args);
bool Test_TileSize8_Airline(TestArgs_t &args);

bool Test_Scalar_AirlineOHE(TestArgs_t &args);
bool Test_TileSize2_AirlineOHE(TestArgs_t &args);
bool Test_TileSize3_AirlineOHE(TestArgs_t &args);
bool Test_TileSize4_AirlineOHE(TestArgs_t &args);
bool Test_TileSize8_AirlineOHE(TestArgs_t &args);

bool Test_Scalar_Bosch(TestArgs_t &args);
bool Test_TileSize2_Bosch(TestArgs_t &args);
bool Test_TileSize3_Bosch(TestArgs_t &args);
bool Test_TileSize4_Bosch(TestArgs_t &args);
bool Test_TileSize8_Bosch(TestArgs_t &args);

bool Test_Scalar_Epsilon(TestArgs_t &args);
bool Test_TileSize2_Epsilon(TestArgs_t &args);
bool Test_TileSize3_Epsilon(TestArgs_t &args);
bool Test_TileSize4_Epsilon(TestArgs_t &args);
bool Test_TileSize8_Epsilon(TestArgs_t &args);

bool Test_Scalar_Higgs(TestArgs_t &args);
bool Test_TileSize2_Higgs(TestArgs_t &args);
bool Test_TileSize3_Higgs(TestArgs_t &args);
bool Test_TileSize4_Higgs(TestArgs_t &args);
bool Test_TileSize8_Higgs(TestArgs_t &args);

bool Test_Scalar_Year(TestArgs_t &args);
bool Test_TileSize2_Year(TestArgs_t &args);
bool Test_TileSize3_Year(TestArgs_t &args);
bool Test_TileSize4_Year(TestArgs_t &args);
bool Test_TileSize8_Year(TestArgs_t &args);

// XGBoost benchmark models tests with one tree at a time schedule
bool Test_Scalar_Abalone_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize2_Abalone_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize3_Abalone_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize4_Abalone_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize8_Abalone_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_Scalar_Airline_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize2_Airline_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize3_Airline_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize4_Airline_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize8_Airline_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_Scalar_AirlineOHE_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize2_AirlineOHE_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize3_AirlineOHE_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize4_AirlineOHE_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize8_AirlineOHE_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_Scalar_Bosch_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize2_Bosch_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize3_Bosch_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize4_Bosch_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize8_Bosch_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_Scalar_Epsilon_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize2_Epsilon_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize3_Epsilon_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize4_Epsilon_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize8_Epsilon_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_Scalar_Higgs_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize2_Higgs_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize3_Higgs_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize4_Higgs_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize8_Higgs_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_Scalar_Year_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize2_Year_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize3_Year_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize4_Year_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize8_Year_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize1_CovType_Int8Type(TestArgs_t &args);
bool Test_TileSize2_CovType_Int8Type(TestArgs_t &args);
bool Test_TileSize3_CovType_Int8Type(TestArgs_t &args);
bool Test_TileSize4_CovType_Int8Type(TestArgs_t &args);
bool Test_TileSize8_CovType_Int8Type(TestArgs_t &args);

// Sparse XGBoost Scalar Tests
bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize1(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize2(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize4(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize1(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize2(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize4(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize1(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize2(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize4(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize1_Float(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize2_Float(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize4_Float(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize1_Float(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize2_Float(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize4_Float(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize1_Float(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize2_Float(TestArgs_t& args);
bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize4_Float(TestArgs_t& args);

// Sparse Tiled Code Gen
bool Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1(TestArgs_t& args);
bool Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize(TestArgs_t& args);
bool Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize(TestArgs_t& args);
bool Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1(TestArgs_t& args);
bool Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1(TestArgs_t& args);
bool Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape(TestArgs_t& args);
bool Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape(TestArgs_t& args);
bool Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape(TestArgs_t& args);
bool Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape(TestArgs_t& args);

// XGBoost Sparse Uniform Tiling Tests
bool Test_SparseUniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4(TestArgs_t& args);
bool Test_SparseUniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4(TestArgs_t& args);
bool Test_SparseUniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4(TestArgs_t& args);

// Sparse XGBoost Benchmark Correctness Tests
bool Test_SparseScalar_Abalone(TestArgs_t &args);
bool Test_SparseTileSize2_Abalone(TestArgs_t &args);
bool Test_SparseTileSize3_Abalone(TestArgs_t &args);
bool Test_SparseTileSize4_Abalone(TestArgs_t &args);
bool Test_SparseTileSize8_Abalone(TestArgs_t &args);
bool Test_SparseScalar_Airline(TestArgs_t &args);
bool Test_SparseTileSize2_Airline(TestArgs_t &args);
bool Test_SparseTileSize3_Airline(TestArgs_t &args);
bool Test_SparseTileSize4_Airline(TestArgs_t &args);
bool Test_SparseTileSize8_Airline(TestArgs_t &args);
bool Test_SparseScalar_AirlineOHE(TestArgs_t &args);
bool Test_SparseTileSize2_AirlineOHE(TestArgs_t &args);
bool Test_SparseTileSize3_AirlineOHE(TestArgs_t &args);
bool Test_SparseTileSize4_AirlineOHE(TestArgs_t &args);
bool Test_SparseTileSize8_AirlineOHE(TestArgs_t &args);
bool Test_SparseScalar_Bosch(TestArgs_t &args);
bool Test_SparseTileSize2_Bosch(TestArgs_t &args);
bool Test_SparseTileSize3_Bosch(TestArgs_t &args);
bool Test_SparseTileSize4_Bosch(TestArgs_t &args);
bool Test_SparseTileSize8_Bosch(TestArgs_t &args);
bool Test_SparseScalar_CovType_Int8Type(TestArgs_t &args);
bool Test_SparseTileSize8_CovType_Int8Type(TestArgs_t &args);
bool Test_SparseScalar_Epsilon(TestArgs_t &args);
bool Test_SparseTileSize2_Epsilon(TestArgs_t &args);
bool Test_SparseTileSize3_Epsilon(TestArgs_t &args);
bool Test_SparseTileSize4_Epsilon(TestArgs_t &args);
bool Test_SparseTileSize8_Epsilon(TestArgs_t &args);
bool Test_SparseScalar_Higgs(TestArgs_t &args);
bool Test_SparseTileSize2_Higgs(TestArgs_t &args);
bool Test_SparseTileSize3_Higgs(TestArgs_t &args);
bool Test_SparseTileSize4_Higgs(TestArgs_t &args);
bool Test_SparseTileSize8_Higgs(TestArgs_t &args);
bool Test_SparseScalar_Year(TestArgs_t &args);
bool Test_SparseTileSize2_Year(TestArgs_t &args);
bool Test_SparseTileSize3_Year(TestArgs_t &args);
bool Test_SparseTileSize4_Year(TestArgs_t &args);
bool Test_SparseTileSize8_Year(TestArgs_t &args);

// Tests for actual model inputs
bool Test_TileSize8_Abalone_TestInputs(TestArgs_t &args);
bool Test_TileSize8_Airline_TestInputs(TestArgs_t &args);
bool Test_TileSize8_AirlineOHE_TestInputs(TestArgs_t &args);
bool Test_TileSize8_Bosch_TestInputs(TestArgs_t &args);
bool Test_TileSize8_Epsilon_TestInputs(TestArgs_t &args);
bool Test_TileSize8_Higgs_TestInputs(TestArgs_t &args);
bool Test_TileSize8_Year_TestInputs(TestArgs_t &args);
bool Test_TileSize8_CovType_TestInputs(TestArgs_t &args);

// Tiled schedule test
bool Test_TileSize8_Abalone_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_TileSize8_AirlineOHE_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_TileSize8_Airline_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_TileSize8_Bosch_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_TileSize8_Epsilon_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_TileSize8_Higgs_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_TileSize8_Year_TestInputs_TiledSchedule(TestArgs_t &args);

// Sparse tests with flipped loops
bool Test_SparseScalar_Abalone_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_Abalone_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseScalar_Airline_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_Airline_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseScalar_AirlineOHE_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_AirlineOHE_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseScalar_Bosch_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_Bosch_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseScalar_CovType_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_CovType_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseScalar_Epsilon_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_Epsilon_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseScalar_Higgs_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_Higgs_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseScalar_Year_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_Year_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_Scalar_CovType_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_TileSize8_CovType_OneTreeAtATimeSchedule(TestArgs_t &args);

bool Test_SparseTileSize8_Abalone_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_AirlineOHE_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_Airline_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_Bosch_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_Epsilon_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_Higgs_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_Year_TestInputs_TiledSchedule(TestArgs_t &args);

// Stats tests
bool Test_AbaloneStatGenerationAndReading(TestArgs_t &args);
bool Test_AirlineStatGenerationAndReading(TestArgs_t &args);
bool Test_AirlineOHEStatGenerationAndReading(TestArgs_t &args);
bool Test_CovtypeStatGenerationAndReading(TestArgs_t &args);
bool Test_EpsilonStatGenerationAndReading(TestArgs_t &args);
bool Test_HiggsStatGenerationAndReading(TestArgs_t &args);
bool Test_YearStatGenerationAndReading(TestArgs_t &args);

// Probability Based Tiling Tests
bool Test_ProbabilisticTiling_TileSize8_Abalone(TestArgs_t &args);
bool Test_SparseProbabilisticTiling_TileSize8_Abalone(TestArgs_t &args);
bool Test_ProbabilisticTiling_TileSize8_Airline(TestArgs_t &args);
bool Test_SparseProbabilisticTiling_TileSize8_Airline(TestArgs_t &args);
bool Test_ProbabilisticTiling_TileSize8_AirlineOHE(TestArgs_t &args);
bool Test_SparseProbabilisticTiling_TileSize8_AirlineOHE(TestArgs_t &args);
bool Test_ProbabilisticTiling_TileSize8_Covtype(TestArgs_t &args);
bool Test_SparseProbabilisticTiling_TileSize8_Covtype(TestArgs_t &args);
bool Test_ProbabilisticTiling_TileSize8_Epsilon(TestArgs_t &args);
bool Test_SparseProbabilisticTiling_TileSize8_Epsilon(TestArgs_t &args);
bool Test_ProbabilisticTiling_TileSize8_Higgs(TestArgs_t &args);
bool Test_SparseProbabilisticTiling_TileSize8_Higgs(TestArgs_t &args);
bool Test_ProbabilisticTiling_TileSize8_Year(TestArgs_t &args);
bool Test_SparseProbabilisticTiling_TileSize8_Year(TestArgs_t &args);

// Make all leaves equal depth tests
bool Test_UniformTiling_Balanced_BatchSize1_EqualDepth(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize4_EqualDepth_TileSize8(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize4_EqualDepth_TileSize8(TestArgs_t& args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize4_EqualDepth_TileSize8(TestArgs_t& args);

void InitializeVectorWithRandValues(std::vector<double>& vec) {
  for(size_t i=0 ; i<vec.size() ; ++i)
    vec[i] = (double)rand()/RAND_MAX;
}

template<typename ThresholdType, typename IndexType>
bool Test_BufferInit_RightHeavy(TestArgs_t& args) {
  using TileType = NumericalTileType_Packed<ThresholdType, IndexType>;
  auto& context = args.context;
  mlir::OpBuilder builder(&context);
  mlir::decisionforest::DecisionForest<> forest;
  auto expectedArray = AddRightHeavyTree<TileType>(forest);

  // Construct basic tree types for the tree
  // (Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type thresholdType, Type featureIndexType
  auto thresholdType = TreeBeard::GetMLIRType(ThresholdType(), builder);
  auto indexType = TreeBeard::GetMLIRType(IndexType(), builder);
  auto treeType = mlir::decisionforest::TreeType::get(thresholdType, 1 /*tileSize*/, thresholdType, indexType);
  //(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType, Type treeType)
  auto forestType = mlir::decisionforest::TreeEnsembleType::get(thresholdType, 1, thresholdType /*HACK type doesn't matter for this test*/,
                                                                mlir::decisionforest::ReductionType::kAdd, treeType);
  
  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(GetGlobalJSONNameForTests());
  mlir::decisionforest::PersistDecisionForest(forest, forestType);
  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(GetGlobalJSONNameForTests());
  mlir::decisionforest::ForestJSONReader::GetInstance().ParseJSONFile();

  std::vector<TileType> serializedTree(std::pow(2, 3) - 1); //Depth of the tree is 3, so this is the size of the dense array
  // InitializeBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth, std::vector<int32_t>& treeOffsets)
  std::vector<int32_t> offsets(1, -1);
  int32_t thresholdSize = sizeof(ThresholdType)*8;
  int32_t indexSize = sizeof(IndexType)*8;
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(serializedTree.data(), 1, thresholdSize, indexSize, offsets);
  Test_ASSERT(expectedArray == serializedTree);
  Test_ASSERT(offsets[0] == 0);
  
  std::vector<int64_t> offsetVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsetVec.data(), 1, thresholdSize, indexSize);
  Test_ASSERT(offsetVec[0] == 0);
  
  std::vector<int64_t> lengthVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengthVec.data(), 1, thresholdSize, indexSize);
  Test_ASSERT(lengthVec[0] == 7);

  mlir::decisionforest::ClearPersistedForest();

  return true;
}

bool Test_BufferInitializationWithOneTree_RightHeavy(TestArgs_t& args) {
  return Test_BufferInit_RightHeavy<double, int32_t>(args);
}

bool Test_BufferInitializationWithOneTree_RightHeavy_Int16(TestArgs_t& args) {
  return Test_BufferInit_RightHeavy<double, int16_t>(args);
}

bool Test_BufferInitializationWithOneTree_RightHeavy_Int8(TestArgs_t& args) {
  return Test_BufferInit_RightHeavy<double, int8_t>(args);
}

bool Test_BufferInitializationWithOneTree_RightHeavy_Float(TestArgs_t& args) {
  return Test_BufferInit_RightHeavy<float, int32_t>(args);
}

bool Test_BufferInitializationWithOneTree_RightHeavy_FloatInt16(TestArgs_t& args) {
  return Test_BufferInit_RightHeavy<float, int16_t>(args);
}

bool Test_BufferInitializationWithOneTree_RightHeavy_FloatInt8(TestArgs_t& args) {
  return Test_BufferInit_RightHeavy<float, int8_t>(args);
}

template<typename ThresholdType, typename IndexType>
bool Test_BufferInitialization_TwoTrees(TestArgs_t& args) {
  using TileType = NumericalTileType_Packed<ThresholdType, IndexType>;
  auto& context = args.context;
  mlir::OpBuilder builder(&context);
  mlir::decisionforest::DecisionForest<> forest;
  auto expectedArray = AddRightHeavyTree<TileType>(forest);
  auto expectedArray2 = AddLeftHeavyTree<TileType>(forest);
  expectedArray.insert(std::end(expectedArray), std::begin(expectedArray2), std::end(expectedArray2));

  // Construct basic tree types for the tree
  // (Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type thresholdType, Type featureIndexType
  auto thresholdType = TreeBeard::GetMLIRType(ThresholdType(), builder);
  auto indexType = TreeBeard::GetMLIRType(IndexType(), builder);
  auto treeType = mlir::decisionforest::TreeType::get(thresholdType, 1 /*tileSize*/, thresholdType, indexType);
  //(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType, Type treeType)
  auto forestType = mlir::decisionforest::TreeEnsembleType::get(thresholdType, 1, thresholdType /*HACK type doesn't matter for this test*/,
                                                                mlir::decisionforest::ReductionType::kAdd, treeType);
  
  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(GetGlobalJSONNameForTests());
  mlir::decisionforest::PersistDecisionForest(forest, forestType);
  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(GetGlobalJSONNameForTests());
  mlir::decisionforest::ForestJSONReader::GetInstance().ParseJSONFile();

  std::vector<TileType> serializedTree(2*(std::pow(2, 3) - 1)); //Depth of the tree is 3, so this is the size of the dense array
  // InitializeBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth, std::vector<int32_t>& treeOffsets)
  int32_t thresholdSize = sizeof(ThresholdType)*8;
  int32_t indexSize = sizeof(IndexType)*8;
  std::vector<int32_t> offsets(2, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(serializedTree.data(), 1, thresholdSize, indexSize, offsets);
  Test_ASSERT(expectedArray == serializedTree);
  Test_ASSERT(offsets[0] == 0);
  Test_ASSERT(offsets[1] == 7);

  std::vector<int64_t> offsetVec(2, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsetVec.data(), 1, thresholdSize, indexSize);
  Test_ASSERT(offsetVec[0] == 0);
  Test_ASSERT(offsetVec[1] == 7);

  std::vector<int64_t> lengthVec(2, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengthVec.data(), 1, thresholdSize, indexSize);
  Test_ASSERT(lengthVec[0] == 7);
  Test_ASSERT(lengthVec[1] == 7);

  mlir::decisionforest::ClearPersistedForest();

  return true;
}

bool Test_BufferInitializationWithOneTree_LeftHeavy(TestArgs_t& args) {
  using DoubleInt32Tile = NumericalTileType_Packed<double, int32_t>;
  mlir::MLIRContext& context = args.context;
  mlir::decisionforest::DecisionForest<> forest;
  auto expectedArray = AddLeftHeavyTree<DoubleInt32Tile>(forest);  

  // Construct basic tree types for the tree
  // (Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type thresholdType, Type featureIndexType
  auto doubleType = mlir::Float64Type::get(&context);
  auto int32Type = mlir::IntegerType::get(&context, 32);
  auto treeType = mlir::decisionforest::TreeType::get(doubleType, 1 /*tileSize*/, doubleType, int32Type);
  //(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType, Type treeType)
  auto forestType = mlir::decisionforest::TreeEnsembleType::get(doubleType, 1, doubleType /*HACK type doesn't matter for this test*/,
                                                                mlir::decisionforest::ReductionType::kAdd, treeType);
  
  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(GetGlobalJSONNameForTests());
  mlir::decisionforest::PersistDecisionForest(forest, forestType);
  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(GetGlobalJSONNameForTests());
  mlir::decisionforest::ForestJSONReader::GetInstance().ParseJSONFile();
  
  std::vector<DoubleInt32Tile> serializedTree(std::pow(2, 3) - 1); //Depth of the tree is 3, so this is the size of the dense array

  // InitializeBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth, int32_t indexBitWidth, std::vector<int32_t>& treeOffsets)
  std::vector<int32_t> offsets(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(serializedTree.data(), 1, 64, 32, offsets);
  Test_ASSERT(expectedArray == serializedTree);
  Test_ASSERT(offsets[0] == 0);

  std::vector<int64_t> offsetVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsetVec.data(), 1, 64, 32);
  Test_ASSERT(offsetVec[0] == 0);

  std::vector<int64_t> lengthVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengthVec.data(), 1, 64, 32);
  Test_ASSERT(lengthVec[0] == 7);

  mlir::decisionforest::ClearPersistedForest();

  return true;
}

bool Test_BufferInitializationWithTwoTrees(TestArgs_t& args) {
  return Test_BufferInitialization_TwoTrees<double, int32_t>(args);
}

bool Test_BufferInitializationWithTwoTrees_Int16(TestArgs_t& args) {
  return Test_BufferInitialization_TwoTrees<double, int16_t>(args);
}

bool Test_BufferInitializationWithTwoTrees_Int8(TestArgs_t& args) {
  return Test_BufferInitialization_TwoTrees<double, int8_t>(args);
}

bool Test_BufferInitializationWithTwoTrees_Float(TestArgs_t& args) {
  return Test_BufferInitialization_TwoTrees<float, int32_t>(args);
}

bool Test_BufferInitializationWithTwoTrees_FloatInt16(TestArgs_t& args) {
  return Test_BufferInitialization_TwoTrees<float, int16_t>(args);
}

bool Test_BufferInitializationWithTwoTrees_FloatInt8(TestArgs_t& args) {
  return Test_BufferInitialization_TwoTrees<float, int8_t>(args);
}

void OneTreeAtATimeSchedule(decisionforest::Schedule* schedule) {
  auto& batchIndexVar = schedule->GetBatchIndex();
  auto& treeIndexVar = schedule->GetTreeIndex();
  schedule->Reorder(std::vector<mlir::decisionforest::IndexVariable*>{ &treeIndexVar, &batchIndexVar });
}

void OneTreeAtATimeUnrolledSchedule(decisionforest::Schedule* schedule) {
  auto& batchIndexVar = schedule->GetBatchIndex();
  auto& treeIndexVar = schedule->GetTreeIndex();
  schedule->Reorder(std::vector<mlir::decisionforest::IndexVariable*>{ &treeIndexVar, &batchIndexVar });
  schedule->Unroll(treeIndexVar);
}

void UnrollTreeLoop(decisionforest::Schedule* schedule) {
  auto& treeIndexVar = schedule->GetTreeIndex();
  schedule->Unroll(treeIndexVar);
}

// IR Tests
bool Test_ForestCodeGen_BatchSize1(TestArgs_t& args, ForestConstructor_t forestConstructor, std::vector< std::vector<double> >& inputData,
                                   int32_t childIndexBitWidth=1, ScheduleManipulator_t scheduleManipulator=nullptr) {
  FixedTreeIRConstructor<> irConstructor(args.context, 1, forestConstructor);
  irConstructor.Parse();
  // If sparse representation is turned on, then child index bit width should be passed
  assert (!mlir::decisionforest::UseSparseTreeRepresentation || childIndexBitWidth!=1 );
  irConstructor.SetChildIndexBitWidth(childIndexBitWidth);
  auto module = irConstructor.GetEvaluationFunction();

  if (scheduleManipulator) {
    auto schedule = irConstructor.GetSchedule();
    scheduleManipulator(schedule);
  }

  // module->dump();
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(args.context, module);
  // module->dump();
  mlir::decisionforest::LowerEnsembleToMemrefs(args.context, module);
  // module->dump();
  mlir::decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  // module->dump();
  mlir::decisionforest::LowerToLLVM(args.context, module);
  // module->dump();
  // mlir::decisionforest::dumpLLVMIR(module);
  decisionforest::InferenceRunner inferenceRunner(irConstructor.GetModelGlobalsJSONFilePath(), module, 1, 64, 32);
  
  // inferenceRunner.PrintLengthsArray();
  // inferenceRunner.PrintOffsetsArray();
  
  for(auto& row : inputData) {
    double result = -1;
    inferenceRunner.RunInference<double, double>(row.data(), &result, row.size(), 1);
    double expectedResult = irConstructor.GetForest().Predict(row);
    Test_ASSERT(FPEqual(result, expectedResult));
  }
  return true;
}

bool Test_ForestCodeGen_VariableBatchSize(TestArgs_t& args, ForestConstructor_t forestConstructor, 
                                          int64_t batchSize, std::vector< std::vector<double> >& inputData, int32_t childIndexBitWidth=1,
                                          ScheduleManipulator_t scheduleManipulator=nullptr) {
  FixedTreeIRConstructor<> irConstructor(args.context, batchSize, forestConstructor);
  irConstructor.Parse();
  irConstructor.SetChildIndexBitWidth(childIndexBitWidth);
  auto module = irConstructor.GetEvaluationFunction();

  if (scheduleManipulator) {
    auto schedule = irConstructor.GetSchedule();
    scheduleManipulator(schedule);
  }
  
  // module->dump();
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(args.context, module);
  // module->dump();
  mlir::decisionforest::LowerEnsembleToMemrefs(args.context, module);
  mlir::decisionforest::ConvertNodeTypeToIndexType(args.context, module);
  // module->dump();
  mlir::decisionforest::LowerToLLVM(args.context, module);
  // module->dump();
  // mlir::decisionforest::dumpLLVMIR(module);
  decisionforest::InferenceRunner inferenceRunner(irConstructor.GetModelGlobalsJSONFilePath(), module, 1, 64, 32);
  
  // inferenceRunner.PrintLengthsArray();
  // inferenceRunner.PrintOffsetsArray();
  
  for(auto& batch : inputData) {
    assert (batch.size() % batchSize == 0);
    size_t rowSize = batch.size()/batchSize;
    std::vector<double> result(batchSize, -1);
    inferenceRunner.RunInference<double, double>(batch.data(), result.data(), batch.size()/batchSize, batchSize);
    for(int64_t rowIdx=0 ; rowIdx<batchSize ; ++rowIdx) {
      std::vector<double> row(batch.begin() + rowIdx*rowSize, batch.begin() + (rowIdx+1)*rowSize);
      double expectedResult = irConstructor.GetForest().Predict(row);
      Test_ASSERT(FPEqual(result[rowIdx], expectedResult));
    }
  }
  return true;
}

std::vector<std::vector<double>> GetBatchSize1Data() {
  std::vector<double> inputData1 = {0.1, 0.2, 0.5, 0.3, 0.25};
  std::vector<double> inputData2 = {0.1, 0.2, 0.6, 0.3, 0.25};
  std::vector<std::vector<double>> data = {inputData1, inputData2};
  return data;
}

std::vector<std::vector<double>> GetBatchSize2Data() {
  std::vector<double> inputData1 = {0.1, 0.2, 0.5, 0.3, 0.25,
                                    0.1, 0.2, 0.6, 0.3, 0.25};
  std::vector<std::vector<double>> data = { inputData1 };
  return data;
}

bool Test_CodeGeneration_LeftHeavy_BatchSize1(TestArgs_t& args) {
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddLeftHeavyTree<DoubleInt32Tile>, data);
}

bool Test_CodeGeneration_RightHeavy_BatchSize1(TestArgs_t& args) {
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddRightHeavyTree<DoubleInt32Tile>, data);
}

bool Test_CodeGeneration_RightAndLeftHeavy_BatchSize1(TestArgs_t& args) {
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, data);
}

bool Test_CodeGeneration_LeftHeavy_BatchSize2(TestArgs_t& args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(args, AddLeftHeavyTree<DoubleInt32Tile>, 2, data);
}

bool Test_CodeGeneration_RightHeavy_BatchSize2(TestArgs_t& args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(args, AddRightHeavyTree<DoubleInt32Tile>, 2, data);
}

bool Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2(TestArgs_t& args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 2, data);
}

// ===----------------------------------------=== //
// Basic non-trivial schedule code gen tests
// ===----------------------------------------=== //

bool Test_CodeGeneration_LeftHeavy_BatchSize2_XGBoostSchedule(TestArgs_t& args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(args, AddLeftHeavyTree<DoubleInt32Tile>, 2, data, 1, OneTreeAtATimeSchedule);
}

bool Test_CodeGeneration_RightHeavy_BatchSize2_XGBoostSchedule(TestArgs_t& args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(args, AddRightHeavyTree<DoubleInt32Tile>, 2, data, 1, OneTreeAtATimeSchedule);
}

bool Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2_XGBoostSchedule(TestArgs_t& args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 2, data, 1, OneTreeAtATimeSchedule);
}

bool Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2_UnrollTreeLoop(TestArgs_t& args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 2, data, 1, UnrollTreeLoop);
}

// ===----------------------------------------=== //
// Basic sparse code gen tests
// ===----------------------------------------=== //

bool Test_SparseCodeGeneration_LeftHeavy_BatchSize1_I32ChildIdx(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddLeftHeavyTree<DoubleInt32Tile>, data, 32);
}

bool Test_SparseCodeGeneration_RightHeavy_BatchSize1_I32ChildIdx(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddRightHeavyTree<DoubleInt32Tile>, data, 32);
}

bool Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize1_I32ChildIdx(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, data, 32);
}

bool Test_SparseCodeGeneration_LeftHeavy_BatchSize2_I32ChildIdx(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(args, AddLeftHeavyTree<DoubleInt32Tile>, 2, data, 32);
}

bool Test_SparseCodeGeneration_RightHeavy_BatchSize2_I32ChildIdx(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(args, AddRightHeavyTree<DoubleInt32Tile>, 2, data, 32);
}

bool Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize2_I32ChildIdx(TestArgs_t& args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 2, data, 32);
}

// Tests for Tiled Buffer Initialization
template<typename ThresholdType, typename IndexType>
bool Test_BufferInit_SingleTree_Tiled(TestArgs_t& args, ForestConstructor_t forestConstructor, std::vector<int32_t>& tileIDs) {
  using VectorTileType = NumericalVectorTileType_Packed<ThresholdType, IndexType, 3>;
  auto& context = args.context;
  mlir::OpBuilder builder(&context);
  mlir::decisionforest::DecisionForest<> forest;
  forestConstructor(forest);

  // Construct basic tree types for the tree
  // (Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type thresholdType, Type featureIndexType
  auto thresholdType = TreeBeard::GetMLIRType(ThresholdType(), builder);
  auto indexType = TreeBeard::GetMLIRType(IndexType(), builder);
  
  int32_t tileSize = 3;
  decisionforest::TreeTilingDescriptor tilingDescriptor(tileSize /*tile size*/, 4 /*num tiles*/, tileIDs, decisionforest::TilingType::kRegular);
  forest.GetTree(0).SetTilingDescriptor(tilingDescriptor);

  auto treeType = mlir::decisionforest::TreeType::get(thresholdType, tilingDescriptor.MaxTileSize(), thresholdType, indexType);
  //(Type resultType, size_t numTrees, Type rowType, ReductionType reductionType, Type treeType)
  std::vector<Type> treeTypes = {treeType};
  auto forestType = mlir::decisionforest::TreeEnsembleType::get(thresholdType, 1, thresholdType /*HACK type doesn't matter for this test*/,
                                                                mlir::decisionforest::ReductionType::kAdd, treeTypes);
  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(GetGlobalJSONNameForTests());
  mlir::decisionforest::PersistDecisionForest(forest, forestType);
  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(GetGlobalJSONNameForTests());
  mlir::decisionforest::ForestJSONReader::GetInstance().ParseJSONFile();

  mlir::decisionforest::TiledTree tiledTree(forest.GetTree(0));
  auto numTiles = tiledTree.GetNumberOfTiles();
  std::vector<VectorTileType> serializedTree(numTiles);
  auto thresholds = tiledTree.SerializeThresholds();
  auto featureIndices = tiledTree.SerializeFeatureIndices();
  auto tileShapeIDs = tiledTree.SerializeTileShapeIDs();

  std::vector<int32_t> offsets(1, -1);
  int32_t thresholdSize = sizeof(ThresholdType)*8;
  int32_t indexSize = sizeof(IndexType)*8;
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(serializedTree.data(), tileSize, thresholdSize, indexSize, offsets);
  for(int32_t i=0 ; i<numTiles ; ++i) {
    for (int32_t j=0 ; j<tileSize ; ++j) {
      Test_ASSERT(FPEqual(serializedTree[i].threshold[j], thresholds[i*tileSize + j]));
      Test_ASSERT(serializedTree[i].index[j] == featureIndices[i*tileSize + j]);
    }
    // std::cout << tileShapeIDs[i] << std::endl;
    Test_ASSERT(tileShapeIDs[i] == serializedTree[i].tileShapeID);
  }
  Test_ASSERT(offsets[0] == 0);
  
  std::vector<int64_t> offsetVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(offsetVec.data(), tileSize, thresholdSize, indexSize);
  Test_ASSERT(offsetVec[0] == 0);
  
  std::vector<int64_t> lengthVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(lengthVec.data(), tileSize, thresholdSize, indexSize);
  Test_ASSERT(lengthVec[0] == numTiles);

  mlir::decisionforest::ClearPersistedForest();
  // std::cout << "**********\n";
  return true;
}

bool Test_BufferInitializationWithOneTree_RightHeavy_Tiled(TestArgs_t& args) {
  using TileType = NumericalTileType_Packed<double, int32_t>;
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_BufferInit_SingleTree_Tiled<double, int32_t>(args, AddRightHeavyTree<TileType>, tileIDs);
}

bool Test_BufferInitializationWithOneTree_LeftHeavy_Tiled(TestArgs_t& args) {
  using TileType = NumericalTileType_Packed<double, int32_t>;
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 3 }; // The root and one of its children are in one tile and all leaves are in separate tiles
  return Test_BufferInit_SingleTree_Tiled<double, int32_t>(args, AddLeftHeavyTree<TileType>, tileIDs);
}

bool Test_BufferInitializationWithOneTree_Balanced_Tiled(TestArgs_t& args) {
  using TileType = NumericalTileType_Packed<double, int32_t>;
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 0, 3, 4 };
  return Test_BufferInit_SingleTree_Tiled<double, int32_t>(args, AddBalancedTree<TileType>, tileIDs);
}

// ===-------------------------------------------------------------=== //
// Tiled Tree Tests
// ===-------------------------------------------------------------=== //
bool CheckAllLeavesAreAtSameDepth(mlir::decisionforest::TiledTree* tiledTree) {
  auto depth = tiledTree->GetTreeDepth();
  for (int32_t i=0 ; i<(int32_t)tiledTree->NumTiles() ; ++i) {
    auto& tile = tiledTree->GetTile(i);
    if (!tile.IsLeafTile())
      continue;
    auto tilePtr = &tile;
    int32_t leafDepth = 1;
    while (tilePtr->GetParent() != decisionforest::DecisionTree<>::INVALID_NODE_INDEX) {
      tilePtr = &(tiledTree->GetTile(tilePtr->GetParent()));
      ++leafDepth;
    }
    Test_ASSERT(leafDepth == depth);
  }
  return true;  
}

bool Test_PadTiledTree_BalancedTree_TileSize2(TestArgs_t& args) {
  decisionforest::DecisionTree<> decisionTree;
  InitializeBalancedTree(decisionTree);
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 5, 3, 4 };
  
  decisionforest::TreeTilingDescriptor tilingDescriptor(2, 6, tileIDs, decisionforest::TilingType::kRegular);
  decisionTree.SetTilingDescriptor(tilingDescriptor);

  auto tiledTree = decisionTree.GetTiledTree();
  tiledTree->MakeAllLeavesSameDepth();
  // std::string dotFile = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/temp/tiledTree.dot";
  // tiledTree->WriteDOTFile(dotFile);
  return CheckAllLeavesAreAtSameDepth(tiledTree);
}

bool Test_PadTiledTree_BalancedTree_TileSize2_2(TestArgs_t& args) {
  decisionforest::DecisionTree<> decisionTree;
  InitializeBalancedTree(decisionTree);
  std::vector<int32_t> tileIDs = { 0, 5, 1, 2, 0, 3, 4 };
  
  decisionforest::TreeTilingDescriptor tilingDescriptor(2, 6, tileIDs, decisionforest::TilingType::kRegular);
  decisionTree.SetTilingDescriptor(tilingDescriptor);

  auto tiledTree = decisionTree.GetTiledTree();
  tiledTree->MakeAllLeavesSameDepth();
  // std::string dotFile = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/temp/tiledTree.dot";
  // tiledTree->WriteDOTFile(dotFile);
  return CheckAllLeavesAreAtSameDepth(tiledTree);
}

bool Test_PadTiledTree_BalancedTree_TileSize3(TestArgs_t& args) {
  decisionforest::DecisionTree<> decisionTree;
  InitializeBalancedTree(decisionTree);
  std::vector<int32_t> tileIDs = { 0, 0, 1, 2, 0, 3, 4 };
  
  decisionforest::TreeTilingDescriptor tilingDescriptor(3, 5, tileIDs, decisionforest::TilingType::kRegular);
  decisionTree.SetTilingDescriptor(tilingDescriptor);

  auto tiledTree = decisionTree.GetTiledTree();
  tiledTree->MakeAllLeavesSameDepth();
  // std::string dotFile = "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/temp/tiledTree.dot";
  // tiledTree->WriteDOTFile(dotFile);
  return CheckAllLeavesAreAtSameDepth(tiledTree);
}

#define RUN_ALL_TESTS

#ifdef RUN_ALL_TESTS
TestDescriptor testList[] = {
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_LeftHeavy),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_Int16),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_Int8),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_Float),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_FloatInt16),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_FloatInt8),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy),
  TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees),
  TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees_Int16),
  TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees_Int8),
  TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees_Float),
  TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees_FloatInt16),
  TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees_FloatInt8),
  TEST_LIST_ENTRY(Test_CodeGeneration_LeftHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_CodeGeneration_RightHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_CodeGeneration_RightAndLeftHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_CodeGeneration_LeftHeavy_BatchSize2),
  TEST_LIST_ENTRY(Test_CodeGeneration_RightHeavy_BatchSize2),
  TEST_LIST_ENTRY(Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2),
  TEST_LIST_ENTRY(Test_LoadTileFeatureIndicesOp_DoubleInt32_TileSize1),
  TEST_LIST_ENTRY(Test_LoadTileThresholdOp_DoubleInt32_TileSize1),
  TEST_LIST_ENTRY(Test_LoadTileThresholdOp_Subview_DoubleInt32_TileSize1),
  TEST_LIST_ENTRY(Test_LoadTileFeatureIndicesOp_Subview_DoubleInt32_TileSize1),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_1Tree_BatchSize4),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_1Tree_BatchSize2),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_1Tree_BatchSize1),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_2Trees_BatchSize1),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_2Trees_BatchSize2),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_2Trees_BatchSize4),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_4Trees_BatchSize1),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_4Trees_BatchSize2),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_4Trees_BatchSize4),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_1Tree_BatchSize1_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_1Tree_BatchSize2_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_1Tree_BatchSize4_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_2Trees_BatchSize1_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_2Trees_BatchSize2_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_2Trees_BatchSize4_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_4Trees_BatchSize1_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_4Trees_BatchSize2_Float),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_4Trees_BatchSize4_Float),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_Tiled),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_LeftHeavy_Tiled),
  TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_Balanced_Tiled),
  TEST_LIST_ENTRY(Test_TiledCodeGeneration_LeftHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_TiledCodeGeneration_RightHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_TiledCodeGeneration_BalancedTree_BatchSize1),
  TEST_LIST_ENTRY(Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_TiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape),
  TEST_LIST_ENTRY(Test_TiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape),
  TEST_LIST_ENTRY(Test_TiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape),
  TEST_LIST_ENTRY(Test_TiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape),
  TEST_LIST_ENTRY(Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize),
  TEST_LIST_ENTRY(Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize),
  TEST_LIST_ENTRY(Test_ModelInit_LeftHeavy),
  TEST_LIST_ENTRY(Test_ModelInit_RightHeavy),
  TEST_LIST_ENTRY(Test_ModelInit_RightAndLeftHeavy),
  TEST_LIST_ENTRY(Test_ModelInit_Balanced),
  TEST_LIST_ENTRY(Test_ModelInit_LeftHeavy_Int8TileShape),
  TEST_LIST_ENTRY(Test_ModelInit_LeftHeavy_Int16TileShape),
  TEST_LIST_ENTRY(Test_ModelInit_RightHeavy_Int8TileShape),
  TEST_LIST_ENTRY(Test_ModelInit_RightHeavy_Int16TileShape),
  TEST_LIST_ENTRY(Test_ModelInit_Balanced_Int8TileShape),
  TEST_LIST_ENTRY(Test_ModelInit_Balanced_Int16TileShape),
  TEST_LIST_ENTRY(Test_ModelInit_RightAndLeftHeavy_Int8TileShape),
  TEST_LIST_ENTRY(Test_ModelInit_RightAndLeftHeavy_Int16TileShape),
  TEST_LIST_ENTRY(Test_UniformTiling_LeftHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_UniformTiling_LeftHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_UniformTiling_RightHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_UniformTiling_Balanced_BatchSize1),
  TEST_LIST_ENTRY(Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_UniformTiling_LeftHeavy_BatchSize1_Int8TileShape),
  TEST_LIST_ENTRY(Test_UniformTiling_RightHeavy_BatchSize1_Int8TileShape),
  TEST_LIST_ENTRY(Test_UniformTiling_Balanced_BatchSize1_Int8TileShape),
  TEST_LIST_ENTRY(Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1_Int8TileShape),
  TEST_LIST_ENTRY(Test_UniformTiling_LeftHeavy_BatchSize1_Int16TileShape),
  TEST_LIST_ENTRY(Test_UniformTiling_RightHeavy_BatchSize1_Int16TileShape),
  TEST_LIST_ENTRY(Test_UniformTiling_Balanced_BatchSize1_Int16TileShape),
  TEST_LIST_ENTRY(Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1_Int16TileShape),
  TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize1),
  TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize1),
  TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize1),
  TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize2),
  TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize2),
  TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize2),
  TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4),
  TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4),
  TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4),
  TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4_Int8TileShape),
  TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4_Int16TileShape),
  TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4_Int8TileShape),
  TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4_Int16TileShape),
  TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4_Int8TileShape),
  TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4_Int16TileShape),
  TEST_LIST_ENTRY(Test_Scalar_Abalone),
  TEST_LIST_ENTRY(Test_TileSize2_Abalone),
  TEST_LIST_ENTRY(Test_TileSize3_Abalone),
  TEST_LIST_ENTRY(Test_TileSize4_Abalone),
  TEST_LIST_ENTRY(Test_TileSize8_Abalone),
  TEST_LIST_ENTRY(Test_Scalar_Airline),
  TEST_LIST_ENTRY(Test_TileSize2_Airline),
  TEST_LIST_ENTRY(Test_TileSize3_Airline),
  TEST_LIST_ENTRY(Test_TileSize4_Airline),
  TEST_LIST_ENTRY(Test_TileSize8_Airline),
  TEST_LIST_ENTRY(Test_Scalar_AirlineOHE),
  TEST_LIST_ENTRY(Test_TileSize2_AirlineOHE),
  TEST_LIST_ENTRY(Test_TileSize3_AirlineOHE),
  TEST_LIST_ENTRY(Test_TileSize4_AirlineOHE),
  TEST_LIST_ENTRY(Test_TileSize8_AirlineOHE),
  TEST_LIST_ENTRY(Test_Scalar_Bosch), 
  TEST_LIST_ENTRY(Test_TileSize2_Bosch),
  TEST_LIST_ENTRY(Test_TileSize3_Bosch),
  TEST_LIST_ENTRY(Test_TileSize4_Bosch),
  TEST_LIST_ENTRY(Test_TileSize8_Bosch),
  TEST_LIST_ENTRY(Test_Scalar_Epsilon),
  TEST_LIST_ENTRY(Test_TileSize2_Epsilon),
  TEST_LIST_ENTRY(Test_TileSize3_Epsilon),
  TEST_LIST_ENTRY(Test_TileSize4_Epsilon),
  TEST_LIST_ENTRY(Test_TileSize8_Epsilon),
  TEST_LIST_ENTRY(Test_Scalar_Higgs),
  TEST_LIST_ENTRY(Test_TileSize2_Higgs),
  TEST_LIST_ENTRY(Test_TileSize3_Higgs),
  TEST_LIST_ENTRY(Test_TileSize4_Higgs),
  TEST_LIST_ENTRY(Test_TileSize8_Higgs),
  TEST_LIST_ENTRY(Test_Scalar_Year),
  TEST_LIST_ENTRY(Test_TileSize2_Year),
  TEST_LIST_ENTRY(Test_TileSize3_Year),
  TEST_LIST_ENTRY(Test_TileSize4_Year),
  TEST_LIST_ENTRY(Test_TileSize8_Year),
  TEST_LIST_ENTRY(Test_TileSize1_CovType_Int8Type),
  TEST_LIST_ENTRY(Test_TileSize2_CovType_Int8Type),
  TEST_LIST_ENTRY(Test_TileSize3_CovType_Int8Type),
  TEST_LIST_ENTRY(Test_TileSize4_CovType_Int8Type),
  TEST_LIST_ENTRY(Test_TileSize8_CovType_Int8Type),

  // Sparse tests
  TEST_LIST_ENTRY(Test_SparseCodeGeneration_LeftHeavy_BatchSize1_I32ChildIdx),
  TEST_LIST_ENTRY(Test_SparseCodeGeneration_RightHeavy_BatchSize1_I32ChildIdx),
  TEST_LIST_ENTRY(Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize1_I32ChildIdx),
  TEST_LIST_ENTRY(Test_SparseCodeGeneration_LeftHeavy_BatchSize2_I32ChildIdx),
  TEST_LIST_ENTRY(Test_SparseCodeGeneration_RightHeavy_BatchSize2_I32ChildIdx),
  TEST_LIST_ENTRY(Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize2_I32ChildIdx),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize1),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize2),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize4),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize1),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize2),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize4),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize1),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize2),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize4),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize1_Float),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize2_Float),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize4_Float),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize1_Float),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize2_Float),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize4_Float),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize1_Float),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize2_Float),
  TEST_LIST_ENTRY(Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize4_Float),
  TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape),
  TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape),
  TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize),
  TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize),
  TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1),
  TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape),
  TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape),
  TEST_LIST_ENTRY(Test_SparseUniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4),
  TEST_LIST_ENTRY(Test_SparseUniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4),
  // XGBoost Benchmarks Sparse Tests
  TEST_LIST_ENTRY(Test_SparseScalar_Abalone),
  TEST_LIST_ENTRY(Test_SparseTileSize2_Abalone),
  TEST_LIST_ENTRY(Test_SparseTileSize3_Abalone),
  TEST_LIST_ENTRY(Test_SparseTileSize4_Abalone),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Abalone),
  TEST_LIST_ENTRY(Test_SparseScalar_Airline),
  TEST_LIST_ENTRY(Test_SparseTileSize2_Airline),
  TEST_LIST_ENTRY(Test_SparseTileSize3_Airline),
  TEST_LIST_ENTRY(Test_SparseTileSize4_Airline),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Airline),
  TEST_LIST_ENTRY(Test_SparseScalar_AirlineOHE),
  TEST_LIST_ENTRY(Test_SparseTileSize2_AirlineOHE),
  TEST_LIST_ENTRY(Test_SparseTileSize3_AirlineOHE),
  TEST_LIST_ENTRY(Test_SparseTileSize4_AirlineOHE),
  TEST_LIST_ENTRY(Test_SparseTileSize8_AirlineOHE),
  TEST_LIST_ENTRY(Test_SparseScalar_Bosch),
  TEST_LIST_ENTRY(Test_SparseTileSize2_Bosch),
  TEST_LIST_ENTRY(Test_SparseTileSize3_Bosch),
  TEST_LIST_ENTRY(Test_SparseTileSize4_Bosch),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Bosch),
  TEST_LIST_ENTRY(Test_SparseTileSize8_CovType_Int8Type),
  TEST_LIST_ENTRY(Test_SparseScalar_CovType_Int8Type),
  TEST_LIST_ENTRY(Test_SparseScalar_Epsilon),
  TEST_LIST_ENTRY(Test_SparseTileSize2_Epsilon),
  TEST_LIST_ENTRY(Test_SparseTileSize3_Epsilon),
  TEST_LIST_ENTRY(Test_SparseTileSize4_Epsilon),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Epsilon),
  TEST_LIST_ENTRY(Test_SparseScalar_Higgs),
  TEST_LIST_ENTRY(Test_SparseTileSize2_Higgs),
  TEST_LIST_ENTRY(Test_SparseTileSize3_Higgs),
  TEST_LIST_ENTRY(Test_SparseTileSize4_Higgs),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Higgs),
  TEST_LIST_ENTRY(Test_SparseScalar_Year),
  TEST_LIST_ENTRY(Test_SparseTileSize2_Year),
  TEST_LIST_ENTRY(Test_SparseTileSize3_Year),
  TEST_LIST_ENTRY(Test_SparseTileSize4_Year),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Year),
  TEST_LIST_ENTRY(Test_TileSize8_Abalone_TestInputs),
  TEST_LIST_ENTRY(Test_TileSize8_Airline_TestInputs),
  TEST_LIST_ENTRY(Test_TileSize8_AirlineOHE_TestInputs),
  TEST_LIST_ENTRY(Test_TileSize8_Epsilon_TestInputs),
  TEST_LIST_ENTRY(Test_TileSize8_Higgs_TestInputs),
  TEST_LIST_ENTRY(Test_TileSize8_Year_TestInputs),
  TEST_LIST_ENTRY(Test_TileSize8_CovType_TestInputs),

  // Non-trivial schedule array representation tests
  TEST_LIST_ENTRY(Test_CodeGeneration_LeftHeavy_BatchSize2_XGBoostSchedule),
  TEST_LIST_ENTRY(Test_CodeGeneration_RightHeavy_BatchSize2_XGBoostSchedule),
  TEST_LIST_ENTRY(Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2_XGBoostSchedule),
  TEST_LIST_ENTRY(Test_Scalar_Abalone_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize2_Abalone_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize3_Abalone_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize4_Abalone_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize8_Abalone_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_Scalar_Airline_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize2_Airline_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize3_Airline_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize4_Airline_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize8_Airline_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_Scalar_AirlineOHE_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize2_AirlineOHE_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize3_AirlineOHE_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize4_AirlineOHE_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize8_AirlineOHE_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_Scalar_Bosch_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize2_Bosch_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize3_Bosch_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize4_Bosch_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize8_Bosch_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_Scalar_CovType_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize8_CovType_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_Scalar_Epsilon_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize2_Epsilon_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize3_Epsilon_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize4_Epsilon_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize8_Epsilon_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_Scalar_Higgs_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize2_Higgs_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize3_Higgs_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize4_Higgs_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize8_Higgs_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_Scalar_Year_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize2_Year_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize3_Year_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize4_Year_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_TileSize8_Year_OneTreeAtATimeSchedule),

  TEST_LIST_ENTRY(Test_TileSize8_Abalone_TestInputs_TiledSchedule),
  TEST_LIST_ENTRY(Test_TileSize8_AirlineOHE_TestInputs_TiledSchedule),
  TEST_LIST_ENTRY(Test_TileSize8_Airline_TestInputs_TiledSchedule),
  TEST_LIST_ENTRY(Test_TileSize8_Epsilon_TestInputs_TiledSchedule),
  TEST_LIST_ENTRY(Test_TileSize8_Higgs_TestInputs_TiledSchedule),
  TEST_LIST_ENTRY(Test_TileSize8_Year_TestInputs_TiledSchedule),

  // Sparse code gen tests with loops interchanged
  TEST_LIST_ENTRY(Test_SparseScalar_Abalone_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Abalone_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_SparseScalar_Airline_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Airline_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_SparseScalar_AirlineOHE_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_SparseTileSize8_AirlineOHE_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_SparseScalar_Bosch_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Bosch_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_SparseScalar_CovType_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_SparseTileSize8_CovType_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_SparseScalar_Epsilon_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Epsilon_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_SparseScalar_Higgs_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Higgs_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_SparseScalar_Year_OneTreeAtATimeSchedule),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Year_OneTreeAtATimeSchedule),

  // Sparse code gen tests with loops tiled
  TEST_LIST_ENTRY(Test_SparseTileSize8_Abalone_TestInputs_TiledSchedule),
  TEST_LIST_ENTRY(Test_SparseTileSize8_AirlineOHE_TestInputs_TiledSchedule),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Airline_TestInputs_TiledSchedule),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Epsilon_TestInputs_TiledSchedule),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Higgs_TestInputs_TiledSchedule),
  TEST_LIST_ENTRY(Test_SparseTileSize8_Year_TestInputs_TiledSchedule),

  // Stats tests
  TEST_LIST_ENTRY(Test_AbaloneStatGenerationAndReading),
  TEST_LIST_ENTRY(Test_AirlineStatGenerationAndReading),
  TEST_LIST_ENTRY(Test_AirlineOHEStatGenerationAndReading),
  TEST_LIST_ENTRY(Test_CovtypeStatGenerationAndReading),
  TEST_LIST_ENTRY(Test_EpsilonStatGenerationAndReading),
  TEST_LIST_ENTRY(Test_HiggsStatGenerationAndReading),
  TEST_LIST_ENTRY(Test_YearStatGenerationAndReading),

  // Sparse Probabilistic Tiling Tests
  TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Abalone),
  TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Airline),
  TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_AirlineOHE),
  TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Covtype),
  TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Epsilon),
  TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Higgs),
  TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Year),

  // Tiled tree padding tests
  TEST_LIST_ENTRY(Test_PadTiledTree_BalancedTree_TileSize2),
  TEST_LIST_ENTRY(Test_PadTiledTree_BalancedTree_TileSize2_2),
  TEST_LIST_ENTRY(Test_PadTiledTree_BalancedTree_TileSize3),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_1Tree_BatchSize4_EqualDepth_TileSize8),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_2Trees_BatchSize4_EqualDepth_TileSize8),
  TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_4Trees_BatchSize4_EqualDepth_TileSize8),
};

#else // RUN_ALL_TESTS

TestDescriptor testList[] = {
  // TEST_LIST_ENTRY(Test_ProbabilisticTiling_TileSize8_Covtype),
  // TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Airline),
  // TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_AirlineOHE),
  // TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Epsilon),
  // TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Higgs),
  // TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Year),
  // TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Abalone),
  // TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Airline),
  // TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_AirlineOHE),
  // TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Epsilon),
  // TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Higgs),
  // TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Year),
  // TEST_LIST_ENTRY(Test_CovtypeStatGenerationAndReading),
  // TEST_LIST_ENTRY(Test_EpsilonStatGenerationAndReading),
  // TEST_LIST_ENTRY(Test_HiggsStatGenerationAndReading),
  // TEST_LIST_ENTRY(Test_YearStatGenerationAndReading),
  // TEST_LIST_ENTRY(Test_TileSize8_Higgs_TestInputs),
  // TEST_LIST_ENTRY(Test_TileSize8_Year_TestInputs),
  // TEST_LIST_ENTRY(Test_SparseUniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4),
  // TEST_LIST_ENTRY(Test_SparseUniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4),
  // TEST_LIST_ENTRY(Test_SparseUniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4),
  // TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape),
  // TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape),
  // TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize),
  // TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1),
  // TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize),
  // TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1),
  // TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape),
  // TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape),
  // TEST_LIST_ENTRY(Test_SparseCodeGeneration_RightHeavy_BatchSize1_I32ChildIdx),
  // TEST_LIST_ENTRY(Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize1_I32ChildIdx),
  // TEST_LIST_ENTRY(Test_SparseCodeGeneration_LeftHeavy_BatchSize2_I32ChildIdx),
  // TEST_LIST_ENTRY(Test_SparseCodeGeneration_RightHeavy_BatchSize2_I32ChildIdx),
  // TEST_LIST_ENTRY(Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize2_I32ChildIdx),
  // TEST_LIST_ENTRY(Test_TileSize8_Airline),
  // TEST_LIST_ENTRY(Test_TileSize8_AirlineOHE),
  // TEST_LIST_ENTRY(Test_TileSize8_Bosch),
  // TEST_LIST_ENTRY(Test_TileSize8_Epsilon),
  // TEST_LIST_ENTRY(Test_TileSize8_Higgs),
  // TEST_LIST_ENTRY(Test_TileSize8_Year),
  // TEST_LIST_ENTRY(Test_TileSize3_Abalone),
  // TEST_LIST_ENTRY(Test_TileSize4_Abalone),
};
#endif // RUN_ALL_TESTS

const size_t numTests = sizeof(testList) / sizeof(testList[0]);

// void PrintExceptionInfo(std::exception_ptr eptr) {
// 	try {
// 		if (eptr) {
// 			std::rethrow_exception(eptr);
// 		}
// 	}
// 	catch (const std::exception& e) {
// 		std::cout << "\"" << e.what() << "\"\n";
// 	}
// }

const std::string reset("\033[0m");
const std::string red("\033[0;31m");
const std::string boldRed("\033[1;31m");
const std::string green("\033[0;32m");
const std::string boldGreen("\033[1;32m");
const std::string blue("\033[0;34m");
const std::string boldBlue("\033[1;34m");
const std::string white("\033[0;37m");
const std::string underline("\033[4m");

bool RunTest(TestDescriptor test, TestArgs_t& args) {
	std::string errStr;
	std::cout << white << "Running test " << blue << test.m_testName << reset << ".... ";
	bool pass = false;
  // try
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	{
		pass = test.m_testFunc(args);
	}
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	// catch s(...)
	// {
	// 	std::exception_ptr eptr = std::current_exception();
	// 	std::cout << "Crashed with exception ";
	// 	PrintExceptionInfo(eptr);
	// 	pass = false;
	// }
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << (pass ? green + "Passed" : red + "Failed") << white + " (Duration : " << duration << " ms)" << reset << std::endl;
	return pass;
}

void RunTests() {
 	bool overallPass = true;

  std::cout << "Running Treebeard Tests " << std::endl << std::endl;
  int32_t numPassed = 0;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for (size_t i = 0; i < numTests; ++i) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
    context.getOrLoadDialect<mlir::decisionforest::DecisionForestDialect>();
    context.getOrLoadDialect<mlir::StandardOpsDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::vector::VectorDialect>();
    context.getOrLoadDialect<mlir::math::MathDialect>();
    TestArgs_t args = { context };
    
    // Disable sparse code generation by default
    decisionforest::UseSparseTreeRepresentation = false;
    mlir::decisionforest::ForestJSONReader::GetInstance().SetChildIndexBitWidth(-1);
    
    bool pass = RunTest(testList[i], args);
    numPassed += pass ? 1 : 0;
    overallPass = overallPass && pass;
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto totalTime = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();

  std::cout << std::endl << boldBlue << underline << numPassed << "/" << numTests << reset << white << " tests passed.";
  std::cout << underline << (overallPass ? boldGreen + "\nTest Suite Passed." : boldRed + "\nTest Suite Failed.") << reset;
  std::cout << std::endl <<"Total time taken : " << totalTime << " seconds." << std::endl << std::endl;
}

} // test
} // TreeBeard