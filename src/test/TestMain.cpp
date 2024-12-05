#include "ExecutionHelpers.h"
#include "ForestTestUtils.h"
#include "GPUExecutionHelper.h"
#include "GPUSupportUtils.h"
#include "LowerReduceOps.h"
#include "ModelSerializers.h"
#include "Representations.h"
#include "TestUtilsCommon.h"
#include "TiledTree.h"
#include "TreeTilingUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "xgboostparser.h"
#include <chrono>
#include <sstream>
#include <vector>


using namespace mlir::decisionforest;

namespace TreeBeard {
namespace test {

// Codegen tests
bool Test_LoadTileFeatureIndicesOp_DoubleInt32_TileSize1(TestArgs_t &args);
bool Test_LoadTileThresholdOp_DoubleInt32_TileSize1(TestArgs_t &args);
bool Test_LoadTileThresholdOp_Subview_DoubleInt32_TileSize1(TestArgs_t &args);
bool Test_LoadTileFeatureIndicesOp_Subview_DoubleInt32_TileSize1(
    TestArgs_t &args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize1(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize8_TileSize2_4Pipelined(
    TestArgs_t &args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize8_TlieSize4_4Pipelined(
    TestArgs_t &args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize4_4Pipelined(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize2(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize4(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize1(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize2(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize4(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize1(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize2(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize4(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize1_Float(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize2_Float(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_1Tree_BatchSize4_Float(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize1_Float(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize2_Float(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize4_Float(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize1_Float(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize2_Float(TestArgs_t &args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize4_Float(TestArgs_t &args);

// Tiled Codegen tests
bool Test_TiledCodeGeneration_RightHeavy_BatchSize1(TestArgs_t &args);
bool Test_TiledCodeGeneration_LeftHeavy_BatchSize1(TestArgs_t &args);
bool Test_TiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape(
    TestArgs_t &args);
bool Test_TiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape(
    TestArgs_t &args);
bool Test_TiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape(
    TestArgs_t &args);
bool Test_TiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape(
    TestArgs_t &args);
bool Test_TiledCodeGeneration_BalancedTree_BatchSize1(TestArgs_t &args);
bool Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1(TestArgs_t &args);
bool Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize(
    TestArgs_t &args);
bool Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize(
    TestArgs_t &args);

// Tiled Init Tests
bool Test_ModelInit_LeftHeavy(TestArgs_t &args);
bool Test_ModelInit_RightHeavy(TestArgs_t &args);
bool Test_ModelInit_RightAndLeftHeavy(TestArgs_t &args);
bool Test_ModelInit_Balanced(TestArgs_t &args);
bool Test_ModelInit_LeftHeavy_Int8TileShape(TestArgs_t &args);
bool Test_ModelInit_LeftHeavy_Int16TileShape(TestArgs_t &args);
bool Test_ModelInit_RightHeavy_Int8TileShape(TestArgs_t &args);
bool Test_ModelInit_RightHeavy_Int16TileShape(TestArgs_t &args);
bool Test_ModelInit_Balanced_Int8TileShape(TestArgs_t &args);
bool Test_ModelInit_Balanced_Int16TileShape(TestArgs_t &args);
bool Test_ModelInit_RightAndLeftHeavy_Int8TileShape(TestArgs_t &args);
bool Test_ModelInit_RightAndLeftHeavy_Int16TileShape(TestArgs_t &args);

// Uniform Tiling Tests
bool Test_UniformTiling_LeftHeavy_BatchSize1(TestArgs_t &args);
bool Test_UniformTiling_RightHeavy_BatchSize1(TestArgs_t &args);
bool Test_UniformTiling_Balanced_BatchSize1(TestArgs_t &args);
bool Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1(TestArgs_t &args);
bool Test_UniformTiling_LeftHeavy_BatchSize1_Int8TileShape(TestArgs_t &args);
bool Test_UniformTiling_RightHeavy_BatchSize1_Int8TileShape(TestArgs_t &args);
bool Test_UniformTiling_Balanced_BatchSize1_Int8TileShape(TestArgs_t &args);
bool Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1_Int8TileShape(
    TestArgs_t &args);
bool Test_UniformTiling_LeftHeavy_BatchSize1_Int16TileShape(TestArgs_t &args);
bool Test_UniformTiling_RightHeavy_BatchSize1_Int16TileShape(TestArgs_t &args);
bool Test_UniformTiling_Balanced_BatchSize1_Int16TileShape(TestArgs_t &args);
bool Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1_Int16TileShape(
    TestArgs_t &args);

bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize1(TestArgs_t &args);
bool Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize1(TestArgs_t &args);
bool Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize1(TestArgs_t &args);
bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize2(TestArgs_t &args);
bool Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize2(TestArgs_t &args);
bool Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize2(TestArgs_t &args);
bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4(TestArgs_t &args);
bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4_Int8TileShape(
    TestArgs_t &args);
bool Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4_Int16TileShape(
    TestArgs_t &args);
bool Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4(TestArgs_t &args);
bool Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4_Int8TileShape(
    TestArgs_t &args);
bool Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4_Int16TileShape(
    TestArgs_t &args);
bool Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4(TestArgs_t &args);
bool Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4_Int8TileShape(
    TestArgs_t &args);
bool Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4_Int16TileShape(
    TestArgs_t &args);

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

bool Test_TileSize1_CovType_Int8Type(TestArgs_t &args);
bool Test_TileSize2_CovType_Int8Type(TestArgs_t &args);
bool Test_TileSize3_CovType_Int8Type(TestArgs_t &args);
bool Test_TileSize4_CovType_Int8Type(TestArgs_t &args);
bool Test_TileSize8_CovType_Int8Type(TestArgs_t &args);

bool Test_TileSize1_Letters_Int8Type(TestArgs_t &args);
bool Test_TileSize2_Letters_Int8Type(TestArgs_t &args);
bool Test_TileSize3_Letters_Int8Type(TestArgs_t &args);
bool Test_TileSize4_Letters_Int8Type(TestArgs_t &args);
bool Test_TileSize8_Letters_Int8Type(TestArgs_t &args);
bool Test_TileSize3_Letters_2Pipelined_Int8Type(TestArgs_t &args);
bool Test_TileSize4_Letters_3Pipelined_Int8Type(TestArgs_t &args);
bool Test_TileSize8_Letters_5Pipelined_Int8Type(TestArgs_t &args);

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

// Sparse XGBoost Scalar Tests
bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize1(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize2(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize4(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize1(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize2(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize4(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize1(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize2(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize4(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize1_Float(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize2_Float(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize4_Float(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize1_Float(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize2_Float(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize4_Float(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize1_Float(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize2_Float(TestArgs_t &args);
bool Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize4_Float(TestArgs_t &args);

// Sparse Tiled Code Gen
bool Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1(
    TestArgs_t &args);
bool Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize(
    TestArgs_t &args);
bool Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize(
    TestArgs_t &args);
bool Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1(TestArgs_t &args);
bool Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1(TestArgs_t &args);
bool Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape(
    TestArgs_t &args);
bool Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape(
    TestArgs_t &args);
bool Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape(
    TestArgs_t &args);
bool Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape(
    TestArgs_t &args);

// XGBoost Sparse Uniform Tiling Tests
bool Test_SparseUniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4(
    TestArgs_t &args);
bool Test_SparseUniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4(
    TestArgs_t &args);
bool Test_SparseUniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4(
    TestArgs_t &args);

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
bool Test_SparseTileSize8_Pipeline4_Airline(TestArgs_t &args);
bool Test_SparseScalar_AirlineOHE(TestArgs_t &args);
bool Test_SparseTileSize2_AirlineOHE(TestArgs_t &args);
bool Test_SparseTileSize3_AirlineOHE(TestArgs_t &args);
bool Test_SparseTileSize4_AirlineOHE(TestArgs_t &args);
bool Test_SparseTileSize8_AirlineOHE(TestArgs_t &args);
bool Test_SparseTileSize8_Pipelined4_AirlineOHE(TestArgs_t &args);
bool Test_SparseScalar_Bosch(TestArgs_t &args);
bool Test_SparseTileSize2_Bosch(TestArgs_t &args);
bool Test_SparseTileSize3_Bosch(TestArgs_t &args);
bool Test_SparseTileSize4_Bosch(TestArgs_t &args);
bool Test_SparseTileSize8_Bosch(TestArgs_t &args);
bool Test_SparseTileSize8_4Pipelined_Bosch(TestArgs_t &args);
bool Test_SparseScalar_CovType_Int8Type(TestArgs_t &args);
bool Test_SparseTileSize8_CovType_Int8Type(TestArgs_t &args);
bool Test_SparseScalar_Epsilon(TestArgs_t &args);
bool Test_SparseTileSize2_Epsilon(TestArgs_t &args);
bool Test_SparseTileSize3_Epsilon(TestArgs_t &args);
bool Test_SparseTileSize4_Epsilon(TestArgs_t &args);
bool Test_SparseTileSize8_Epsilon(TestArgs_t &args);
bool Test_SparseTileSize8_Pipelined_Epsilon(TestArgs_t &args);
bool Test_SparseScalar_Higgs(TestArgs_t &args);
bool Test_SparseTileSize2_Higgs(TestArgs_t &args);
bool Test_SparseTileSize3_Higgs(TestArgs_t &args);
bool Test_SparseTileSize4_Higgs(TestArgs_t &args);
bool Test_SparseTileSize8_Higgs(TestArgs_t &args);
bool Test_SparseTileSize8_Pipelined_Higgs(TestArgs_t &args);
bool Test_SparseScalar_Letters_Int8Type(TestArgs_t &args);
bool Test_SparseTileSize8_Letters_Int8Type(TestArgs_t &args);
bool Test_SparseScalar_Year(TestArgs_t &args);
bool Test_SparseTileSize2_Year(TestArgs_t &args);
bool Test_SparseTileSize3_Year(TestArgs_t &args);
bool Test_SparseTileSize4_Year(TestArgs_t &args);
bool Test_SparseTileSize8_Year(TestArgs_t &args);
bool Test_SparseTileSize8_Pipelined_Year(TestArgs_t &args);

// Tests for actual model inputs
bool Test_TileSize8_Abalone_TestInputs(TestArgs_t &args);
bool Test_TileSize8_Abalone_4Pipelined_TestInputs(TestArgs_t &args);
bool Test_TileSize8_Airline_TestInputs(TestArgs_t &args);
bool Test_TileSize8_AirlineOHE_TestInputs(TestArgs_t &args);
bool Test_TileSize8_Bosch_TestInputs(TestArgs_t &args);
bool Test_TileSize8_Epsilon_TestInputs(TestArgs_t &args);
bool Test_TileSize8_Higgs_TestInputs(TestArgs_t &args);
bool Test_TileSize8_Year_TestInputs(TestArgs_t &args);
bool Test_TileSize8_CovType_TestInputs(TestArgs_t &args);
bool Test_TileSize8_CovType_4Pipelined_TestInputs(TestArgs_t &args);

bool Test_TileSize8_Abalone_4PipelinedTrees_TestInputs(TestArgs_t &args);
bool Test_TileSize8_Abalone_PipelinedTreesPeeling_TestInputs(TestArgs_t &args);

// Tiled schedule test
bool Test_TileSize8_Abalone_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_TileSize8_AirlineOHE_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_TileSize8_Airline_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_TileSize8_Bosch_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_TileSize8_Epsilon_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_TileSize8_Higgs_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_TileSize8_Year_TestInputs_TiledSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_Letters_TiledSchedule(TestArgs_t &args);

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
bool Test_SparseScalar_Letters_OneTreeAtATimeSchedule(TestArgs_t &args);
bool Test_SparseTileSize8_Letters_OneTreeAtATimeSchedule(TestArgs_t &args);

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
bool Test_RandomXGBoostJSONs_1Tree_BatchSize4_EqualDepth_TileSize8(
    TestArgs_t &args);
bool Test_RandomXGBoostJSONs_2Trees_BatchSize4_EqualDepth_TileSize8(
    TestArgs_t &args);
bool Test_RandomXGBoostJSONs_4Trees_BatchSize4_EqualDepth_TileSize8(
    TestArgs_t &args);
bool Test_TileSize8_Abalone_TestInputs_MakeLeavesSameDepth(TestArgs_t &args);
bool Test_TileSize8_AirlineOHE_TestInputs_MakeLeavesSameDepth(TestArgs_t &args);
bool Test_TileSize8_Airline_TestInputs_MakeLeavesSameDepth(TestArgs_t &args);
bool Test_TileSize8_Bosch_TestInputs_MakeLeavesSameDepth(TestArgs_t &args);
bool Test_TileSize8_Epsilon_TestInputs_MakeLeavesSameDepth(TestArgs_t &args);
bool Test_TileSize8_Higgs_TestInputs_MakeLeavesSameDepth(TestArgs_t &args);
bool Test_TileSize8_Year_TestInputs_MakeLeavesSameDepth(TestArgs_t &args);
bool Test_TileSize8_CovType_TestInputs_MakeLeavesSameDepth(TestArgs_t &args);

bool Test_TileSize8_Abalone_TestInputs_ReorderTrees(TestArgs_t &args);

// Split schedule tests
bool Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1_SplitTreeLoop(
    TestArgs_t &args);
bool Test_TileSize8_Abalone_TestInputs_SplitTreeLoopSchedule(TestArgs_t &args);
bool Test_TileSize8_Abalone_TestInputs_SplitTreeLoopSchedule(TestArgs_t &args);
bool Test_TileSize8_AirlineOHE_TestInputs_SplitTreeLoopSchedule(
    TestArgs_t &args);
bool Test_TileSize8_Airline_TestInputs_SplitTreeLoopSchedule(TestArgs_t &args);
bool Test_TileSize8_Bosch_TestInputs_SplitTreeLoopSchedule(TestArgs_t &args);
bool Test_TileSize8_Epsilon_TestInputs_SplitTreeLoopSchedule(TestArgs_t &args);
bool Test_TileSize8_Higgs_TestInputs_SplitTreeLoopSchedule(TestArgs_t &args);
bool Test_TileSize8_Year_TestInputs_SplitTreeLoopSchedule(TestArgs_t &args);
bool Test_TileSize8_Abalone_TestInputs_SwapAndSplitTreeIndex(TestArgs_t &args);
bool Test_TileSize8_AirlineOHE_TestInputs_SwapAndSplitTreeIndex(
    TestArgs_t &args);
bool Test_TileSize8_Airline_TestInputs_SwapAndSplitTreeIndex(TestArgs_t &args);
bool Test_TileSize8_Bosch_TestInputs_SwapAndSplitTreeIndex(TestArgs_t &args);
bool Test_TileSize8_Epsilon_TestInputs_SwapAndSplitTreeIndex(TestArgs_t &args);
bool Test_TileSize8_Higgs_TestInputs_SwapAndSplitTreeIndex(TestArgs_t &args);
bool Test_TileSize8_Year_TestInputs_SwapAndSplitTreeIndex(TestArgs_t &args);

bool Test_TileSize8_Abalone_TestInputs_ParallelBatch(TestArgs_t &args);
bool Test_TileSize8_Airline_TestInputs_ParallelBatch(TestArgs_t &args);
bool Test_TileSize8_AirlineOHE_TestInputs_ParallelBatch(TestArgs_t &args);
bool Test_TileSize8_Covtype_TestInputs_ParallelBatch(TestArgs_t &args);
bool Test_TileSize8_Letters_TestInputs_ParallelBatch(TestArgs_t &args);
bool Test_TileSize8_Epsilon_TestInputs_ParallelBatch(TestArgs_t &args);
bool Test_TileSize8_Higgs_TestInputs_ParallelBatch(TestArgs_t &args);
bool Test_TileSize8_Year_TestInputs_ParallelBatch(TestArgs_t &args);

// Peeling
bool Test_WalkPeeling_BalancedTree_TileSize2(TestArgs_t &args);
bool Test_HybridTilingAndPeeling_RandomXGBoostJSONs_1Tree_FloatBatchSize4(
    TestArgs_t &args);
bool Test_HybridTilingAndPeeling_RandomXGBoostJSONs_2Tree_FloatBatchSize4(
    TestArgs_t &args);
bool Test_HybridTilingAndPeeling_RandomXGBoostJSONs_4Tree_FloatBatchSize4(
    TestArgs_t &args);
bool Test_UniformAndHybridTilingAndPeeling_RandomXGBoostJSONs_2Tree_FloatBatchSize4(
    TestArgs_t &args);
bool Test_UniformAndHybridTilingAndPeeling_RandomXGBoostJSONs_4Tree_FloatBatchSize4(
    TestArgs_t &args);
bool Test_PeeledHybridProbabilisticTiling_TileSize8_Abalone(TestArgs_t &args);
bool Test_PeeledHybridProbabilisticTiling_TileSize8_Airline(TestArgs_t &args);
bool Test_PeeledHybridProbabilisticTiling_TileSize8_AirlineOHE(
    TestArgs_t &args);
bool Test_PeeledHybridProbabilisticTiling_TileSize8_Covtype(TestArgs_t &args);
bool Test_PeeledHybridProbabilisticTiling_TileSize8_Letters(TestArgs_t &args);
bool Test_PeeledHybridProbabilisticTiling_TileSize8_Epsilon(TestArgs_t &args);
bool Test_PeeledHybridProbabilisticTiling_TileSize8_Higgs(TestArgs_t &args);
bool Test_PeeledHybridProbabilisticTiling_TileSize8_Year(TestArgs_t &args);

// ONNXTests
bool Test_ONNX_TileSize8_Abalone(TestArgs_t &args);

// GPU model initialization tests
bool Test_GPUModelInit_LeftHeavy_Scalar_DoubleInt(TestArgs_t &args);
bool Test_GPUModelInit_RightHeavy_Scalar_DoubleInt(TestArgs_t &args);
bool Test_GPUModelInit_Balanced_Scalar_DoubleInt(TestArgs_t &args);
bool Test_GPUModelInit_LeftAndRightHeavy_Scalar_DoubleInt(TestArgs_t &args);
bool Test_GPUModelInit_LeftHeavy_Scalar_FloatInt(TestArgs_t &args);
bool Test_GPUModelInit_RightHeavy_Scalar_FloatInt(TestArgs_t &args);
bool Test_GPUModelInit_Balanced_Scalar_FloatInt(TestArgs_t &args);
bool Test_GPUModelInit_LeftAndRightHeavy_Scalar_FloatInt(TestArgs_t &args);
bool Test_GPUModelInit_LeftHeavy_Scalar_FloatInt16(TestArgs_t &args);
bool Test_GPUModelInit_RightHeavy_Scalar_FloatInt16(TestArgs_t &args);
bool Test_GPUModelInit_Balanced_Scalar_FloatInt16(TestArgs_t &args);
bool Test_GPUModelInit_LeftAndRightHeavy_Scalar_FloatInt16(TestArgs_t &args);

bool Test_GPUModelInit_LeftHeavy_Reorg_DoubleInt(TestArgs_t &args);
bool Test_GPUModelInit_RightHeavy_Reorg_DoubleInt(TestArgs_t &args);
bool Test_GPUModelInit_Balanced_Reorg_DoubleInt(TestArgs_t &args);
bool Test_GPUModelInit_LeftAndRightHeavy_Reorg_DoubleInt(TestArgs_t &args);
bool Test_GPUModelInit_LeftHeavy_Reorg_FloatInt(TestArgs_t &args);
bool Test_GPUModelInit_RightHeavy_Reorg_FloatInt(TestArgs_t &args);
bool Test_GPUModelInit_Balanced_Reorg_FloatInt(TestArgs_t &args);
bool Test_GPUModelInit_LeftAndRightHeavy_Reorg_FloatInt(TestArgs_t &args);
bool Test_GPUModelInit_LeftHeavy_Reorg_FloatInt16(TestArgs_t &args);
bool Test_GPUModelInit_RightHeavy_Reorg_FloatInt16(TestArgs_t &args);
bool Test_GPUModelInit_Balanced_Reorg_FloatInt16(TestArgs_t &args);
bool Test_GPUModelInit_LeftAndRightHeavy_Reorg_FloatInt16(TestArgs_t &args);

// GPU basic code generation tests
bool Test_GPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32(TestArgs_t &args);
bool Test_GPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args);
bool Test_GPUCodeGeneration_Balanced_DoubleInt32_BatchSize32(TestArgs_t &args);
bool Test_GPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args);
bool Test_GPUCodeGeneration_LeftHeavy_FloatInt16_BatchSize32(TestArgs_t &args);
bool Test_GPUCodeGeneration_RightHeavy_FloatInt16_BatchSize32(TestArgs_t &args);
bool Test_GPUCodeGeneration_Balanced_FloatInt16_BatchSize32(TestArgs_t &args);
bool Test_GPUCodeGeneration_LeftAndRightHeavy_FloatInt16_BatchSize32(
    TestArgs_t &args);

bool Test_SparseGPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args);
bool Test_SparseGPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args);
bool Test_SparseGPUCodeGeneration_Balanced_DoubleInt32_BatchSize32(
    TestArgs_t &args);
bool Test_SparseGPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args);
bool Test_SparseGPUCodeGeneration_LeftHeavy_FloatInt16_ChI16_BatchSize32(
    TestArgs_t &args);
bool Test_SparseGPUCodeGeneration_RightHeavy_FloatInt16_ChI16_BatchSize32(
    TestArgs_t &args);
bool Test_SparseGPUCodeGeneration_Balanced_FloatInt16_ChI16_BatchSize32(
    TestArgs_t &args);
bool Test_SparseGPUCodeGeneration_LeftAndRightHeavy_FloatInt16_ChI16_BatchSize32(
    TestArgs_t &args);

bool Test_ReorgGPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args);
bool Test_ReorgGPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args);
bool Test_ReorgGPUCodeGeneration_Balanced_DoubleInt32_BatchSize32(
    TestArgs_t &args);
bool Test_ReorgGPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32(
    TestArgs_t &args);
bool Test_ReorgGPUCodeGeneration_LeftHeavy_FloatInt16_BatchSize32(
    TestArgs_t &args);
bool Test_ReorgGPUCodeGeneration_RightHeavy_FloatInt16_BatchSize32(
    TestArgs_t &args);
bool Test_ReorgGPUCodeGeneration_LeftAndRightHeavy_FloatInt16_BatchSize32(
    TestArgs_t &args);
bool Test_ReorgGPUCodeGeneration_LeftRightAndBalanced_FloatInt16_BatchSize32(
    TestArgs_t &args);

bool Test_SimpleSharedMem_LeftHeavy(TestArgs_t &args);
bool Test_SimpleSharedMem_LeftRightAndBalanced(TestArgs_t &args);
bool Test_SimpleSharedMem_LeftHeavy_F32I16(TestArgs_t &args);
bool Test_SimpleSharedMem_LeftRightAndBalanced_F32I16(TestArgs_t &args);

bool Test_SimpleSharedMem_LeftHeavy_ReorgRep(TestArgs_t &args);
bool Test_SimpleSharedMem_LeftRightAndBalanced_Reorg(TestArgs_t &args);
bool Test_SimpleSharedMem_LeftHeavy_ReorgRep_F32I16(TestArgs_t &args);
bool Test_SimpleSharedMem_LeftRightAndBalanced_Reorg_F32I16(TestArgs_t &args);

bool Test_SimpleSharedMem_LeftHeavy_SparseRep(TestArgs_t &args);
bool Test_SimpleSharedMem_LeftRightAndBalanced_SparseRep(TestArgs_t &args);
bool Test_SimpleSharedMem_LeftHeavy_SparseRep_F32I16(TestArgs_t &args);
bool Test_SimpleSharedMem_LeftRightAndBalanced_SparseRep_F32I16(
    TestArgs_t &args);

bool Test_GPUCodeGeneration_Abalone_TileSize1_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Abalone_TileSize2_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Abalone_TileSize4_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Abalone_TileSize8_BasicSchedule(TestArgs_t &args);

bool Test_GPUCodeGeneration_Airline_TileSize1_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Airline_TileSize2_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Airline_TileSize4_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Airline_TileSize8_BasicSchedule(TestArgs_t &args);

bool Test_GPUCodeGeneration_AirlineOHE_TileSize1_BasicSchedule(
    TestArgs_t &args);
bool Test_GPUCodeGeneration_AirlineOHE_TileSize2_BasicSchedule(
    TestArgs_t &args);
bool Test_GPUCodeGeneration_AirlineOHE_TileSize4_BasicSchedule(
    TestArgs_t &args);
bool Test_GPUCodeGeneration_AirlineOHE_TileSize8_BasicSchedule(
    TestArgs_t &args);

bool Test_GPUCodeGeneration_Bosch_TileSize1_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Bosch_TileSize2_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Bosch_TileSize4_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Bosch_TileSize8_BasicSchedule(TestArgs_t &args);

bool Test_GPUCodeGeneration_CovType_TileSize1_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_CovType_TileSize2_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_CovType_TileSize4_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_CovType_TileSize8_BasicSchedule(TestArgs_t &args);

bool Test_GPUCodeGeneration_Epsilon_TileSize1_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Epsilon_TileSize2_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Epsilon_TileSize4_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Epsilon_TileSize8_BasicSchedule(TestArgs_t &args);

bool Test_GPUCodeGeneration_Higgs_TileSize1_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Higgs_TileSize2_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Higgs_TileSize4_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Higgs_TileSize8_BasicSchedule(TestArgs_t &args);

bool Test_GPUCodeGeneration_Letters_TileSize1_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Letters_TileSize2_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Letters_TileSize4_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Letters_TileSize8_BasicSchedule(TestArgs_t &args);

bool Test_GPUCodeGeneration_Year_TileSize1_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Year_TileSize2_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Year_TileSize4_BasicSchedule(TestArgs_t &args);
bool Test_GPUCodeGeneration_Year_TileSize8_BasicSchedule(TestArgs_t &args);

bool Test_InputSharedMem_LeftRightAndBalanced(TestArgs_t &args);
bool Test_InputSharedMem_LeftHeavy(TestArgs_t &args);
bool Test_InputSharedMem_RightHeavy(TestArgs_t &args);

// Basic tiled tests
bool Test_TiledSparseGPU_LeftHeavy_DblI32_B32_TSz2(TestArgs_t &args);
bool Test_TiledSparseGPU_RightHeavy_DblI32_B32_TSz2(TestArgs_t &args);
bool Test_TiledSparseGPU_Balanced_DblI32_B32_TSz2(TestArgs_t &args);
bool Test_TiledSparseGPU_LeftAndRightHeavy_DblI32_B32_TSz2(TestArgs_t &args);
bool Test_TiledSparseGPU_LeftHeavy_FltI16_B32_TSz2(TestArgs_t &args);
bool Test_TiledSparseGPU_RightHeavy_FltI16_B32_TSz2(TestArgs_t &args);
bool Test_TiledSparseGPU_Balanced_FltI16_B32_TSz2(TestArgs_t &args);
bool Test_TiledSparseGPU_LeftAndRightHeavy_FltI16_B32_TSz2(TestArgs_t &args);

bool Test_TiledArrayGPU_LeftHeavy_DblI32_B32_TSz2(TestArgs_t &args);
bool Test_TiledArrayGPU_RightHeavy_DblI32_B32_TSz2(TestArgs_t &args);
bool Test_TiledArrayGPU_Balanced_DblI32_B32_TSz2(TestArgs_t &args);
bool Test_TiledArrayGPU_LeftAndRightHeavy_DblI32_B32_TSz2(TestArgs_t &args);
bool Test_TiledArrayGPU_LeftHeavy_FltI16_B32_TSz2(TestArgs_t &args);
bool Test_TiledArrayGPU_RightHeavy_FltI16_B32_TSz2(TestArgs_t &args);
bool Test_TiledArrayGPU_Balanced_FltI16_B32_TSz2(TestArgs_t &args);
bool Test_TiledArrayGPU_LeftAndRightHeavy_FltI16_B32_TSz2(TestArgs_t &args);

// Tiling + Shared memory
bool Test_TiledCachedArrayGPU_LeftHeavy_DblI32_B32_TSz2(TestArgs_t &args);
bool Test_TiledCachedArrayGPU_RightHeavy_DblI32_B32_TSz2(TestArgs_t &args);
bool Test_TiledCachedArrayGPU_Balanced_DblI32_B32_TSz2(TestArgs_t &args);
bool Test_TiledCachedArrayGPU_LeftAndRightHeavy_DblI32_B32_TSz2(
    TestArgs_t &args);
bool Test_TiledCachedArrayGPU_LeftRightAndBalanced_DblI32_B32_TSz2(
    TestArgs_t &args);
bool Test_TiledCachedArrayGPU_LeftHeavy_FltI16_B32_TSz2(TestArgs_t &args);
bool Test_TiledCachedArrayGPU_RightHeavy_FltI16_B32_TSz2(TestArgs_t &args);
bool Test_TiledCachedArrayGPU_Balanced_FltI16_B32_TSz2(TestArgs_t &args);
bool Test_TiledCachedArrayGPU_LeftAndRightHeavy_FltI16_B32_TSz2(
    TestArgs_t &args);
bool Test_TiledCachedArrayGPU_LeftRightAndBalanced_FltI16_B32_TSz2(
    TestArgs_t &args);

bool Test_TiledCachedSparseGPU_LeftHeavy_DblI32_B32_TSz2(TestArgs_t &args);
bool Test_TiledCachedSparseGPU_RightHeavy_DblI32_B32_TSz2(TestArgs_t &args);
bool Test_TiledCachedSparseGPU_Balanced_DblI32_B32_TSz2(TestArgs_t &args);
bool Test_TiledCachedSparseGPU_LeftAndRightHeavy_DblI32_B32_TSz2(
    TestArgs_t &args);
bool Test_TiledCachedSparseGPU_LeftRightAndBalanced_DblI32_B32_TSz2(
    TestArgs_t &args);
bool Test_TiledCachedSparseGPU_LeftHeavy_FltI16_B32_TSz2(TestArgs_t &args);
bool Test_TiledCachedSparseGPU_RightHeavy_FltI16_B32_TSz2(TestArgs_t &args);
bool Test_TiledCachedSparseGPU_Balanced_FltI16_B32_TSz2(TestArgs_t &args);
bool Test_TiledCachedSparseGPU_LeftAndRightHeavy_FltI16_B32_TSz2(
    TestArgs_t &args);
bool Test_TiledCachedSparseGPU_LeftRightAndBalanced_FltI16_B32_TSz2(
    TestArgs_t &args);

// Random XGBoost GPU tests
bool Test_GPU_1TreeXGB_Array_Scalar(TestArgs_t &args);
bool Test_GPU_2TreeXGB_Array_Scalar(TestArgs_t &args);
bool Test_GPU_4TreeXGB_Array_Scalar(TestArgs_t &args);
bool Test_GPU_1TreeXGB_Array_Scalar_f32i16(TestArgs_t &args);
bool Test_GPU_2TreeXGB_Array_Scalar_f32i16(TestArgs_t &args);
bool Test_GPU_4TreeXGB_Array_Scalar_f32i16(TestArgs_t &args);
bool Test_GPU_1TreeXGB_Sparse_Scalar(TestArgs_t &args);
bool Test_GPU_2TreeXGB_Sparse_Scalar(TestArgs_t &args);
bool Test_GPU_4TreeXGB_Sparse_Scalar(TestArgs_t &args);
bool Test_GPU_1TreeXGB_Sparse_Scalar_f32i16(TestArgs_t &args);
bool Test_GPU_2TreeXGB_Sparse_Scalar_f32i16(TestArgs_t &args);
bool Test_GPU_4TreeXGB_Sparse_Scalar_f32i16(TestArgs_t &args);
bool Test_GPU_1TreeXGB_Reorg_Scalar(TestArgs_t &args);
bool Test_GPU_2TreeXGB_Reorg_Scalar(TestArgs_t &args);
bool Test_GPU_4TreeXGB_Reorg_Scalar(TestArgs_t &args);
bool Test_GPU_1TreeXGB_Reorg_Scalar_f32i16(TestArgs_t &args);
bool Test_GPU_2TreeXGB_Reorg_Scalar_f32i16(TestArgs_t &args);
bool Test_GPU_4TreeXGB_Reorg_Scalar_f32i16(TestArgs_t &args);

// Random XGB tests - tile size 4
bool Test_GPU_1TreeXGB_Array_Tile4(TestArgs_t &args);
bool Test_GPU_2TreeXGB_Array_Tile4(TestArgs_t &args);
bool Test_GPU_4TreeXGB_Array_Tile4(TestArgs_t &args);
bool Test_GPU_1TreeXGB_Array_Tile4_f32i16(TestArgs_t &args);
bool Test_GPU_2TreeXGB_Array_Tile4_f32i16(TestArgs_t &args);
bool Test_GPU_4TreeXGB_Array_Tile4_f32i16(TestArgs_t &args);

bool Test_GPU_1TreeXGB_Sparse_Tile4(TestArgs_t &args);
bool Test_GPU_2TreeXGB_Sparse_Tile4(TestArgs_t &args);
bool Test_GPU_4TreeXGB_Sparse_Tile4(TestArgs_t &args);
bool Test_GPU_1TreeXGB_Sparse_Tile4_f32i16(TestArgs_t &args);
bool Test_GPU_2TreeXGB_Sparse_Tile4_f32i16(TestArgs_t &args);
bool Test_GPU_4TreeXGB_Sparse_Tile4_f32i16(TestArgs_t &args);

// Shared forest Random XGB Tests - Scalar
bool Test_GPU_SharedForest_1TreeXGB_Array_Scalar(TestArgs_t &args);
bool Test_GPU_SharedForest_2TreeXGB_Array_Scalar(TestArgs_t &args);
bool Test_GPU_SharedForest_4TreeXGB_Array_Scalar(TestArgs_t &args);
bool Test_GPU_SharedForest_1TreeXGB_Array_Scalar_f32i16(TestArgs_t &args);
bool Test_GPU_SharedForest_2TreeXGB_Array_Scalar_f32i16(TestArgs_t &args);
bool Test_GPU_SharedForest_4TreeXGB_Array_Scalar_f32i16(TestArgs_t &args);

bool Test_GPU_SharedForest_1TreeXGB_Sparse_Scalar(TestArgs_t &args);
bool Test_GPU_SharedForest_2TreeXGB_Sparse_Scalar(TestArgs_t &args);
bool Test_GPU_SharedForest_4TreeXGB_Sparse_Scalar(TestArgs_t &args);
bool Test_GPU_SharedForest_1TreeXGB_Sparse_Scalar_f32i16(TestArgs_t &args);
bool Test_GPU_SharedForest_2TreeXGB_Sparse_Scalar_f32i16(TestArgs_t &args);
bool Test_GPU_SharedForest_4TreeXGB_Sparse_Scalar_f32i16(TestArgs_t &args);

bool Test_GPU_SharedForest_1TreeXGB_Reorg_Scalar(TestArgs_t &args);
bool Test_GPU_SharedForest_2TreeXGB_Reorg_Scalar(TestArgs_t &args);
bool Test_GPU_SharedForest_4TreeXGB_Reorg_Scalar(TestArgs_t &args);
bool Test_GPU_SharedForest_1TreeXGB_Reorg_Scalar_f32i16(TestArgs_t &args);
bool Test_GPU_SharedForest_2TreeXGB_Reorg_Scalar_f32i16(TestArgs_t &args);
bool Test_GPU_SharedForest_4TreeXGB_Reorg_Scalar_f32i16(TestArgs_t &args);

bool Test_GPU_CachePartialForest1Tree_2TreeXGB_Sparse_Scalar(TestArgs_t &args);
bool Test_GPU_CachePartialForest2Trees_4TreeXGB_Sparse_Scalar(TestArgs_t &args);
bool Test_GPU_CachePartialForest2Trees_2TreeXGB_Sparse_Scalar_f32i16(
    TestArgs_t &args);
bool Test_GPU_CachePartialForest2Trees_4TreeXGB_Sparse_Scalar_f32i16(
    TestArgs_t &args);

bool Test_GPU_CachePartialForest1Tree_2TreeXGB_Array_Scalar(TestArgs_t &args);
bool Test_GPU_CachePartialForest1Tree_4TreeXGB_Array_Scalar(TestArgs_t &args);
bool Test_GPU_CachePartialForest2Trees_2TreeXGB_Array_Scalar_f32i16(
    TestArgs_t &args);
bool Test_GPU_CachePartialForest2Trees_4TreeXGB_Array_Scalar_f32i16(
    TestArgs_t &args);

bool Test_GPU_CachePartialForest1Tree_2TreeXGB_Reorg_Scalar(TestArgs_t &args);
bool Test_GPU_CachePartialForest2Trees_4TreeXGB_Reorg_Scalar(TestArgs_t &args);
bool Test_GPU_CachePartialForest2Trees_2TreeXGB_Reorg_Scalar_f32i16(
    TestArgs_t &args);
bool Test_GPU_CachePartialForest2Trees_4TreeXGB_Reorg_Scalar_f32i16(
    TestArgs_t &args);

// Tree parallelization tests
bool Test_TreePar_LeftRightAndBalanced_DblI32(TestArgs_t &args);
bool Test_NestedTreePar_LeftRightAndBalanced_DblI32(TestArgs_t &args);
bool Test_AtomicReduction_TwiceLeftRightAndBalanced_DblI32(TestArgs_t &args);
bool Test_VectorReduction_TwiceLeftRightAndBalanced_DblI32(TestArgs_t &args);

// Benchmark Model Tree Parallelization Tests
bool Test_SparseTileSize8_Abalone_TestInputs_4ParallelTreeSets(
    TestArgs_t &args);
bool Test_SparseTileSize8_Airline_TestInputs_4ParallelTreeSets(
    TestArgs_t &args);
bool Test_SparseTileSize8_AirlineOHE_TestInputs_4ParallelTreeSets(
    TestArgs_t &args);
bool Test_SparseTileSize8_Covtype_TestInputs_4ParallelTreeSets(
    TestArgs_t &args);
bool Test_SparseTileSize8_Letters_TestInputs_4ParallelTreeSets(
    TestArgs_t &args);
bool Test_SparseTileSize8_Epsilon_TestInputs_4ParallelTreeSets(
    TestArgs_t &args);
bool Test_SparseTileSize8_Higgs_TestInputs_4ParallelTreeSets(TestArgs_t &args);
bool Test_SparseTileSize8_Year_TestInputs_4ParallelTreeSets(TestArgs_t &args);

bool Test_SparseTileSize8_Abalone_TestInputs_4ParallelTreeSets_AtomicReduce(
    TestArgs_t &args);
bool Test_SparseTileSize8_Airline_TestInputs_4ParallelTreeSets_AtomicReduce(
    TestArgs_t &args);
bool Test_SparseTileSize8_AirlineOHE_TestInputs_4ParallelTreeSets_AtomicReduce(
    TestArgs_t &args);
bool Test_SparseTileSize8_Covtype_TestInputs_4ParallelTreeSets_AtomicReduce(
    TestArgs_t &args);
bool Test_SparseTileSize8_Letters_TestInputs_4ParallelTreeSets_AtomicReduce(
    TestArgs_t &args);
bool Test_SparseTileSize8_Epsilon_TestInputs_4ParallelTreeSets_AtomicReduce(
    TestArgs_t &args);
bool Test_SparseTileSize8_Higgs_TestInputs_4ParallelTreeSets_AtomicReduce(
    TestArgs_t &args);
bool Test_SparseTileSize8_Year_TestInputs_4ParallelTreeSets_AtomicReduce(
    TestArgs_t &args);

bool Test_SparseTileSize8_Abalone_TestInputs_4ParallelTreeSets_VectorReduce(
    TestArgs_t &args);
bool Test_SparseTileSize8_Airline_TestInputs_4ParallelTreeSets_VectorReduce(
    TestArgs_t &args);
bool Test_SparseTileSize8_AirlineOHE_TestInputs_4ParallelTreeSets_VectorReduce(
    TestArgs_t &args);
bool Test_SparseTileSize8_Covtype_TestInputs_4ParallelTreeSets_VectorReduce(
    TestArgs_t &args);
bool Test_SparseTileSize8_Letters_TestInputs_4ParallelTreeSets_VectorReduce(
    TestArgs_t &args);
bool Test_SparseTileSize8_Epsilon_TestInputs_4ParallelTreeSets_VectorReduce(
    TestArgs_t &args);
bool Test_SparseTileSize8_Higgs_TestInputs_4ParallelTreeSets_VectorReduce(
    TestArgs_t &args);
bool Test_SparseTileSize8_Year_TestInputs_4ParallelTreeSets_VectorReduce(
    TestArgs_t &args);

// GPU Tree parallelization tests
bool Test_ScalarSparseGPU_LeftRightAndBalanced_TahoeShdInp_FltI16_B32(
    TestArgs_t &args);
bool Test_ScalarSparseGPU_LeftRightAndBalanced_TahoeShdInpMultiRow_FltI16_B32(
    TestArgs_t &args);
bool Test_ScalarSparseGPU_TwiceLeftRightBalanced_TahoeShdInpMultiRow_FltI16_B32(
    TestArgs_t &args);
bool Test_ScalarSparseGPU_TwiceLeftRightBalanced_IterShdPartialForest_FltI16_B32(
    TestArgs_t &args);

// GPU Synthetic XGB Models Tree Parallelization Tests
bool Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Sparse_Scalar(TestArgs_t &args);
bool Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Sparse_Scalar(TestArgs_t &args);
bool Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Sparse_Scalar_f32i16(
    TestArgs_t &args);
bool Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Sparse_Scalar_f32i16(
    TestArgs_t &args);
bool Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Array_Scalar(TestArgs_t &args);
bool Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Array_Scalar(TestArgs_t &args);
bool Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Array_Scalar_f32i16(
    TestArgs_t &args);
bool Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Array_Scalar_f32i16(
    TestArgs_t &args);
bool Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Reorg_Scalar(TestArgs_t &args);
bool Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Reorg_Scalar(TestArgs_t &args);
bool Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Reorg_Scalar_f32i16(
    TestArgs_t &args);
bool Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Reorg_Scalar_f32i16(
    TestArgs_t &args);
bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Scalar(
    TestArgs_t &args);
bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Scalar_f32i16(
    TestArgs_t &args);
bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Array_Scalar(
    TestArgs_t &args);
bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Array_Scalar_f32i16(
    TestArgs_t &args);
bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Reorg_Scalar(
    TestArgs_t &args);
bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Reorg_Scalar_f32i16(
    TestArgs_t &args);

// GPU Tree parallelization tests - Tile size 4
bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Tile4(
    TestArgs_t &args);
bool Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Tile4_f32i16(
    TestArgs_t &args);

// GPU Tree parallelization - Shared reduce
bool Test_ScalarSparseGPU_TwiceLeftRightBalanced_iterCachedPartialForest_NoCache_SharedReduce_FltI16_B64(
    TestArgs_t &args);

// Specialize schedule tests
bool Test_ScalarSparseGPU_TwiceLeftRightBalanced_iterCachedPartialForest_NoCache_SpecializedTreeLoop_FltI16_B64(
    TestArgs_t &args);

// Multi-class with trees split across multiple threads
bool Test_GPUCodeGeneration_Covtype_SparseRep_f32i16_B32_iterativeCachedPartialForestStrategy_NoCache(
    TestArgs_t &args);
bool Test_GPUCodeGeneration_Covtype_SparseRep_f32i16_B32_iterativeCachedPartialForestStrategy_NoCache_SharedReduce(
    TestArgs_t &args);

// GPU basic auto scheduling tests
bool Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleBasic(
    TestArgs_t &args);
bool Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleCachedRows(
    TestArgs_t &args);
bool Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleCachedTrees(
    TestArgs_t &args);

// GPU XGBoost auto scheduling tests
bool Test_ScalarGPU_Airline_AutoScheduleBasic(TestArgs_t &args);
bool Test_ScalarGPU_Abalone_AutoScheduleBasic(TestArgs_t &args);
bool Test_GPUCodeGeneration_Covtype_SparseRep_f32i16_B512_AutoSched_SharedReduce(
    TestArgs_t &args);
bool Test_GPUCodeGeneration_Letters_SparseRep_f32i16_B512_AutoSched_SharedReduce(
    TestArgs_t &args);

// CPU Autoschedule tests
bool Test_TileSize8_Abalone_TestInputs_CPUAutoSchedule_TreeParallel_f32i16(
    TestArgs_t &args);
bool Test_TileSize8_Airline_TestInputs_CPUAutoSchedule_TreeParallel_f32i16(
    TestArgs_t &args);
bool Test_TileSize8_AirlineOHE_TestInputs_CPUAutoSchedule_TreeParallel_f32i16(
    TestArgs_t &args);
bool Test_TileSize8_Covtype_TestInputs_CPUAutoSchedule_TreeParallel_f32i16(
    TestArgs_t &args);
bool Test_TileSize8_Epsilon_TestInputs_CPUAutoSchedule_TreeParallel_f32i16(
    TestArgs_t &args);
bool Test_TileSize8_Higgs_TestInputs_CPUAutoSchedule_TreeParallel_f32i16(
    TestArgs_t &args);
bool Test_TileSize8_Letters_TestInputs_CPUAutoSchedule_TreeParallel_f32i16(
    TestArgs_t &args);
bool Test_TileSize8_Year_TestInputs_CPUAutoSchedule_TreeParallel_f32i16(
    TestArgs_t &args);

bool Test_GPUCodeGeneration_Abalone_SparseRep_f32i16_B32_iterativeCachedPartialForestStrategy_NoCache_SharedReduce(
    TestArgs_t &args);


void InitializeVectorWithRandValues(std::vector<double> &vec) {
  for (size_t i = 0; i < vec.size(); ++i)
    vec[i] = (double)rand() / RAND_MAX;
}

template <typename ThresholdType, typename IndexType>
bool Test_BufferInit_RightHeavy(TestArgs_t &args) {
  using TileType = NumericalTileType_Packed<ThresholdType, IndexType>;

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);

  mlir::OpBuilder builder(&context);
  mlir::decisionforest::DecisionForest forest;
  auto expectedArray = AddRightHeavyTree<TileType>(forest);

  // Construct basic tree types for the tree
  // (Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type
  // thresholdType, Type featureIndexType
  auto thresholdType = TreeBeard::GetMLIRType(ThresholdType(), builder);
  auto indexType = TreeBeard::GetMLIRType(IndexType(), builder);
  auto treeType = mlir::decisionforest::TreeType::get(
      thresholdType, 1 /*tileSize*/, thresholdType, indexType);
  //(Type resultType, size_t numTrees, Type rowType, ReductionType
  // reductionType, Type treeType)
  auto forestType = mlir::decisionforest::TreeEnsembleType::get(
      thresholdType, 1,
      thresholdType /*HACK type doesn't matter for this test*/,
      mlir::decisionforest::ReductionType::kAdd, treeType);

  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(
      GetGlobalJSONNameForTests());
  decisionforest::ConstructModelSerializer(GetGlobalJSONNameForTests())
      ->Persist(forest, forestType);
  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(
      GetGlobalJSONNameForTests());
  mlir::decisionforest::ForestJSONReader::GetInstance().ParseJSONFile();

  std::vector<TileType> serializedTree(
      std::pow(2, 3) -
      1); // Depth of the tree is 3, so this is the size of the dense array
  // InitializeBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth,
  // int32_t indexBitWidth, std::vector<int32_t>& treeOffsets)
  std::vector<int32_t> offsets(1, -1);
  int32_t thresholdSize = sizeof(ThresholdType) * 8;
  int32_t indexSize = sizeof(IndexType) * 8;
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(
      serializedTree.data(), 1, thresholdSize, indexSize, offsets);
  Test_ASSERT(expectedArray == serializedTree);
  Test_ASSERT(offsets[0] == 0);

  std::vector<int64_t> offsetVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(
      offsetVec.data(), 1, thresholdSize, indexSize);
  Test_ASSERT(offsetVec[0] == 0);

  std::vector<int64_t> lengthVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(
      lengthVec.data(), 1, thresholdSize, indexSize);
  Test_ASSERT(lengthVec[0] == 7);

  return true;
}

bool Test_BufferInitializationWithOneTree_RightHeavy(TestArgs_t &args) {
  return Test_BufferInit_RightHeavy<double, int32_t>(args);
}

bool Test_BufferInitializationWithOneTree_RightHeavy_Int16(TestArgs_t &args) {
  return Test_BufferInit_RightHeavy<double, int16_t>(args);
}

bool Test_BufferInitializationWithOneTree_RightHeavy_Int8(TestArgs_t &args) {
  return Test_BufferInit_RightHeavy<double, int8_t>(args);
}

bool Test_BufferInitializationWithOneTree_RightHeavy_Float(TestArgs_t &args) {
  return Test_BufferInit_RightHeavy<float, int32_t>(args);
}

bool Test_BufferInitializationWithOneTree_RightHeavy_FloatInt16(
    TestArgs_t &args) {
  return Test_BufferInit_RightHeavy<float, int16_t>(args);
}

bool Test_BufferInitializationWithOneTree_RightHeavy_FloatInt8(
    TestArgs_t &args) {
  return Test_BufferInit_RightHeavy<float, int8_t>(args);
}

template <typename ThresholdType, typename IndexType>
bool Test_BufferInitialization_TwoTrees(TestArgs_t &args) {
  using TileType = NumericalTileType_Packed<ThresholdType, IndexType>;

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);

  mlir::OpBuilder builder(&context);
  mlir::decisionforest::DecisionForest forest;
  auto expectedArray = AddRightHeavyTree<TileType>(forest);
  auto expectedArray2 = AddLeftHeavyTree<TileType>(forest);
  expectedArray.insert(std::end(expectedArray), std::begin(expectedArray2),
                       std::end(expectedArray2));

  // Construct basic tree types for the tree
  // (Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type
  // thresholdType, Type featureIndexType
  auto thresholdType = TreeBeard::GetMLIRType(ThresholdType(), builder);
  auto indexType = TreeBeard::GetMLIRType(IndexType(), builder);
  auto treeType = mlir::decisionforest::TreeType::get(
      thresholdType, 1 /*tileSize*/, thresholdType, indexType);
  //(Type resultType, size_t numTrees, Type rowType, ReductionType
  // reductionType, Type treeType)
  auto forestType = mlir::decisionforest::TreeEnsembleType::get(
      thresholdType, 1,
      thresholdType /*HACK type doesn't matter for this test*/,
      mlir::decisionforest::ReductionType::kAdd, treeType);

  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(
      GetGlobalJSONNameForTests());
  decisionforest::ConstructModelSerializer(GetGlobalJSONNameForTests())
      ->Persist(forest, forestType);
  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(
      GetGlobalJSONNameForTests());
  mlir::decisionforest::ForestJSONReader::GetInstance().ParseJSONFile();

  std::vector<TileType> serializedTree(
      2 *
      (std::pow(2, 3) -
       1)); // Depth of the tree is 3, so this is the size of the dense array
  // InitializeBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth,
  // int32_t indexBitWidth, std::vector<int32_t>& treeOffsets)
  int32_t thresholdSize = sizeof(ThresholdType) * 8;
  int32_t indexSize = sizeof(IndexType) * 8;
  std::vector<int32_t> offsets(2, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(
      serializedTree.data(), 1, thresholdSize, indexSize, offsets);
  Test_ASSERT(expectedArray == serializedTree);
  Test_ASSERT(offsets[0] == 0);
  Test_ASSERT(offsets[1] == 7);

  std::vector<int64_t> offsetVec(2, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(
      offsetVec.data(), 1, thresholdSize, indexSize);
  Test_ASSERT(offsetVec[0] == 0);
  Test_ASSERT(offsetVec[1] == 7);

  std::vector<int64_t> lengthVec(2, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(
      lengthVec.data(), 1, thresholdSize, indexSize);
  Test_ASSERT(lengthVec[0] == 7);
  Test_ASSERT(lengthVec[1] == 7);

  return true;
}

bool Test_BufferInitializationWithOneTree_LeftHeavy(TestArgs_t &args) {
  using DoubleInt32Tile = NumericalTileType_Packed<double, int32_t>;

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);

  mlir::decisionforest::DecisionForest forest;
  auto expectedArray = AddLeftHeavyTree<DoubleInt32Tile>(forest);

  // Construct basic tree types for the tree
  // (Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type
  // thresholdType, Type featureIndexType
  auto doubleType = mlir::Float64Type::get(&context);
  auto int32Type = mlir::IntegerType::get(&context, 32);
  auto treeType = mlir::decisionforest::TreeType::get(
      doubleType, 1 /*tileSize*/, doubleType, int32Type);
  //(Type resultType, size_t numTrees, Type rowType, ReductionType
  // reductionType, Type treeType)
  auto forestType = mlir::decisionforest::TreeEnsembleType::get(
      doubleType, 1, doubleType /*HACK type doesn't matter for this test*/,
      mlir::decisionforest::ReductionType::kAdd, treeType);

  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(
      GetGlobalJSONNameForTests());
  decisionforest::ConstructModelSerializer(GetGlobalJSONNameForTests())
      ->Persist(forest, forestType);
  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(
      GetGlobalJSONNameForTests());
  mlir::decisionforest::ForestJSONReader::GetInstance().ParseJSONFile();

  std::vector<DoubleInt32Tile> serializedTree(
      std::pow(2, 3) -
      1); // Depth of the tree is 3, so this is the size of the dense array

  // InitializeBuffer(void* bufPtr, int32_t tileSize, int32_t thresholdBitWidth,
  // int32_t indexBitWidth, std::vector<int32_t>& treeOffsets)
  std::vector<int32_t> offsets(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(
      serializedTree.data(), 1, 64, 32, offsets);
  Test_ASSERT(expectedArray == serializedTree);
  Test_ASSERT(offsets[0] == 0);

  std::vector<int64_t> offsetVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(
      offsetVec.data(), 1, 64, 32);
  Test_ASSERT(offsetVec[0] == 0);

  std::vector<int64_t> lengthVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(
      lengthVec.data(), 1, 64, 32);
  Test_ASSERT(lengthVec[0] == 7);

  return true;
}

bool Test_BufferInitializationWithTwoTrees(TestArgs_t &args) {
  return Test_BufferInitialization_TwoTrees<double, int32_t>(args);
}

bool Test_BufferInitializationWithTwoTrees_Int16(TestArgs_t &args) {
  return Test_BufferInitialization_TwoTrees<double, int16_t>(args);
}

bool Test_BufferInitializationWithTwoTrees_Int8(TestArgs_t &args) {
  return Test_BufferInitialization_TwoTrees<double, int8_t>(args);
}

bool Test_BufferInitializationWithTwoTrees_Float(TestArgs_t &args) {
  return Test_BufferInitialization_TwoTrees<float, int32_t>(args);
}

bool Test_BufferInitializationWithTwoTrees_FloatInt16(TestArgs_t &args) {
  return Test_BufferInitialization_TwoTrees<float, int16_t>(args);
}

bool Test_BufferInitializationWithTwoTrees_FloatInt8(TestArgs_t &args) {
  return Test_BufferInitialization_TwoTrees<float, int8_t>(args);
}

// IR Tests
bool Test_ForestCodeGen_BatchSize1(
    TestArgs_t &args, ForestConstructor_t forestConstructor,
    std::vector<std::vector<double>> &inputData, int32_t childIndexBitWidth = 1,
    ScheduleManipulator_t scheduleManipulator = nullptr) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  auto serializer =
      decisionforest::ConstructModelSerializer(modelGlobalsJSONPath);

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);

  FixedTreeIRConstructor<> irConstructor(context, serializer, 1,
                                         forestConstructor);
  irConstructor.ConstructForest();
  // If sparse representation is turned on, then child index bit width should be
  // passed
  assert(!mlir::decisionforest::UseSparseTreeRepresentation ||
         childIndexBitWidth != 1);
  irConstructor.SetChildIndexBitWidth(childIndexBitWidth);
  auto module = irConstructor.GetEvaluationFunction();

  if (scheduleManipulator) {
    auto schedule = irConstructor.GetSchedule();
    scheduleManipulator(schedule);
  }

  // module->dump();
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(context, module);
  mlir::decisionforest::legalizeReductionsAndCanonicalize(context, module);
  mlir::decisionforest::lowerReductionsAndCanonicalize(context, module);
  // module->dump();
  auto representation = decisionforest::ConstructRepresentation();
  mlir::decisionforest::LowerEnsembleToMemrefs(context, module, serializer,
                                               representation);
  // module->dump();
  mlir::decisionforest::ConvertNodeTypeToIndexType(context, module);
  // module->dump();
  mlir::decisionforest::LowerToLLVM(context, module, representation);
  // module->dump();
  // mlir::decisionforest::dumpLLVMIR(module);
  decisionforest::InferenceRunner inferenceRunner(serializer, module, 1, 64,
                                                  32);

  // inferenceRunner.PrintLengthsArray();
  // inferenceRunner.PrintOffsetsArray();

  for (auto &row : inputData) {
    double result = -1;
    inferenceRunner.RunInference<double, double>(row.data(), &result);
    double expectedResult = irConstructor.GetForest().Predict(row);
    Test_ASSERT(FPEqual(result, expectedResult));
  }
  return true;
}

bool Test_ForestCodeGen_VariableBatchSize(
    TestArgs_t &args, ForestConstructor_t forestConstructor, int64_t batchSize,
    std::vector<std::vector<double>> &inputData, int32_t childIndexBitWidth = 1,
    ScheduleManipulator_t scheduleManipulator = nullptr) {
  auto modelGlobalsJSONPath =
      TreeBeard::ForestCreator::ModelGlobalJSONFilePathFromJSONFilePath(
          TreeBeard::test::GetGlobalJSONNameForTests());
  auto serializer =
      decisionforest::ConstructModelSerializer(modelGlobalsJSONPath);

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);

  FixedTreeIRConstructor<> irConstructor(context, serializer, batchSize,
                                         forestConstructor);
  irConstructor.ConstructForest();
  irConstructor.SetChildIndexBitWidth(childIndexBitWidth);
  auto module = irConstructor.GetEvaluationFunction();

  if (scheduleManipulator) {
    auto schedule = irConstructor.GetSchedule();
    scheduleManipulator(schedule);
  }

  // module->dump();
  mlir::decisionforest::LowerFromHighLevelToMidLevelIR(context, module);
  // module->dump();

  mlir::decisionforest::legalizeReductionsAndCanonicalize(context, module);
  mlir::decisionforest::lowerReductionsAndCanonicalize(context, module);
  // module->dump();

  auto representation = decisionforest::ConstructRepresentation();
  mlir::decisionforest::LowerEnsembleToMemrefs(context, module, serializer,
                                               representation);
  mlir::decisionforest::ConvertNodeTypeToIndexType(context, module);
  // module->dump();
  mlir::decisionforest::LowerToLLVM(context, module, representation);
  // module->dump();
  // mlir::decisionforest::dumpLLVMIR(module);
  decisionforest::InferenceRunner inferenceRunner(serializer, module, 1, 64,
                                                  32);

  // inferenceRunner.PrintLengthsArray();
  // inferenceRunner.PrintOffsetsArray();

  for (auto &batch : inputData) {
    assert(batch.size() % batchSize == 0);
    size_t rowSize = batch.size() / batchSize;
    std::vector<double> result(batchSize, -1);
    inferenceRunner.RunInference<double, double>(batch.data(), result.data());
    for (int64_t rowIdx = 0; rowIdx < batchSize; ++rowIdx) {
      std::vector<double> row(batch.begin() + rowIdx * rowSize,
                              batch.begin() + (rowIdx + 1) * rowSize);
      double expectedResult = irConstructor.GetForest().Predict(row);
      Test_ASSERT(FPEqual(result[rowIdx], expectedResult));
    }
  }
  return true;
}

std::vector<std::vector<double>> GetBatchSize1Data() {
  std::vector<double> inputData1 = {0.1, 0.2, 0.5, 0.3, 0.25};
  std::vector<double> inputData2 = {0.1, 0.2, 0.6, 0.3, 0.25};
  std::vector<double> inputData3 = {0.1, 0.2, 0.4, 0.3, 0.25};
  std::vector<std::vector<double>> data = {inputData1, inputData2, inputData3};
  return data;
}

std::vector<std::vector<double>> GetBatchSize2Data() {
  std::vector<double> inputData1 = {0.1, 0.2, 0.4, 0.3, 0.25,
                                    0.1, 0.2, 0.6, 0.3, 0.25};
  std::vector<std::vector<double>> data = {inputData1};
  return data;
}

bool Test_CodeGeneration_LeftHeavy_BatchSize1(TestArgs_t &args) {
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddLeftHeavyTree<DoubleInt32Tile>,
                                       data);
}

bool Test_CodeGeneration_RightHeavy_BatchSize1(TestArgs_t &args) {
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddRightHeavyTree<DoubleInt32Tile>,
                                       data);
}

bool Test_CodeGeneration_RightAndLeftHeavy_BatchSize1(TestArgs_t &args) {
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(
      args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, data);
}

bool Test_CodeGeneration_LeftHeavy_BatchSize2(TestArgs_t &args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(
      args, AddLeftHeavyTree<DoubleInt32Tile>, 2, data);
}

bool Test_CodeGeneration_RightHeavy_BatchSize2(TestArgs_t &args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(
      args, AddRightHeavyTree<DoubleInt32Tile>, 2, data);
}

bool Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2(
    TestArgs_t &args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(
      args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 2, data);
}

// ===----------------------------------------=== //
// Basic non-trivial schedule code gen tests
// ===----------------------------------------=== //

bool Test_CodeGeneration_LeftHeavy_BatchSize2_XGBoostSchedule(
    TestArgs_t &args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(
      args, AddLeftHeavyTree<DoubleInt32Tile>, 2, data, 1,
      OneTreeAtATimeSchedule);
}

bool Test_CodeGeneration_RightHeavy_BatchSize2_XGBoostSchedule(
    TestArgs_t &args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(
      args, AddRightHeavyTree<DoubleInt32Tile>, 2, data, 1,
      OneTreeAtATimeSchedule);
}

bool Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2_XGBoostSchedule(
    TestArgs_t &args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(
      args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 2, data, 1,
      OneTreeAtATimeSchedule);
}

bool Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2_UnrollTreeLoop(
    TestArgs_t &args) {
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(
      args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 2, data, 1,
      UnrollTreeLoop);
}

// ===----------------------------------------=== //
// Basic CPU input caching schedule code gen tests
// ===----------------------------------------=== //

void BasicCachedSchedule(mlir::decisionforest::Schedule *schedule) {
  auto &batchIndexVar = schedule->GetBatchIndex();

  auto &b0 = schedule->NewIndexVariable("b0");
  auto &b1 = schedule->NewIndexVariable("b1");

  schedule->Tile(batchIndexVar, b0, b1, 2);
  schedule->Cache(b0);
}

void PopulateDataForBatchSize(std::vector<std::vector<double>> &inputData,
                              int32_t batchSize, int32_t numBatches) {
  for (int32_t i = 0; i < numBatches; ++i) {
    inputData.emplace_back(std::vector<double>());
    auto &firstVec = inputData.back();
    for (int32_t j = 0; j < batchSize / 2; ++j) {
      auto data = GetBatchSize2Data();
      firstVec.insert(firstVec.end(), data.front().begin(), data.front().end());
    }
  }
}

bool Test_CodeGeneration_LeftHeavy_BatchSize8_CacheInputSchedule(
    TestArgs_t &args) {
  std::vector<std::vector<double>> inputData;
  PopulateDataForBatchSize(inputData, 8, 4);
  return Test_ForestCodeGen_VariableBatchSize(
      args, AddLeftHeavyTree<DoubleInt32Tile>, 8, inputData, 1,
      BasicCachedSchedule);
}

bool Test_CodeGeneration_RightHeavy_BatchSize2_CacheInputSchedule(
    TestArgs_t &args) {
  std::vector<std::vector<double>> inputData;
  PopulateDataForBatchSize(inputData, 8, 4);
  return Test_ForestCodeGen_VariableBatchSize(
      args, AddRightHeavyTree<DoubleInt32Tile>, 8, inputData, 1,
      BasicCachedSchedule);
}

bool Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2_CacheInputSchedule(
    TestArgs_t &args) {
  std::vector<std::vector<double>> inputData;
  PopulateDataForBatchSize(inputData, 8, 4);
  return Test_ForestCodeGen_VariableBatchSize(
      args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 8, inputData, 1,
      BasicCachedSchedule);
}

// ===----------------------------------------=== //
// Basic sparse code gen tests
// ===----------------------------------------=== //

bool Test_SparseCodeGeneration_LeftHeavy_BatchSize1_I32ChildIdx(
    TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddLeftHeavyTree<DoubleInt32Tile>,
                                       data, 32);
}

bool Test_SparseCodeGeneration_RightHeavy_BatchSize1_I32ChildIdx(
    TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(args, AddRightHeavyTree<DoubleInt32Tile>,
                                       data, 32);
}

bool Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize1_I32ChildIdx(
    TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto data = GetBatchSize1Data();
  return Test_ForestCodeGen_BatchSize1(
      args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, data, 32);
}

bool Test_SparseCodeGeneration_LeftHeavy_BatchSize2_I32ChildIdx(
    TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(
      args, AddLeftHeavyTree<DoubleInt32Tile>, 2, data, 32);
}

bool Test_SparseCodeGeneration_RightHeavy_BatchSize2_I32ChildIdx(
    TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(
      args, AddRightHeavyTree<DoubleInt32Tile>, 2, data, 32);
}

bool Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize2_I32ChildIdx(
    TestArgs_t &args) {
  decisionforest::UseSparseTreeRepresentation = true;
  auto data = GetBatchSize2Data();
  return Test_ForestCodeGen_VariableBatchSize(
      args, AddRightAndLeftHeavyTrees<DoubleInt32Tile>, 2, data, 32);
}

// Tests for Tiled Buffer Initialization
template <typename ThresholdType, typename IndexType>
bool Test_BufferInit_SingleTree_Tiled(TestArgs_t &args,
                                      ForestConstructor_t forestConstructor,
                                      std::vector<int32_t> &tileIDs) {
  using VectorTileType =
      NumericalVectorTileType_Packed<ThresholdType, IndexType, 3>;

  MLIRContext context;
  TreeBeard::InitializeMLIRContext(context);

  mlir::OpBuilder builder(&context);
  mlir::decisionforest::DecisionForest forest;
  forestConstructor(forest);

  // Construct basic tree types for the tree
  // (Type resultType, const TreeTilingDescriptor& tilingDescriptor, Type
  // thresholdType, Type featureIndexType
  auto thresholdType = TreeBeard::GetMLIRType(ThresholdType(), builder);
  auto indexType = TreeBeard::GetMLIRType(IndexType(), builder);

  int32_t tileSize = 3;
  decisionforest::TreeTilingDescriptor tilingDescriptor(
      tileSize /*tile size*/, 4 /*num tiles*/, tileIDs,
      decisionforest::TilingType::kRegular);
  forest.GetTree(0).SetTilingDescriptor(tilingDescriptor);

  auto treeType = mlir::decisionforest::TreeType::get(
      thresholdType, tilingDescriptor.MaxTileSize(), thresholdType, indexType);
  //(Type resultType, size_t numTrees, Type rowType, ReductionType
  // reductionType, Type treeType)
  std::vector<Type> treeTypes = {treeType};
  auto forestType = mlir::decisionforest::TreeEnsembleType::get(
      thresholdType, 1,
      thresholdType /*HACK type doesn't matter for this test*/,
      mlir::decisionforest::ReductionType::kAdd, treeTypes);
  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(
      GetGlobalJSONNameForTests());
  decisionforest::ConstructModelSerializer(GetGlobalJSONNameForTests())
      ->Persist(forest, forestType);
  mlir::decisionforest::ForestJSONReader::GetInstance().SetFilePath(
      GetGlobalJSONNameForTests());
  mlir::decisionforest::ForestJSONReader::GetInstance().ParseJSONFile();

  mlir::decisionforest::TiledTree tiledTree(forest.GetTree(0));
  auto numTiles = tiledTree.GetNumberOfTiles();
  std::vector<VectorTileType> serializedTree(numTiles);
  auto thresholds = tiledTree.SerializeThresholds();
  auto featureIndices = tiledTree.SerializeFeatureIndices();
  auto tileShapeIDs = tiledTree.SerializeTileShapeIDs();

  std::vector<int32_t> offsets(1, -1);
  int32_t thresholdSize = sizeof(ThresholdType) * 8;
  int32_t indexSize = sizeof(IndexType) * 8;
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeBuffer(
      serializedTree.data(), tileSize, thresholdSize, indexSize, offsets);
  for (int32_t i = 0; i < numTiles; ++i) {
    for (int32_t j = 0; j < tileSize; ++j) {
      Test_ASSERT(FPEqual(serializedTree[i].threshold[j],
                          thresholds[i * tileSize + j]));
      Test_ASSERT(serializedTree[i].index[j] ==
                  featureIndices[i * tileSize + j]);
    }
    // std::cout << tileShapeIDs[i] << std::endl;
    Test_ASSERT(tileShapeIDs[i] == serializedTree[i].tileShapeID);
  }
  Test_ASSERT(offsets[0] == 0);

  std::vector<int64_t> offsetVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeOffsetBuffer(
      offsetVec.data(), tileSize, thresholdSize, indexSize);
  Test_ASSERT(offsetVec[0] == 0);

  std::vector<int64_t> lengthVec(1, -1);
  mlir::decisionforest::ForestJSONReader::GetInstance().InitializeLengthBuffer(
      lengthVec.data(), tileSize, thresholdSize, indexSize);
  Test_ASSERT(lengthVec[0] == numTiles);

  return true;
}

bool Test_BufferInitializationWithOneTree_RightHeavy_Tiled(TestArgs_t &args) {
  using TileType = NumericalTileType_Packed<double, int32_t>;
  std::vector<int32_t> tileIDs = {
      0, 0, 1, 2, 3}; // The root and one of its children are in one tile and
                      // all leaves are in separate tiles
  return Test_BufferInit_SingleTree_Tiled<double, int32_t>(
      args, AddRightHeavyTree<TileType>, tileIDs);
}

bool Test_BufferInitializationWithOneTree_LeftHeavy_Tiled(TestArgs_t &args) {
  using TileType = NumericalTileType_Packed<double, int32_t>;
  std::vector<int32_t> tileIDs = {
      0, 0, 1, 2, 3}; // The root and one of its children are in one tile and
                      // all leaves are in separate tiles
  return Test_BufferInit_SingleTree_Tiled<double, int32_t>(
      args, AddLeftHeavyTree<TileType>, tileIDs);
}

bool Test_BufferInitializationWithOneTree_Balanced_Tiled(TestArgs_t &args) {
  using TileType = NumericalTileType_Packed<double, int32_t>;
  std::vector<int32_t> tileIDs = {0, 0, 1, 2, 0, 3, 4};
  return Test_BufferInit_SingleTree_Tiled<double, int32_t>(
      args, AddBalancedTree<TileType>, tileIDs);
}

// ===-------------------------------------------------------------=== //
// Tiled Tree Tests
// ===-------------------------------------------------------------=== //
bool CheckAllLeavesAreAtSameDepth(mlir::decisionforest::TiledTree *tiledTree) {
  auto depth = tiledTree->GetTreeDepth();
  for (int32_t i = 0; i < (int32_t)tiledTree->NumTiles(); ++i) {
    auto &tile = tiledTree->GetTile(i);
    if (!tile.IsLeafTile())
      continue;
    auto tilePtr = &tile;
    int32_t leafDepth = 1;
    while (tilePtr->GetParent() !=
           decisionforest::DecisionTree::INVALID_NODE_INDEX) {
      tilePtr = &(tiledTree->GetTile(tilePtr->GetParent()));
      ++leafDepth;
    }
    Test_ASSERT(leafDepth == depth);
  }
  return true;
}

bool Test_PadTiledTree_BalancedTree_TileSize2(TestArgs_t &args) {
  decisionforest::DecisionTree decisionTree;
  InitializeBalancedTree(decisionTree);
  std::vector<int32_t> tileIDs = {0, 0, 1, 2, 5, 3, 4};

  decisionforest::TreeTilingDescriptor tilingDescriptor(
      2, 6, tileIDs, decisionforest::TilingType::kRegular);
  decisionTree.SetTilingDescriptor(tilingDescriptor);

  auto tiledTree = decisionTree.GetTiledTree();
  tiledTree->MakeAllLeavesSameDepth();
  // std::string dotFile =
  // "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/temp/tiledTree.dot";
  // tiledTree->WriteDOTFile(dotFile);
  return CheckAllLeavesAreAtSameDepth(tiledTree);
}

bool Test_PadTiledTree_BalancedTree_TileSize2_2(TestArgs_t &args) {
  decisionforest::DecisionTree decisionTree;
  InitializeBalancedTree(decisionTree);
  std::vector<int32_t> tileIDs = {0, 5, 1, 2, 0, 3, 4};

  decisionforest::TreeTilingDescriptor tilingDescriptor(
      2, 6, tileIDs, decisionforest::TilingType::kRegular);
  decisionTree.SetTilingDescriptor(tilingDescriptor);

  auto tiledTree = decisionTree.GetTiledTree();
  tiledTree->MakeAllLeavesSameDepth();
  // std::string dotFile =
  // "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/temp/tiledTree.dot";
  // tiledTree->WriteDOTFile(dotFile);
  return CheckAllLeavesAreAtSameDepth(tiledTree);
}

bool Test_PadTiledTree_BalancedTree_TileSize3(TestArgs_t &args) {
  decisionforest::DecisionTree decisionTree;
  InitializeBalancedTree(decisionTree);
  std::vector<int32_t> tileIDs = {0, 0, 1, 2, 0, 3, 4};

  decisionforest::TreeTilingDescriptor tilingDescriptor(
      3, 5, tileIDs, decisionforest::TilingType::kRegular);
  decisionTree.SetTilingDescriptor(tilingDescriptor);

  auto tiledTree = decisionTree.GetTiledTree();
  tiledTree->MakeAllLeavesSameDepth();
  // std::string dotFile =
  // "/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/debug/temp/tiledTree.dot";
  // tiledTree->WriteDOTFile(dotFile);
  return CheckAllLeavesAreAtSameDepth(tiledTree);
}

bool Test_SplitSchedule(TestArgs_t &args) {
  mlir::decisionforest::Schedule schedule(64, 1000);
  auto &t1 = schedule.NewIndexVariable("t1");
  auto &t2 = schedule.NewIndexVariable("t2");
  auto &t21 = schedule.NewIndexVariable("t21");
  auto &t22 = schedule.NewIndexVariable("t22");
  schedule.Tile(schedule.GetTreeIndex(), t1, t2, 4);
  decisionforest::Schedule::IndexVariableMapType indexMap, indexMap1;
  schedule.Split(t2, t21, t22, 2, indexMap1);
  auto &b1 = schedule.NewIndexVariable("b1");
  auto &b2 = schedule.NewIndexVariable("b2");
  auto &b21 = schedule.NewIndexVariable("b21");
  auto &b22 = schedule.NewIndexVariable("b22");
  schedule.Split(schedule.GetBatchIndex(), b1, b2, 32, indexMap);
  schedule.Split(b2, b21, b22, 48, indexMap);
  std::string dotFile = "/home/ashwin/mlir-build/llvm-project/mlir/examples/"
                        "tree-heavy/debug/temp/split_schedule.dot";
  schedule.WriteToDOTFile(dotFile);
  return true;
}


std::map<std::string, TestFunc_t> testFuncMap = {
    {"Test_TiledCodeGeneration_LeftHeavy_BatchSize1", Test_TiledCodeGeneration_LeftHeavy_BatchSize1},
    {"Test_TiledCodeGeneration_RightHeavy_BatchSize1", Test_TiledCodeGeneration_RightHeavy_BatchSize1},
    {"Test_TiledCodeGeneration_BalancedTree_BatchSize1", Test_TiledCodeGeneration_BalancedTree_BatchSize1},
    {"Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1", Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1},
    {"Test_TiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape", Test_TiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape},
    {"Test_TiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape", Test_TiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape},
    {"Test_TiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape", Test_TiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape},
    {"Test_TiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape", Test_TiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape},
    {"Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize", Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize},
    {"Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize", Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize},
    {"Test_UniformTiling_LeftHeavy_BatchSize1", Test_UniformTiling_LeftHeavy_BatchSize1},
    {"Test_UniformTiling_RightHeavy_BatchSize1", Test_UniformTiling_RightHeavy_BatchSize1},
    {"Test_UniformTiling_Balanced_BatchSize1", Test_UniformTiling_Balanced_BatchSize1},
    {"Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1", Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1},
    {"Test_UniformTiling_LeftHeavy_BatchSize1_Int8TileShape", Test_UniformTiling_LeftHeavy_BatchSize1_Int8TileShape},
    {"Test_UniformTiling_RightHeavy_BatchSize1_Int8TileShape", Test_UniformTiling_RightHeavy_BatchSize1_Int8TileShape},
    {"Test_UniformTiling_Balanced_BatchSize1_Int8TileShape", Test_UniformTiling_Balanced_BatchSize1_Int8TileShape},
    {"Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1_Int8TileShape", Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1_Int8TileShape},
    {"Test_UniformTiling_LeftHeavy_BatchSize1_Int16TileShape", Test_UniformTiling_LeftHeavy_BatchSize1_Int16TileShape},
    {"Test_UniformTiling_RightHeavy_BatchSize1_Int16TileShape", Test_UniformTiling_RightHeavy_BatchSize1_Int16TileShape},
    {"Test_UniformTiling_Balanced_BatchSize1_Int16TileShape", Test_UniformTiling_Balanced_BatchSize1_Int16TileShape},
    {"Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1_Int16TileShape", Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1_Int16TileShape},
    {"Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize1", Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize1},
    {"Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize1", Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize1},
    {"Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize1", Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize1},
    {"Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize2", Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize2},
    {"Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize2", Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize2},
    {"Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize2", Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize2},
    {"Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4", Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4},
    {"Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4", Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4},
    {"Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4", Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4},
    {"Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4_Int8TileShape", Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4_Int8TileShape},
    {"Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4_Int16TileShape", Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4_Int16TileShape},
    {"Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4_Int8TileShape", Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4_Int8TileShape},
    {"Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4_Int16TileShape", Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4_Int16TileShape},
    {"Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4_Int8TileShape", Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4_Int8TileShape},
    {"Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4_Int16TileShape", Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4_Int16TileShape},
    {"Test_Scalar_Abalone", Test_Scalar_Abalone},
    {"Test_TileSize2_Abalone", Test_TileSize2_Abalone},
    {"Test_TileSize3_Abalone", Test_TileSize3_Abalone},
    {"Test_TileSize4_Abalone", Test_TileSize4_Abalone},
    {"Test_TileSize8_Abalone", Test_TileSize8_Abalone},
    {"Test_Scalar_Airline", Test_Scalar_Airline},
    {"Test_TileSize2_Airline", Test_TileSize2_Airline},
    {"Test_TileSize3_Airline", Test_TileSize3_Airline},
    {"Test_TileSize4_Airline", Test_TileSize4_Airline},
    {"Test_TileSize8_Airline", Test_TileSize8_Airline},
    {"Test_Scalar_AirlineOHE", Test_Scalar_AirlineOHE},
    {"Test_TileSize2_AirlineOHE", Test_TileSize2_AirlineOHE},
    {"Test_TileSize3_AirlineOHE", Test_TileSize3_AirlineOHE},
    {"Test_TileSize4_AirlineOHE", Test_TileSize4_AirlineOHE},
    {"Test_TileSize8_AirlineOHE", Test_TileSize8_AirlineOHE},
    {"Test_Scalar_Bosch", Test_Scalar_Bosch},
    {"Test_TileSize2_Bosch", Test_TileSize2_Bosch},
    {"Test_TileSize3_Bosch", Test_TileSize3_Bosch},
    {"Test_TileSize4_Bosch", Test_TileSize4_Bosch},
    {"Test_TileSize8_Bosch", Test_TileSize8_Bosch},
    {"Test_Scalar_Epsilon", Test_Scalar_Epsilon},
    {"Test_TileSize2_Epsilon", Test_TileSize2_Epsilon},
    {"Test_TileSize3_Epsilon", Test_TileSize3_Epsilon},
    {"Test_TileSize4_Epsilon", Test_TileSize4_Epsilon},
    {"Test_TileSize8_Epsilon", Test_TileSize8_Epsilon},
    {"Test_Scalar_Higgs", Test_Scalar_Higgs},
    {"Test_TileSize2_Higgs", Test_TileSize2_Higgs},
    {"Test_TileSize3_Higgs", Test_TileSize3_Higgs},
    {"Test_TileSize4_Higgs", Test_TileSize4_Higgs},
    {"Test_TileSize8_Higgs", Test_TileSize8_Higgs},
    {"Test_TileSize1_Letters_Int8Type", Test_TileSize1_Letters_Int8Type},
    {"Test_TileSize2_Letters_Int8Type", Test_TileSize2_Letters_Int8Type},
    {"Test_TileSize3_Letters_Int8Type", Test_TileSize3_Letters_Int8Type},
    {"Test_TileSize4_Letters_Int8Type", Test_TileSize4_Letters_Int8Type},
    {"Test_TileSize8_Letters_Int8Type", Test_TileSize8_Letters_Int8Type},
    {"Test_Scalar_Year", Test_Scalar_Year},
    {"Test_TileSize2_Year", Test_TileSize2_Year},
    {"Test_TileSize3_Year", Test_TileSize3_Year},
    {"Test_TileSize4_Year", Test_TileSize4_Year},
    {"Test_TileSize8_Year", Test_TileSize8_Year},
    {"Test_TileSize1_CovType_Int8Type", Test_TileSize1_CovType_Int8Type},
    {"Test_TileSize2_CovType_Int8Type", Test_TileSize2_CovType_Int8Type},
    {"Test_TileSize3_CovType_Int8Type", Test_TileSize3_CovType_Int8Type},
    {"Test_TileSize4_CovType_Int8Type", Test_TileSize4_CovType_Int8Type},
    {"Test_TileSize8_CovType_Int8Type", Test_TileSize8_CovType_Int8Type},
    {"Test_SparseCodeGeneration_LeftHeavy_BatchSize1_I32ChildIdx", Test_SparseCodeGeneration_LeftHeavy_BatchSize1_I32ChildIdx},
    {"Test_SparseCodeGeneration_RightHeavy_BatchSize1_I32ChildIdx", Test_SparseCodeGeneration_RightHeavy_BatchSize1_I32ChildIdx},
    {"Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize1_I32ChildIdx", Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize1_I32ChildIdx},
    {"Test_SparseCodeGeneration_LeftHeavy_BatchSize2_I32ChildIdx", Test_SparseCodeGeneration_LeftHeavy_BatchSize2_I32ChildIdx},
    {"Test_SparseCodeGeneration_RightHeavy_BatchSize2_I32ChildIdx", Test_SparseCodeGeneration_RightHeavy_BatchSize2_I32ChildIdx},
    {"Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize2_I32ChildIdx", Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize2_I32ChildIdx},
    {"Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize1", Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize1},
    {"Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize2", Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize2},
    {"Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize4", Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize4},
    {"Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize1", Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize1},
    {"Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize2", Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize2},
    {"Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize4", Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize4},
    {"Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize1", Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize1},
    {"Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize2", Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize2},
    {"Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize4", Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize4},
    {"Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize1_Float", Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize1_Float},
    {"Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize2_Float", Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize2_Float},
    {"Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize4_Float", Test_Sparse_RandomXGBoostJSONs_1Tree_BatchSize4_Float},
    {"Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize1_Float", Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize1_Float},
    {"Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize2_Float", Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize2_Float},
    {"Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize4_Float", Test_Sparse_RandomXGBoostJSONs_2Trees_BatchSize4_Float},
    {"Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize1_Float", Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize1_Float},
    {"Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize2_Float", Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize2_Float},
    {"Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize4_Float", Test_Sparse_RandomXGBoostJSONs_4Trees_BatchSize4_Float},
    {"Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1", Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1},
    {"Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape", Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape},
    {"Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape", Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape},
    {"Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize", Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize},
    {"Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1", Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1},
    {"Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize", Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize},
    {"Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1", Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1},
    {"Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape", Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape},
    {"Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape", Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape},
    {"Test_SparseUniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4", Test_SparseUniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4},
    {"Test_SparseUniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4", Test_SparseUniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4},
    {"Test_SparseScalar_Abalone", Test_SparseScalar_Abalone},
    {"Test_SparseTileSize2_Abalone", Test_SparseTileSize2_Abalone},
    {"Test_SparseTileSize3_Abalone", Test_SparseTileSize3_Abalone},
    {"Test_SparseTileSize4_Abalone", Test_SparseTileSize4_Abalone},
    {"Test_SparseTileSize8_Abalone", Test_SparseTileSize8_Abalone},
    {"Test_SparseScalar_Airline", Test_SparseScalar_Airline},
    {"Test_SparseTileSize2_Airline", Test_SparseTileSize2_Airline},
    {"Test_SparseTileSize3_Airline", Test_SparseTileSize3_Airline},
    {"Test_SparseTileSize4_Airline", Test_SparseTileSize4_Airline},
    {"Test_SparseTileSize8_Airline", Test_SparseTileSize8_Airline},
    {"Test_SparseScalar_AirlineOHE", Test_SparseScalar_AirlineOHE},
    {"Test_SparseTileSize2_AirlineOHE", Test_SparseTileSize2_AirlineOHE},
    {"Test_SparseTileSize3_AirlineOHE", Test_SparseTileSize3_AirlineOHE},
    {"Test_SparseTileSize4_AirlineOHE", Test_SparseTileSize4_AirlineOHE},
    {"Test_SparseTileSize8_AirlineOHE", Test_SparseTileSize8_AirlineOHE},
    {"Test_SparseScalar_Bosch", Test_SparseScalar_Bosch},
    {"Test_SparseTileSize2_Bosch", Test_SparseTileSize2_Bosch},
    {"Test_SparseTileSize3_Bosch", Test_SparseTileSize3_Bosch},
    {"Test_SparseTileSize4_Bosch", Test_SparseTileSize4_Bosch},
    {"Test_SparseTileSize8_Bosch", Test_SparseTileSize8_Bosch},
    {"Test_SparseTileSize8_CovType_Int8Type", Test_SparseTileSize8_CovType_Int8Type},
    {"Test_SparseScalar_CovType_Int8Type", Test_SparseScalar_CovType_Int8Type},
    {"Test_SparseScalar_Letters_Int8Type", Test_SparseScalar_Letters_Int8Type},
    {"Test_SparseTileSize8_Letters_Int8Type", Test_SparseTileSize8_Letters_Int8Type},
    {"Test_SparseScalar_Epsilon", Test_SparseScalar_Epsilon},
    {"Test_SparseTileSize2_Epsilon", Test_SparseTileSize2_Epsilon},
    {"Test_SparseTileSize3_Epsilon", Test_SparseTileSize3_Epsilon},
    {"Test_SparseTileSize4_Epsilon", Test_SparseTileSize4_Epsilon},
    {"Test_SparseTileSize8_Epsilon", Test_SparseTileSize8_Epsilon},
    {"Test_SparseScalar_Higgs", Test_SparseScalar_Higgs},
    {"Test_SparseTileSize2_Higgs", Test_SparseTileSize2_Higgs},
    {"Test_SparseTileSize3_Higgs", Test_SparseTileSize3_Higgs},
    {"Test_SparseTileSize4_Higgs", Test_SparseTileSize4_Higgs},
    {"Test_SparseTileSize8_Higgs", Test_SparseTileSize8_Higgs},
    {"Test_SparseScalar_Year", Test_SparseScalar_Year},
    {"Test_SparseTileSize2_Year", Test_SparseTileSize2_Year},
    {"Test_SparseTileSize3_Year", Test_SparseTileSize3_Year},
    {"Test_SparseTileSize4_Year", Test_SparseTileSize4_Year},
    {"Test_SparseTileSize8_Year", Test_SparseTileSize8_Year},
    {"Test_TileSize8_Abalone_TestInputs", Test_TileSize8_Abalone_TestInputs},
    {"Test_TileSize8_Airline_TestInputs", Test_TileSize8_Airline_TestInputs},
    {"Test_TileSize8_AirlineOHE_TestInputs", Test_TileSize8_AirlineOHE_TestInputs},
    {"Test_TileSize8_Epsilon_TestInputs", Test_TileSize8_Epsilon_TestInputs},
    {"Test_TileSize8_Higgs_TestInputs", Test_TileSize8_Higgs_TestInputs},
    {"Test_TileSize8_Year_TestInputs", Test_TileSize8_Year_TestInputs},
    {"Test_TileSize8_CovType_TestInputs", Test_TileSize8_CovType_TestInputs},
    {"Test_CodeGeneration_LeftHeavy_BatchSize2_XGBoostSchedule", Test_CodeGeneration_LeftHeavy_BatchSize2_XGBoostSchedule},
    {"Test_CodeGeneration_RightHeavy_BatchSize2_XGBoostSchedule", Test_CodeGeneration_RightHeavy_BatchSize2_XGBoostSchedule},
    {"Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2_XGBoostSchedule", Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2_XGBoostSchedule},
    {"Test_CodeGeneration_LeftHeavy_BatchSize8_CacheInputSchedule", Test_CodeGeneration_LeftHeavy_BatchSize8_CacheInputSchedule},
    {"Test_CodeGeneration_RightHeavy_BatchSize2_CacheInputSchedule", Test_CodeGeneration_RightHeavy_BatchSize2_CacheInputSchedule},
    {"Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2_CacheInputSchedule", Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2_CacheInputSchedule},
    {"Test_Scalar_Abalone_OneTreeAtATimeSchedule", Test_Scalar_Abalone_OneTreeAtATimeSchedule},
    {"Test_TileSize2_Abalone_OneTreeAtATimeSchedule", Test_TileSize2_Abalone_OneTreeAtATimeSchedule},
    {"Test_TileSize3_Abalone_OneTreeAtATimeSchedule", Test_TileSize3_Abalone_OneTreeAtATimeSchedule},
    {"Test_TileSize4_Abalone_OneTreeAtATimeSchedule", Test_TileSize4_Abalone_OneTreeAtATimeSchedule},
    {"Test_TileSize8_Abalone_OneTreeAtATimeSchedule", Test_TileSize8_Abalone_OneTreeAtATimeSchedule},
    {"Test_Scalar_Airline_OneTreeAtATimeSchedule", Test_Scalar_Airline_OneTreeAtATimeSchedule},
    {"Test_TileSize2_Airline_OneTreeAtATimeSchedule", Test_TileSize2_Airline_OneTreeAtATimeSchedule},
    {"Test_TileSize3_Airline_OneTreeAtATimeSchedule", Test_TileSize3_Airline_OneTreeAtATimeSchedule},
    {"Test_TileSize4_Airline_OneTreeAtATimeSchedule", Test_TileSize4_Airline_OneTreeAtATimeSchedule},
    {"Test_TileSize8_Airline_OneTreeAtATimeSchedule", Test_TileSize8_Airline_OneTreeAtATimeSchedule},
    {"Test_Scalar_AirlineOHE_OneTreeAtATimeSchedule", Test_Scalar_AirlineOHE_OneTreeAtATimeSchedule},
    {"Test_TileSize2_AirlineOHE_OneTreeAtATimeSchedule", Test_TileSize2_AirlineOHE_OneTreeAtATimeSchedule},
    {"Test_TileSize3_AirlineOHE_OneTreeAtATimeSchedule", Test_TileSize3_AirlineOHE_OneTreeAtATimeSchedule},
    {"Test_TileSize4_AirlineOHE_OneTreeAtATimeSchedule", Test_TileSize4_AirlineOHE_OneTreeAtATimeSchedule},
    {"Test_TileSize8_AirlineOHE_OneTreeAtATimeSchedule", Test_TileSize8_AirlineOHE_OneTreeAtATimeSchedule},
    {"Test_Scalar_Bosch_OneTreeAtATimeSchedule", Test_Scalar_Bosch_OneTreeAtATimeSchedule},
    {"Test_TileSize2_Bosch_OneTreeAtATimeSchedule", Test_TileSize2_Bosch_OneTreeAtATimeSchedule},
    {"Test_TileSize3_Bosch_OneTreeAtATimeSchedule", Test_TileSize3_Bosch_OneTreeAtATimeSchedule},
    {"Test_TileSize4_Bosch_OneTreeAtATimeSchedule", Test_TileSize4_Bosch_OneTreeAtATimeSchedule},
    {"Test_TileSize8_Bosch_OneTreeAtATimeSchedule", Test_TileSize8_Bosch_OneTreeAtATimeSchedule},
    {"Test_Scalar_CovType_OneTreeAtATimeSchedule", Test_Scalar_CovType_OneTreeAtATimeSchedule},
    {"Test_TileSize8_CovType_OneTreeAtATimeSchedule", Test_TileSize8_CovType_OneTreeAtATimeSchedule},
    {"Test_Scalar_Epsilon_OneTreeAtATimeSchedule", Test_Scalar_Epsilon_OneTreeAtATimeSchedule},
    {"Test_TileSize2_Epsilon_OneTreeAtATimeSchedule", Test_TileSize2_Epsilon_OneTreeAtATimeSchedule},
    {"Test_TileSize3_Epsilon_OneTreeAtATimeSchedule", Test_TileSize3_Epsilon_OneTreeAtATimeSchedule},
    {"Test_TileSize4_Epsilon_OneTreeAtATimeSchedule", Test_TileSize4_Epsilon_OneTreeAtATimeSchedule},
    {"Test_TileSize8_Epsilon_OneTreeAtATimeSchedule", Test_TileSize8_Epsilon_OneTreeAtATimeSchedule},
    {"Test_Scalar_Higgs_OneTreeAtATimeSchedule", Test_Scalar_Higgs_OneTreeAtATimeSchedule},
    {"Test_TileSize2_Higgs_OneTreeAtATimeSchedule", Test_TileSize2_Higgs_OneTreeAtATimeSchedule},
    {"Test_TileSize3_Higgs_OneTreeAtATimeSchedule", Test_TileSize3_Higgs_OneTreeAtATimeSchedule},
    {"Test_TileSize4_Higgs_OneTreeAtATimeSchedule", Test_TileSize4_Higgs_OneTreeAtATimeSchedule},
    {"Test_TileSize8_Higgs_OneTreeAtATimeSchedule", Test_TileSize8_Higgs_OneTreeAtATimeSchedule},
    {"Test_Scalar_Year_OneTreeAtATimeSchedule", Test_Scalar_Year_OneTreeAtATimeSchedule},
    {"Test_TileSize2_Year_OneTreeAtATimeSchedule", Test_TileSize2_Year_OneTreeAtATimeSchedule},
    {"Test_TileSize3_Year_OneTreeAtATimeSchedule", Test_TileSize3_Year_OneTreeAtATimeSchedule},
    {"Test_TileSize4_Year_OneTreeAtATimeSchedule", Test_TileSize4_Year_OneTreeAtATimeSchedule},
    {"Test_TileSize8_Year_OneTreeAtATimeSchedule", Test_TileSize8_Year_OneTreeAtATimeSchedule},
    {"Test_TileSize8_Abalone_TestInputs_TiledSchedule", Test_TileSize8_Abalone_TestInputs_TiledSchedule},
    {"Test_TileSize8_AirlineOHE_TestInputs_TiledSchedule", Test_TileSize8_AirlineOHE_TestInputs_TiledSchedule},
    {"Test_TileSize8_Airline_TestInputs_TiledSchedule", Test_TileSize8_Airline_TestInputs_TiledSchedule},
    {"Test_TileSize8_Epsilon_TestInputs_TiledSchedule", Test_TileSize8_Epsilon_TestInputs_TiledSchedule},
    {"Test_TileSize8_Higgs_TestInputs_TiledSchedule", Test_TileSize8_Higgs_TestInputs_TiledSchedule},
    {"Test_TileSize8_Year_TestInputs_TiledSchedule", Test_TileSize8_Year_TestInputs_TiledSchedule},
    {"Test_SparseTileSize8_Letters_TiledSchedule", Test_SparseTileSize8_Letters_TiledSchedule},

        // Sparse code gen tests with loops interchanged
    {"Test_SparseScalar_Abalone_OneTreeAtATimeSchedule", Test_SparseScalar_Abalone_OneTreeAtATimeSchedule},
    {"Test_SparseTileSize8_Abalone_OneTreeAtATimeSchedule", Test_SparseTileSize8_Abalone_OneTreeAtATimeSchedule},
    {"Test_SparseScalar_Airline_OneTreeAtATimeSchedule", Test_SparseScalar_Airline_OneTreeAtATimeSchedule},
    {"Test_SparseTileSize8_Airline_OneTreeAtATimeSchedule", Test_SparseTileSize8_Airline_OneTreeAtATimeSchedule},
    {"Test_SparseScalar_AirlineOHE_OneTreeAtATimeSchedule", Test_SparseScalar_AirlineOHE_OneTreeAtATimeSchedule},
    {"Test_SparseTileSize8_AirlineOHE_OneTreeAtATimeSchedule", Test_SparseTileSize8_AirlineOHE_OneTreeAtATimeSchedule},
    {"Test_SparseScalar_Bosch_OneTreeAtATimeSchedule", Test_SparseScalar_Bosch_OneTreeAtATimeSchedule},
    {"Test_SparseTileSize8_Bosch_OneTreeAtATimeSchedule", Test_SparseTileSize8_Bosch_OneTreeAtATimeSchedule},
    {"Test_SparseScalar_CovType_OneTreeAtATimeSchedule", Test_SparseScalar_CovType_OneTreeAtATimeSchedule},
    {"Test_SparseTileSize8_CovType_OneTreeAtATimeSchedule", Test_SparseTileSize8_CovType_OneTreeAtATimeSchedule},
    {"Test_SparseScalar_Letters_OneTreeAtATimeSchedule", Test_SparseScalar_Letters_OneTreeAtATimeSchedule},
    {"Test_SparseTileSize8_Letters_OneTreeAtATimeSchedule", Test_SparseTileSize8_Letters_OneTreeAtATimeSchedule},
    {"Test_SparseTileSize8_Letters_TiledSchedule", Test_SparseTileSize8_Letters_TiledSchedule},
    {"Test_SparseScalar_Epsilon_OneTreeAtATimeSchedule", Test_SparseScalar_Epsilon_OneTreeAtATimeSchedule},
    {"Test_SparseTileSize8_Epsilon_OneTreeAtATimeSchedule", Test_SparseTileSize8_Epsilon_OneTreeAtATimeSchedule},
    {"Test_SparseScalar_Higgs_OneTreeAtATimeSchedule", Test_SparseScalar_Higgs_OneTreeAtATimeSchedule},
    {"Test_SparseTileSize8_Higgs_OneTreeAtATimeSchedule", Test_SparseTileSize8_Higgs_OneTreeAtATimeSchedule},
    {"Test_SparseScalar_Year_OneTreeAtATimeSchedule", Test_SparseScalar_Year_OneTreeAtATimeSchedule},
    {"Test_SparseTileSize8_Year_OneTreeAtATimeSchedule", Test_SparseTileSize8_Year_OneTreeAtATimeSchedule},

    // Sparse code gen tests with loops tiled
    {"Test_SparseTileSize8_Abalone_TestInputs_TiledSchedule", Test_SparseTileSize8_Abalone_TestInputs_TiledSchedule},
    {"Test_SparseTileSize8_AirlineOHE_TestInputs_TiledSchedule", Test_SparseTileSize8_AirlineOHE_TestInputs_TiledSchedule},
    {"Test_SparseTileSize8_Airline_TestInputs_TiledSchedule", Test_SparseTileSize8_Airline_TestInputs_TiledSchedule},
    {"Test_SparseTileSize8_Epsilon_TestInputs_TiledSchedule", Test_SparseTileSize8_Epsilon_TestInputs_TiledSchedule},
    {"Test_SparseTileSize8_Higgs_TestInputs_TiledSchedule", Test_SparseTileSize8_Higgs_TestInputs_TiledSchedule},
    {"Test_SparseTileSize8_Year_TestInputs_TiledSchedule", Test_SparseTileSize8_Year_TestInputs_TiledSchedule},

    // Stats tests
    {"Test_AbaloneStatGenerationAndReading", Test_AbaloneStatGenerationAndReading},
    {"Test_AirlineStatGenerationAndReading", Test_AirlineStatGenerationAndReading},
    {"Test_AirlineOHEStatGenerationAndReading", Test_AirlineOHEStatGenerationAndReading},
    {"Test_CovtypeStatGenerationAndReading", Test_CovtypeStatGenerationAndReading},
    {"Test_EpsilonStatGenerationAndReading", Test_EpsilonStatGenerationAndReading},
    {"Test_HiggsStatGenerationAndReading", Test_HiggsStatGenerationAndReading},
    {"Test_YearStatGenerationAndReading", Test_YearStatGenerationAndReading},

    // Sparse Probabilistic Tiling Tests
    {"Test_SparseProbabilisticTiling_TileSize8_Abalone", Test_SparseProbabilisticTiling_TileSize8_Abalone},
    {"Test_SparseProbabilisticTiling_TileSize8_Airline", Test_SparseProbabilisticTiling_TileSize8_Airline},
    {"Test_SparseProbabilisticTiling_TileSize8_AirlineOHE", Test_SparseProbabilisticTiling_TileSize8_AirlineOHE},
    {"Test_SparseProbabilisticTiling_TileSize8_Covtype", Test_SparseProbabilisticTiling_TileSize8_Covtype},
    {"Test_SparseProbabilisticTiling_TileSize8_Epsilon", Test_SparseProbabilisticTiling_TileSize8_Epsilon},
    {"Test_SparseProbabilisticTiling_TileSize8_Higgs", Test_SparseProbabilisticTiling_TileSize8_Higgs},
    {"Test_SparseProbabilisticTiling_TileSize8_Year", Test_SparseProbabilisticTiling_TileSize8_Year},

    // Tiled tree padding tests
    {"Test_PadTiledTree_BalancedTree_TileSize2", Test_PadTiledTree_BalancedTree_TileSize2},
    {"Test_PadTiledTree_BalancedTree_TileSize2_2", Test_PadTiledTree_BalancedTree_TileSize2_2},
    {"Test_PadTiledTree_BalancedTree_TileSize3", Test_PadTiledTree_BalancedTree_TileSize3},
    {"Test_RandomXGBoostJSONs_1Tree_BatchSize4_EqualDepth_TileSize8", Test_RandomXGBoostJSONs_1Tree_BatchSize4_EqualDepth_TileSize8},
    {"Test_RandomXGBoostJSONs_2Trees_BatchSize4_EqualDepth_TileSize8", Test_RandomXGBoostJSONs_2Trees_BatchSize4_EqualDepth_TileSize8},
    {"Test_RandomXGBoostJSONs_4Trees_BatchSize4_EqualDepth_TileSize8", Test_RandomXGBoostJSONs_4Trees_BatchSize4_EqualDepth_TileSize8},
    {"Test_TileSize8_Abalone_TestInputs_MakeLeavesSameDepth", Test_TileSize8_Abalone_TestInputs_MakeLeavesSameDepth},
    {"Test_TileSize8_AirlineOHE_TestInputs_MakeLeavesSameDepth", Test_TileSize8_AirlineOHE_TestInputs_MakeLeavesSameDepth},
    {"Test_TileSize8_Airline_TestInputs_MakeLeavesSameDepth", Test_TileSize8_Airline_TestInputs_MakeLeavesSameDepth},
    {"Test_TileSize8_Epsilon_TestInputs_MakeLeavesSameDepth", Test_TileSize8_Epsilon_TestInputs_MakeLeavesSameDepth},
    {"Test_TileSize8_Higgs_TestInputs_MakeLeavesSameDepth", Test_TileSize8_Higgs_TestInputs_MakeLeavesSameDepth},
    {"Test_TileSize8_Year_TestInputs_MakeLeavesSameDepth", Test_TileSize8_Year_TestInputs_MakeLeavesSameDepth},
    {"Test_TileSize8_CovType_TestInputs_MakeLeavesSameDepth", Test_TileSize8_CovType_TestInputs_MakeLeavesSameDepth},
    {"Test_TileSize8_Abalone_TestInputs_ReorderTrees", Test_TileSize8_Abalone_TestInputs_ReorderTrees},

    // // Split Schedule
    {"Test_TileSize8_Abalone_TestInputs_SwapAndSplitTreeIndex", Test_TileSize8_Abalone_TestInputs_SwapAndSplitTreeIndex},
    {"Test_TileSize8_AirlineOHE_TestInputs_SwapAndSplitTreeIndex", Test_TileSize8_AirlineOHE_TestInputs_SwapAndSplitTreeIndex},
    {"Test_TileSize8_Airline_TestInputs_SwapAndSplitTreeIndex", Test_TileSize8_Airline_TestInputs_SwapAndSplitTreeIndex},
    {"Test_TileSize8_Epsilon_TestInputs_SwapAndSplitTreeIndex", Test_TileSize8_Epsilon_TestInputs_SwapAndSplitTreeIndex},
    {"Test_TileSize8_Higgs_TestInputs_SwapAndSplitTreeIndex", Test_TileSize8_Higgs_TestInputs_SwapAndSplitTreeIndex},
    {"Test_TileSize8_Year_TestInputs_SwapAndSplitTreeIndex", Test_TileSize8_Year_TestInputs_SwapAndSplitTreeIndex},

#ifdef OMP_SUPPORT
    {"Test_TileSize8_Abalone_TestInputs_ParallelBatch", Test_TileSize8_Abalone_TestInputs_ParallelBatch},
    {"Test_TileSize8_Airline_TestInputs_ParallelBatch", Test_TileSize8_Airline_TestInputs_ParallelBatch},
    {"Test_TileSize8_AirlineOHE_TestInputs_ParallelBatch", Test_TileSize8_AirlineOHE_TestInputs_ParallelBatch},
    {"Test_TileSize8_Covtype_TestInputs_ParallelBatch", Test_TileSize8_Covtype_TestInputs_ParallelBatch},
    {"Test_TileSize8_Letters_TestInputs_ParallelBatch", Test_TileSize8_Letters_TestInputs_ParallelBatch},
    {"Test_TileSize8_Epsilon_TestInputs_ParallelBatch", Test_TileSize8_Epsilon_TestInputs_ParallelBatch},
    {"Test_TileSize8_Higgs_TestInputs_ParallelBatch", Test_TileSize8_Higgs_TestInputs_ParallelBatch},
    {"Test_TileSize8_Year_TestInputs_ParallelBatch", Test_TileSize8_Year_TestInputs_ParallelBatch},
#endif // OMP_SUPPORT

  // Pipelining + Unrolling tests
    {"Test_RandomXGBoostJSONs_1Tree_BatchSize8_TileSize2_4Pipelined", Test_RandomXGBoostJSONs_1Tree_BatchSize8_TileSize2_4Pipelined},
    {"Test_RandomXGBoostJSONs_4Trees_BatchSize4_4Pipelined", Test_RandomXGBoostJSONs_4Trees_BatchSize4_4Pipelined},
    {"Test_TileSize3_Letters_2Pipelined_Int8Type", Test_TileSize3_Letters_2Pipelined_Int8Type},
    {"Test_TileSize4_Letters_3Pipelined_Int8Type", Test_TileSize4_Letters_3Pipelined_Int8Type},
    {"Test_TileSize8_Letters_5Pipelined_Int8Type", Test_TileSize8_Letters_5Pipelined_Int8Type},
    {"Test_SparseTileSize8_4Pipelined_Bosch", Test_SparseTileSize8_4Pipelined_Bosch},
    {"Test_TileSize8_Abalone_4Pipelined_TestInputs", Test_TileSize8_Abalone_4Pipelined_TestInputs},
    {"Test_TileSize8_CovType_4Pipelined_TestInputs", Test_TileSize8_CovType_4Pipelined_TestInputs},
    {"Test_SparseTileSize8_Pipeline4_Airline", Test_SparseTileSize8_Pipeline4_Airline},
    {"Test_SparseTileSize8_Pipelined4_AirlineOHE", Test_SparseTileSize8_Pipelined4_AirlineOHE},
    {"Test_SparseTileSize8_Pipelined_Year", Test_SparseTileSize8_Pipelined_Year},
    {"Test_SparseTileSize8_Pipelined_Higgs", Test_SparseTileSize8_Pipelined_Higgs},
    {"Test_SparseTileSize8_Pipelined_Epsilon", Test_SparseTileSize8_Pipelined_Epsilon},

    {"Test_TileSize8_Abalone_4PipelinedTrees_TestInputs", Test_TileSize8_Abalone_4PipelinedTrees_TestInputs},
    {"Test_TileSize8_Abalone_PipelinedTreesPeeling_TestInputs", Test_TileSize8_Abalone_PipelinedTreesPeeling_TestInputs},

    // Hybrid Tiling
    {"Test_WalkPeeling_BalancedTree_TileSize2", Test_WalkPeeling_BalancedTree_TileSize2},
    {"Test_HybridTilingAndPeeling_RandomXGBoostJSONs_4Tree_FloatBatchSize4", Test_HybridTilingAndPeeling_RandomXGBoostJSONs_4Tree_FloatBatchSize4},
    {"Test_HybridTilingAndPeeling_RandomXGBoostJSONs_1Tree_FloatBatchSize4", Test_HybridTilingAndPeeling_RandomXGBoostJSONs_1Tree_FloatBatchSize4},
    {"Test_HybridTilingAndPeeling_RandomXGBoostJSONs_2Tree_FloatBatchSize4", Test_HybridTilingAndPeeling_RandomXGBoostJSONs_2Tree_FloatBatchSize4},
    {"Test_UniformAndHybridTilingAndPeeling_RandomXGBoostJSONs_2Tree_FloatBatchSize4", Test_UniformAndHybridTilingAndPeeling_RandomXGBoostJSONs_2Tree_FloatBatchSize4},
    {"Test_UniformAndHybridTilingAndPeeling_RandomXGBoostJSONs_4Tree_FloatBatchSize4", Test_UniformAndHybridTilingAndPeeling_RandomXGBoostJSONs_4Tree_FloatBatchSize4},
    {"Test_PeeledHybridProbabilisticTiling_TileSize8_Year", Test_PeeledHybridProbabilisticTiling_TileSize8_Year},
    {"Test_PeeledHybridProbabilisticTiling_TileSize8_Letters", Test_PeeledHybridProbabilisticTiling_TileSize8_Letters},
    {"Test_PeeledHybridProbabilisticTiling_TileSize8_Epsilon", Test_PeeledHybridProbabilisticTiling_TileSize8_Epsilon},
    {"Test_PeeledHybridProbabilisticTiling_TileSize8_Higgs", Test_PeeledHybridProbabilisticTiling_TileSize8_Higgs},
    {"Test_PeeledHybridProbabilisticTiling_TileSize8_AirlineOHE", Test_PeeledHybridProbabilisticTiling_TileSize8_AirlineOHE},
    {"Test_PeeledHybridProbabilisticTiling_TileSize8_Covtype", Test_PeeledHybridProbabilisticTiling_TileSize8_Covtype},
    {"Test_PeeledHybridProbabilisticTiling_TileSize8_Airline", Test_PeeledHybridProbabilisticTiling_TileSize8_Airline},
    {"Test_PeeledHybridProbabilisticTiling_TileSize8_Abalone", Test_PeeledHybridProbabilisticTiling_TileSize8_Abalone},

    #ifdef TREEBEARD_GPU_SUPPORT
   // GPU model buffer initialization tests (scalar)
    {"Test_GPUModelInit_LeftHeavy_Scalar_DoubleInt", Test_GPUModelInit_LeftHeavy_Scalar_DoubleInt},
    {"Test_GPUModelInit_RightHeavy_Scalar_DoubleInt", Test_GPUModelInit_RightHeavy_Scalar_DoubleInt},
    {"Test_GPUModelInit_Balanced_Scalar_DoubleInt", Test_GPUModelInit_Balanced_Scalar_DoubleInt},
    {"Test_GPUModelInit_LeftAndRightHeavy_Scalar_DoubleInt", Test_GPUModelInit_LeftAndRightHeavy_Scalar_DoubleInt},
    {"Test_GPUModelInit_LeftHeavy_Scalar_FloatInt", Test_GPUModelInit_LeftHeavy_Scalar_FloatInt},
    {"Test_GPUModelInit_RightHeavy_Scalar_FloatInt", Test_GPUModelInit_RightHeavy_Scalar_FloatInt},
    {"Test_GPUModelInit_Balanced_Scalar_FloatInt", Test_GPUModelInit_Balanced_Scalar_FloatInt},
    {"Test_GPUModelInit_LeftAndRightHeavy_Scalar_FloatInt", Test_GPUModelInit_LeftAndRightHeavy_Scalar_FloatInt},
    {"Test_GPUModelInit_LeftHeavy_Scalar_FloatInt16", Test_GPUModelInit_LeftHeavy_Scalar_FloatInt16},
    {"Test_GPUModelInit_RightHeavy_Scalar_FloatInt16", Test_GPUModelInit_RightHeavy_Scalar_FloatInt16},
    {"Test_GPUModelInit_Balanced_Scalar_FloatInt16", Test_GPUModelInit_Balanced_Scalar_FloatInt16},
    {"Test_GPUModelInit_LeftAndRightHeavy_Scalar_FloatInt16", Test_GPUModelInit_LeftAndRightHeavy_Scalar_FloatInt16},

    {"Test_GPUModelInit_LeftHeavy_Reorg_DoubleInt", Test_GPUModelInit_LeftHeavy_Reorg_DoubleInt},
    {"Test_GPUModelInit_RightHeavy_Reorg_DoubleInt", Test_GPUModelInit_RightHeavy_Reorg_DoubleInt},
    {"Test_GPUModelInit_Balanced_Reorg_DoubleInt", Test_GPUModelInit_Balanced_Reorg_DoubleInt},
    {"Test_GPUModelInit_LeftAndRightHeavy_Reorg_DoubleInt", Test_GPUModelInit_LeftAndRightHeavy_Reorg_DoubleInt},
    {"Test_GPUModelInit_LeftHeavy_Reorg_FloatInt", Test_GPUModelInit_LeftHeavy_Reorg_FloatInt},
    {"Test_GPUModelInit_RightHeavy_Reorg_FloatInt", Test_GPUModelInit_RightHeavy_Reorg_FloatInt},
    {"Test_GPUModelInit_Balanced_Reorg_FloatInt", Test_GPUModelInit_Balanced_Reorg_FloatInt},
    {"Test_GPUModelInit_LeftAndRightHeavy_Reorg_FloatInt", Test_GPUModelInit_LeftAndRightHeavy_Reorg_FloatInt},
    {"Test_GPUModelInit_LeftHeavy_Reorg_FloatInt16", Test_GPUModelInit_LeftHeavy_Reorg_FloatInt16},
    {"Test_GPUModelInit_RightHeavy_Reorg_FloatInt16", Test_GPUModelInit_RightHeavy_Reorg_FloatInt16},
    {"Test_GPUModelInit_Balanced_Reorg_FloatInt16", Test_GPUModelInit_Balanced_Reorg_FloatInt16},
    {"Test_GPUModelInit_LeftAndRightHeavy_Reorg_FloatInt16", Test_GPUModelInit_LeftAndRightHeavy_Reorg_FloatInt16},

    // Basic Array Scalar GPU Codegen Tests
    {"Test_GPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32", Test_GPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32},
    {"Test_GPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32", Test_GPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32},
    {"Test_GPUCodeGeneration_Balanced_DoubleInt32_BatchSize32", Test_GPUCodeGeneration_Balanced_DoubleInt32_BatchSize32},
    {"Test_GPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32", Test_GPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32},
    {"Test_GPUCodeGeneration_LeftHeavy_FloatInt16_BatchSize32", Test_GPUCodeGeneration_LeftHeavy_FloatInt16_BatchSize32},
    {"Test_GPUCodeGeneration_RightHeavy_FloatInt16_BatchSize32", Test_GPUCodeGeneration_RightHeavy_FloatInt16_BatchSize32},
    {"Test_GPUCodeGeneration_Balanced_FloatInt16_BatchSize32", Test_GPUCodeGeneration_Balanced_FloatInt16_BatchSize32},
    {"Test_GPUCodeGeneration_LeftAndRightHeavy_FloatInt16_BatchSize32", Test_GPUCodeGeneration_LeftAndRightHeavy_FloatInt16_BatchSize32},

    // Basic scalar sparse GPU codegen tests
    {"Test_SparseGPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32", Test_SparseGPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32},
    {"Test_SparseGPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32", Test_SparseGPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32},
    {"Test_SparseGPUCodeGeneration_Balanced_DoubleInt32_BatchSize32", Test_SparseGPUCodeGeneration_Balanced_DoubleInt32_BatchSize32},
    {"Test_SparseGPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32", Test_SparseGPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32},
    {"Test_SparseGPUCodeGeneration_LeftHeavy_FloatInt16_ChI16_BatchSize32", Test_SparseGPUCodeGeneration_LeftHeavy_FloatInt16_ChI16_BatchSize32},
    {"Test_SparseGPUCodeGeneration_RightHeavy_FloatInt16_ChI16_BatchSize32", Test_SparseGPUCodeGeneration_RightHeavy_FloatInt16_ChI16_BatchSize32},
    {"Test_SparseGPUCodeGeneration_Balanced_FloatInt16_ChI16_BatchSize32", Test_SparseGPUCodeGeneration_Balanced_FloatInt16_ChI16_BatchSize32},
    {"Test_SparseGPUCodeGeneration_LeftAndRightHeavy_FloatInt16_ChI16_BatchSize32", Test_SparseGPUCodeGeneration_LeftAndRightHeavy_FloatInt16_ChI16_BatchSize32},

    // Basic reorg forest tests
    {"Test_ReorgGPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32", Test_ReorgGPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32},
    {"Test_ReorgGPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32", Test_ReorgGPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32},
    {"Test_ReorgGPUCodeGeneration_Balanced_DoubleInt32_BatchSize32", Test_ReorgGPUCodeGeneration_Balanced_DoubleInt32_BatchSize32},
    {"Test_ReorgGPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32", Test_ReorgGPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32},
    {"Test_ReorgGPUCodeGeneration_LeftHeavy_FloatInt16_BatchSize32", Test_ReorgGPUCodeGeneration_LeftHeavy_FloatInt16_BatchSize32},
    {"Test_ReorgGPUCodeGeneration_RightHeavy_FloatInt16_BatchSize32", Test_ReorgGPUCodeGeneration_RightHeavy_FloatInt16_BatchSize32},
    {"Test_ReorgGPUCodeGeneration_LeftAndRightHeavy_FloatInt16_BatchSize32", Test_ReorgGPUCodeGeneration_LeftAndRightHeavy_FloatInt16_BatchSize32},
    {"Test_ReorgGPUCodeGeneration_LeftRightAndBalanced_FloatInt16_BatchSize32", Test_ReorgGPUCodeGeneration_LeftRightAndBalanced_FloatInt16_BatchSize32},

    // Basic GPU caching tests
    {"Test_SimpleSharedMem_LeftRightAndBalanced", Test_SimpleSharedMem_LeftRightAndBalanced},
    {"Test_SimpleSharedMem_LeftHeavy", Test_SimpleSharedMem_LeftHeavy},
    {"Test_SimpleSharedMem_LeftHeavy_F32I16", Test_SimpleSharedMem_LeftHeavy_F32I16},
    {"Test_SimpleSharedMem_LeftRightAndBalanced_F32I16", Test_SimpleSharedMem_LeftRightAndBalanced_F32I16},

    {"Test_SimpleSharedMem_LeftHeavy_ReorgRep", Test_SimpleSharedMem_LeftHeavy_ReorgRep},
    {"Test_SimpleSharedMem_LeftRightAndBalanced_Reorg", Test_SimpleSharedMem_LeftRightAndBalanced_Reorg},
    {"Test_SimpleSharedMem_LeftHeavy_ReorgRep_F32I16", Test_SimpleSharedMem_LeftHeavy_ReorgRep_F32I16},
    {"Test_SimpleSharedMem_LeftRightAndBalanced_Reorg_F32I16", Test_SimpleSharedMem_LeftRightAndBalanced_Reorg_F32I16},

    {"Test_SimpleSharedMem_LeftHeavy_SparseRep", Test_SimpleSharedMem_LeftHeavy_SparseRep},
    {"Test_SimpleSharedMem_LeftRightAndBalanced_SparseRep", Test_SimpleSharedMem_LeftRightAndBalanced_SparseRep},
    {"Test_SimpleSharedMem_LeftHeavy_SparseRep_F32I16", Test_SimpleSharedMem_LeftHeavy_SparseRep_F32I16},
    {"Test_SimpleSharedMem_LeftRightAndBalanced_SparseRep_F32I16", Test_SimpleSharedMem_LeftRightAndBalanced_SparseRep_F32I16},
    {"Test_InputSharedMem_LeftHeavy", Test_InputSharedMem_LeftHeavy},
    {"Test_InputSharedMem_RightHeavy", Test_InputSharedMem_RightHeavy},
    {"Test_InputSharedMem_LeftRightAndBalanced", Test_InputSharedMem_LeftRightAndBalanced},

    // Simple GPU Tiling tests
    {"Test_TiledSparseGPU_LeftHeavy_DblI32_B32_TSz2", Test_TiledSparseGPU_LeftHeavy_DblI32_B32_TSz2},
    {"Test_TiledSparseGPU_RightHeavy_DblI32_B32_TSz2", Test_TiledSparseGPU_RightHeavy_DblI32_B32_TSz2},
    {"Test_TiledSparseGPU_Balanced_DblI32_B32_TSz2", Test_TiledSparseGPU_Balanced_DblI32_B32_TSz2},
    {"Test_TiledSparseGPU_LeftAndRightHeavy_DblI32_B32_TSz2", Test_TiledSparseGPU_LeftAndRightHeavy_DblI32_B32_TSz2},
    {"Test_TiledSparseGPU_LeftHeavy_FltI16_B32_TSz2", Test_TiledSparseGPU_LeftHeavy_FltI16_B32_TSz2},
    {"Test_TiledSparseGPU_RightHeavy_FltI16_B32_TSz2", Test_TiledSparseGPU_RightHeavy_FltI16_B32_TSz2},
    {"Test_TiledSparseGPU_Balanced_FltI16_B32_TSz2", Test_TiledSparseGPU_Balanced_FltI16_B32_TSz2},
    {"Test_TiledSparseGPU_LeftAndRightHeavy_FltI16_B32_TSz2", Test_TiledSparseGPU_LeftAndRightHeavy_FltI16_B32_TSz2},

    {"Test_TiledArrayGPU_LeftHeavy_DblI32_B32_TSz2", Test_TiledArrayGPU_LeftHeavy_DblI32_B32_TSz2},
    {"Test_TiledArrayGPU_RightHeavy_DblI32_B32_TSz2", Test_TiledArrayGPU_RightHeavy_DblI32_B32_TSz2},
    {"Test_TiledArrayGPU_Balanced_DblI32_B32_TSz2", Test_TiledArrayGPU_Balanced_DblI32_B32_TSz2},
    {"Test_TiledArrayGPU_LeftAndRightHeavy_DblI32_B32_TSz2", Test_TiledArrayGPU_LeftAndRightHeavy_DblI32_B32_TSz2},
    {"Test_TiledArrayGPU_LeftHeavy_FltI16_B32_TSz2", Test_TiledArrayGPU_LeftHeavy_FltI16_B32_TSz2},
    {"Test_TiledArrayGPU_RightHeavy_FltI16_B32_TSz2", Test_TiledArrayGPU_RightHeavy_FltI16_B32_TSz2},
    {"Test_TiledArrayGPU_Balanced_FltI16_B32_TSz2", Test_TiledArrayGPU_Balanced_FltI16_B32_TSz2},
    {"Test_TiledArrayGPU_LeftAndRightHeavy_FltI16_B32_TSz2", Test_TiledArrayGPU_LeftAndRightHeavy_FltI16_B32_TSz2},

    // Tiling + Caching
    {"Test_TiledCachedArrayGPU_LeftHeavy_DblI32_B32_TSz2", Test_TiledCachedArrayGPU_LeftHeavy_DblI32_B32_TSz2},
    {"Test_TiledCachedArrayGPU_RightHeavy_DblI32_B32_TSz2", Test_TiledCachedArrayGPU_RightHeavy_DblI32_B32_TSz2},
    {"Test_TiledCachedArrayGPU_Balanced_DblI32_B32_TSz2", Test_TiledCachedArrayGPU_Balanced_DblI32_B32_TSz2},
    {"Test_TiledCachedArrayGPU_LeftAndRightHeavy_DblI32_B32_TSz2", Test_TiledCachedArrayGPU_LeftAndRightHeavy_DblI32_B32_TSz2},
    {"Test_TiledCachedArrayGPU_LeftRightAndBalanced_DblI32_B32_TSz2", Test_TiledCachedArrayGPU_LeftRightAndBalanced_DblI32_B32_TSz2},
    {"Test_TiledCachedArrayGPU_LeftHeavy_FltI16_B32_TSz2", Test_TiledCachedArrayGPU_LeftHeavy_FltI16_B32_TSz2},
    {"Test_TiledCachedArrayGPU_RightHeavy_FltI16_B32_TSz2", Test_TiledCachedArrayGPU_RightHeavy_FltI16_B32_TSz2},
    {"Test_TiledCachedArrayGPU_Balanced_FltI16_B32_TSz2", Test_TiledCachedArrayGPU_Balanced_FltI16_B32_TSz2},
    {"Test_TiledCachedArrayGPU_LeftAndRightHeavy_FltI16_B32_TSz2", Test_TiledCachedArrayGPU_LeftAndRightHeavy_FltI16_B32_TSz2},
    {"Test_TiledCachedArrayGPU_LeftRightAndBalanced_FltI16_B32_TSz2", Test_TiledCachedArrayGPU_LeftRightAndBalanced_FltI16_B32_TSz2},

    {"Test_TiledCachedSparseGPU_LeftHeavy_DblI32_B32_TSz2", Test_TiledCachedSparseGPU_LeftHeavy_DblI32_B32_TSz2},
    {"Test_TiledCachedSparseGPU_RightHeavy_DblI32_B32_TSz2", Test_TiledCachedSparseGPU_RightHeavy_DblI32_B32_TSz2},
    {"Test_TiledCachedSparseGPU_Balanced_DblI32_B32_TSz2", Test_TiledCachedSparseGPU_Balanced_DblI32_B32_TSz2},
    {"Test_TiledCachedSparseGPU_LeftAndRightHeavy_DblI32_B32_TSz2", Test_TiledCachedSparseGPU_LeftAndRightHeavy_DblI32_B32_TSz2},
    {"Test_TiledCachedSparseGPU_LeftHeavy_FltI16_B32_TSz2", Test_TiledCachedSparseGPU_LeftHeavy_FltI16_B32_TSz2},
    {"Test_TiledCachedSparseGPU_RightHeavy_FltI16_B32_TSz2", Test_TiledCachedSparseGPU_RightHeavy_FltI16_B32_TSz2},
    {"Test_TiledCachedSparseGPU_Balanced_FltI16_B32_TSz2", Test_TiledCachedSparseGPU_Balanced_FltI16_B32_TSz2},
    {"Test_TiledCachedSparseGPU_LeftAndRightHeavy_FltI16_B32_TSz2", Test_TiledCachedSparseGPU_LeftAndRightHeavy_FltI16_B32_TSz2},
    {"Test_TiledCachedSparseGPU_LeftRightAndBalanced_DblI32_B32_TSz2", Test_TiledCachedSparseGPU_LeftRightAndBalanced_DblI32_B32_TSz2},
    {"Test_TiledCachedSparseGPU_LeftRightAndBalanced_FltI16_B32_TSz2", Test_TiledCachedSparseGPU_LeftRightAndBalanced_FltI16_B32_TSz2},

    // GPU Synthetic XGB tests -- scalar
    {"Test_GPU_1TreeXGB_Array_Scalar", Test_GPU_1TreeXGB_Array_Scalar},
    {"Test_GPU_2TreeXGB_Array_Scalar", Test_GPU_2TreeXGB_Array_Scalar},
    {"Test_GPU_4TreeXGB_Array_Scalar", Test_GPU_4TreeXGB_Array_Scalar},
    {"Test_GPU_1TreeXGB_Array_Scalar_f32i16", Test_GPU_1TreeXGB_Array_Scalar_f32i16},
    {"Test_GPU_2TreeXGB_Array_Scalar_f32i16", Test_GPU_2TreeXGB_Array_Scalar_f32i16},
    {"Test_GPU_4TreeXGB_Array_Scalar_f32i16", Test_GPU_4TreeXGB_Array_Scalar_f32i16},
    {"Test_GPU_1TreeXGB_Sparse_Scalar", Test_GPU_1TreeXGB_Sparse_Scalar},
    {"Test_GPU_2TreeXGB_Sparse_Scalar", Test_GPU_2TreeXGB_Sparse_Scalar},
    {"Test_GPU_4TreeXGB_Sparse_Scalar", Test_GPU_4TreeXGB_Sparse_Scalar},
    {"Test_GPU_1TreeXGB_Sparse_Scalar_f32i16", Test_GPU_1TreeXGB_Sparse_Scalar_f32i16},
    {"Test_GPU_2TreeXGB_Sparse_Scalar_f32i16", Test_GPU_2TreeXGB_Sparse_Scalar_f32i16},
    {"Test_GPU_4TreeXGB_Sparse_Scalar_f32i16", Test_GPU_4TreeXGB_Sparse_Scalar_f32i16},
    {"Test_GPU_1TreeXGB_Reorg_Scalar", Test_GPU_1TreeXGB_Reorg_Scalar},
    {"Test_GPU_2TreeXGB_Reorg_Scalar", Test_GPU_2TreeXGB_Reorg_Scalar},
    {"Test_GPU_4TreeXGB_Reorg_Scalar", Test_GPU_4TreeXGB_Reorg_Scalar},
    {"Test_GPU_1TreeXGB_Reorg_Scalar_f32i16", Test_GPU_1TreeXGB_Reorg_Scalar_f32i16},
    {"Test_GPU_2TreeXGB_Reorg_Scalar_f32i16", Test_GPU_2TreeXGB_Reorg_Scalar_f32i16},
    {"Test_GPU_4TreeXGB_Reorg_Scalar_f32i16", Test_GPU_4TreeXGB_Reorg_Scalar_f32i16},

    // GPU Synthetic XGB tests -- tile4
    {"Test_GPU_1TreeXGB_Sparse_Tile4", Test_GPU_1TreeXGB_Sparse_Tile4},
    {"Test_GPU_2TreeXGB_Sparse_Tile4", Test_GPU_2TreeXGB_Sparse_Tile4},
    {"Test_GPU_4TreeXGB_Sparse_Tile4", Test_GPU_4TreeXGB_Sparse_Tile4},
    {"Test_GPU_1TreeXGB_Sparse_Tile4_f32i16", Test_GPU_1TreeXGB_Sparse_Tile4_f32i16},
    {"Test_GPU_2TreeXGB_Sparse_Tile4_f32i16", Test_GPU_2TreeXGB_Sparse_Tile4_f32i16},
    {"Test_GPU_4TreeXGB_Sparse_Tile4_f32i16", Test_GPU_4TreeXGB_Sparse_Tile4_f32i16},
    {"Test_GPU_1TreeXGB_Array_Tile4", Test_GPU_1TreeXGB_Array_Tile4},
    {"Test_GPU_2TreeXGB_Array_Tile4", Test_GPU_2TreeXGB_Array_Tile4},
    {"Test_GPU_4TreeXGB_Array_Tile4", Test_GPU_4TreeXGB_Array_Tile4},
    {"Test_GPU_1TreeXGB_Array_Tile4_f32i16", Test_GPU_1TreeXGB_Array_Tile4_f32i16},
    {"Test_GPU_2TreeXGB_Array_Tile4_f32i16", Test_GPU_2TreeXGB_Array_Tile4_f32i16},
    {"Test_GPU_4TreeXGB_Array_Tile4_f32i16", Test_GPU_4TreeXGB_Array_Tile4_f32i16},

    // GPU Synthetic XGB tests -- Shared Forest -- Scalar
    {"Test_GPU_SharedForest_1TreeXGB_Reorg_Scalar", Test_GPU_SharedForest_1TreeXGB_Reorg_Scalar},
    {"Test_GPU_SharedForest_2TreeXGB_Reorg_Scalar", Test_GPU_SharedForest_2TreeXGB_Reorg_Scalar},
    // {"Test_GPU_SharedForest_4TreeXGB_Reorg_Scalar", Test_GPU_SharedForest_4TreeXGB_Reorg_Scalar}, // Commented out
    {"Test_GPU_SharedForest_1TreeXGB_Reorg_Scalar_f32i16", Test_GPU_SharedForest_1TreeXGB_Reorg_Scalar_f32i16},
    {"Test_GPU_SharedForest_2TreeXGB_Reorg_Scalar_f32i16", Test_GPU_SharedForest_2TreeXGB_Reorg_Scalar_f32i16},
    {"Test_GPU_SharedForest_4TreeXGB_Reorg_Scalar_f32i16", Test_GPU_SharedForest_4TreeXGB_Reorg_Scalar_f32i16},

    {"Test_GPU_SharedForest_1TreeXGB_Array_Scalar", Test_GPU_SharedForest_1TreeXGB_Array_Scalar},
    // {"Test_GPU_SharedForest_2TreeXGB_Array_Scalar", Test_GPU_SharedForest_2TreeXGB_Array_Scalar}, // Commented out
    // {"Test_GPU_SharedForest_4TreeXGB_Array_Scalar", Test_GPU_SharedForest_4TreeXGB_Array_Scalar}, // Commented out
    {"Test_GPU_SharedForest_1TreeXGB_Array_Scalar_f32i16", Test_GPU_SharedForest_1TreeXGB_Array_Scalar_f32i16},
    {"Test_GPU_SharedForest_2TreeXGB_Array_Scalar_f32i16", Test_GPU_SharedForest_2TreeXGB_Array_Scalar_f32i16},
    // {"Test_GPU_SharedForest_4TreeXGB_Array_Scalar_f32i16", Test_GPU_SharedForest_4TreeXGB_Array_Scalar_f32i16}, // Commented out

    {"Test_GPU_SharedForest_1TreeXGB_Sparse_Scalar", Test_GPU_SharedForest_1TreeXGB_Sparse_Scalar},
    {"Test_GPU_SharedForest_2TreeXGB_Sparse_Scalar", Test_GPU_SharedForest_2TreeXGB_Sparse_Scalar},
    {"Test_GPU_SharedForest_4TreeXGB_Sparse_Scalar", Test_GPU_SharedForest_4TreeXGB_Sparse_Scalar},
    {"Test_GPU_SharedForest_1TreeXGB_Sparse_Scalar_f32i16", Test_GPU_SharedForest_1TreeXGB_Sparse_Scalar_f32i16},
    {"Test_GPU_SharedForest_2TreeXGB_Sparse_Scalar_f32i16", Test_GPU_SharedForest_2TreeXGB_Sparse_Scalar_f32i16},
    {"Test_GPU_SharedForest_4TreeXGB_Sparse_Scalar_f32i16", Test_GPU_SharedForest_4TreeXGB_Sparse_Scalar_f32i16},

    // GPU Synthetic XGB tests -- Cache Partial Forest -- Scalar
    // NOTE: These schedules are different than Tahoe's partial shared forest schedule
    {"Test_GPU_CachePartialForest1Tree_2TreeXGB_Sparse_Scalar", Test_GPU_CachePartialForest1Tree_2TreeXGB_Sparse_Scalar},
    {"Test_GPU_CachePartialForest2Trees_4TreeXGB_Sparse_Scalar", Test_GPU_CachePartialForest2Trees_4TreeXGB_Sparse_Scalar},
    {"Test_GPU_CachePartialForest2Trees_2TreeXGB_Sparse_Scalar_f32i16", Test_GPU_CachePartialForest2Trees_2TreeXGB_Sparse_Scalar_f32i16},
    {"Test_GPU_CachePartialForest2Trees_4TreeXGB_Sparse_Scalar_f32i16", Test_GPU_CachePartialForest2Trees_4TreeXGB_Sparse_Scalar_f32i16},

    {"Test_GPU_CachePartialForest1Tree_2TreeXGB_Reorg_Scalar", Test_GPU_CachePartialForest1Tree_2TreeXGB_Reorg_Scalar},
    {"Test_GPU_CachePartialForest2Trees_4TreeXGB_Reorg_Scalar", Test_GPU_CachePartialForest2Trees_4TreeXGB_Reorg_Scalar},
    {"Test_GPU_CachePartialForest2Trees_2TreeXGB_Reorg_Scalar_f32i16", Test_GPU_CachePartialForest2Trees_2TreeXGB_Reorg_Scalar_f32i16},
    {"Test_GPU_CachePartialForest2Trees_4TreeXGB_Reorg_Scalar_f32i16", Test_GPU_CachePartialForest2Trees_4TreeXGB_Reorg_Scalar_f32i16},

    // Adding the new test entries
    {"Test_GPU_CachePartialForest1Tree_2TreeXGB_Array_Scalar", Test_GPU_CachePartialForest1Tree_2TreeXGB_Array_Scalar},
    {"Test_GPU_CachePartialForest1Tree_4TreeXGB_Array_Scalar", Test_GPU_CachePartialForest1Tree_4TreeXGB_Array_Scalar},
    {"Test_GPU_CachePartialForest2Trees_2TreeXGB_Array_Scalar_f32i16", Test_GPU_CachePartialForest2Trees_2TreeXGB_Array_Scalar_f32i16},
    {"Test_GPU_CachePartialForest2Trees_4TreeXGB_Array_Scalar_f32i16", Test_GPU_CachePartialForest2Trees_4TreeXGB_Array_Scalar_f32i16},
    {"Test_ScalarSparseGPU_TwiceLeftRightBalanced_TahoeShdInpMultiRow_FltI16_B32", Test_ScalarSparseGPU_TwiceLeftRightBalanced_TahoeShdInpMultiRow_FltI16_B32},
    {"Test_ScalarSparseGPU_LeftRightAndBalanced_TahoeShdInpMultiRow_FltI16_B32", Test_ScalarSparseGPU_LeftRightAndBalanced_TahoeShdInpMultiRow_FltI16_B32},
    {"Test_ScalarSparseGPU_LeftRightAndBalanced_TahoeShdInp_FltI16_B32", Test_ScalarSparseGPU_LeftRightAndBalanced_TahoeShdInp_FltI16_B32},
    {"Test_ScalarSparseGPU_TwiceLeftRightBalanced_IterShdPartialForest_FltI16_B32", Test_ScalarSparseGPU_TwiceLeftRightBalanced_IterShdPartialForest_FltI16_B32},
    {"Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Sparse_Scalar", Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Sparse_Scalar},
    {"Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Sparse_Scalar", Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Sparse_Scalar},
    {"Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Sparse_Scalar_f32i16", Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Sparse_Scalar_f32i16},
    {"Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Sparse_Scalar_f32i16", Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Sparse_Scalar_f32i16},
    {"Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Array_Scalar", Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Array_Scalar},
    {"Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Array_Scalar", Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Array_Scalar},
    {"Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Array_Scalar_f32i16", Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Array_Scalar_f32i16},
    {"Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Array_Scalar_f32i16", Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Array_Scalar_f32i16},
    {"Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Reorg_Scalar", Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Reorg_Scalar},
    {"Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Reorg_Scalar", Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Reorg_Scalar},
    {"Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Reorg_Scalar_f32i16", Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Reorg_Scalar_f32i16},
    {"Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Reorg_Scalar_f32i16", Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Reorg_Scalar_f32i16},
    {"Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Array_Scalar_f32i16", Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Array_Scalar_f32i16},
    {"Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Scalar", Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Scalar},
    {"Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Scalar_f32i16", Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Scalar_f32i16},
    {"Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Reorg_Scalar", Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Reorg_Scalar},
    {"Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Reorg_Scalar_f32i16", Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Reorg_Scalar_f32i16},
    // GPU Tree parallelization tests - Tile size 4
    {"Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Tile4", Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Tile4},
    {"Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Tile4_f32i16", Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Tile4_f32i16},
    {"Test_ScalarSparseGPU_TwiceLeftRightBalanced_iterCachedPartialForest_NoCache_SharedReduce_FltI16_B64", Test_ScalarSparseGPU_TwiceLeftRightBalanced_iterCachedPartialForest_NoCache_SharedReduce_FltI16_B64},
    {"Test_ScalarSparseGPU_TwiceLeftRightBalanced_iterCachedPartialForest_NoCache_SpecializedTreeLoop_FltI16_B64", Test_ScalarSparseGPU_TwiceLeftRightBalanced_iterCachedPartialForest_NoCache_SpecializedTreeLoop_FltI16_B64},
    // Tree Parallelization Multi-class tests
    {"Test_GPUCodeGeneration_Covtype_SparseRep_f32i16_B32_iterativeCachedPartialForestStrategy_NoCache", Test_GPUCodeGeneration_Covtype_SparseRep_f32i16_B32_iterativeCachedPartialForestStrategy_NoCache},
    {"Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleCachedTrees", Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleCachedTrees},
    {"Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleCachedRows", Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleCachedRows},
    {"Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleBasic", Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleBasic},
    {"Test_GPUCodeGeneration_Covtype_SparseRep_f32i16_B32_iterativeCachedPartialForestStrategy_NoCache_SharedReduce", Test_GPUCodeGeneration_Covtype_SparseRep_f32i16_B32_iterativeCachedPartialForestStrategy_NoCache_SharedReduce}
#endif // TREEBEARD_GPU_SUPPORT  
    
    //————————————— Add more missing tests from the TEST_LIST_ENTRY here ——————————————————————————//
};


#define RUN_ALL_TESTS

#ifdef RUN_ALL_TESTS
TestDescriptor testList[] = {
    TEST_LIST_ENTRY(Test_ONNX_TileSize8_Abalone),

    // [Ashwin] These tests are exercising a part of the code that
    // we intend to remove. Commenting them out to allow assertions
    // that Serializer::Persist is never called.
    // TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_LeftHeavy),
    // TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_Int16),
    // TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_Int8),
    // TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_Float),
    // TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_FloatInt16),
    // TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_FloatInt8),
    // TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy),
    // TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees),
    // TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees_Int16),
    // TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees_Int8),
    // TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees_Float),
    // TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees_FloatInt16),
    // TEST_LIST_ENTRY(Test_BufferInitializationWithTwoTrees_FloatInt8),

    TEST_LIST_ENTRY(Test_CodeGeneration_LeftHeavy_BatchSize1),
    TEST_LIST_ENTRY(Test_CodeGeneration_RightHeavy_BatchSize1),
    TEST_LIST_ENTRY(Test_CodeGeneration_RightAndLeftHeavy_BatchSize1),
    TEST_LIST_ENTRY(Test_CodeGeneration_LeftHeavy_BatchSize2),
    TEST_LIST_ENTRY(Test_CodeGeneration_RightHeavy_BatchSize2),
    TEST_LIST_ENTRY(Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2),
    TEST_LIST_ENTRY(Test_LoadTileFeatureIndicesOp_DoubleInt32_TileSize1),
    TEST_LIST_ENTRY(Test_LoadTileThresholdOp_DoubleInt32_TileSize1),
    TEST_LIST_ENTRY(Test_LoadTileThresholdOp_Subview_DoubleInt32_TileSize1),
    TEST_LIST_ENTRY(
        Test_LoadTileFeatureIndicesOp_Subview_DoubleInt32_TileSize1),
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

    // #TODO - Re-enable these tests
    // TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_RightHeavy_Tiled),
    // TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_LeftHeavy_Tiled),
    // TEST_LIST_ENTRY(Test_BufferInitializationWithOneTree_Balanced_Tiled),

    TEST_LIST_ENTRY(Test_TiledCodeGeneration_LeftHeavy_BatchSize1),
    TEST_LIST_ENTRY(Test_TiledCodeGeneration_RightHeavy_BatchSize1),
    TEST_LIST_ENTRY(Test_TiledCodeGeneration_BalancedTree_BatchSize1),
    TEST_LIST_ENTRY(Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1),
    TEST_LIST_ENTRY(
        Test_TiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape),
    TEST_LIST_ENTRY(
        Test_TiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape),
    TEST_LIST_ENTRY(
        Test_TiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape),
    TEST_LIST_ENTRY(
        Test_TiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape),
    TEST_LIST_ENTRY(
        Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize),
    TEST_LIST_ENTRY(
        Test_TiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize),

    // #TODO - Re-enable these tests
    // TEST_LIST_ENTRY(Test_ModelInit_LeftHeavy),
    // TEST_LIST_ENTRY(Test_ModelInit_RightHeavy),
    // TEST_LIST_ENTRY(Test_ModelInit_RightAndLeftHeavy),
    // TEST_LIST_ENTRY(Test_ModelInit_Balanced),
    // TEST_LIST_ENTRY(Test_ModelInit_LeftHeavy_Int8TileShape),
    // TEST_LIST_ENTRY(Test_ModelInit_LeftHeavy_Int16TileShape),
    // TEST_LIST_ENTRY(Test_ModelInit_RightHeavy_Int8TileShape),
    // TEST_LIST_ENTRY(Test_ModelInit_RightHeavy_Int16TileShape),
    // TEST_LIST_ENTRY(Test_ModelInit_Balanced_Int8TileShape),
    // TEST_LIST_ENTRY(Test_ModelInit_Balanced_Int16TileShape),
    // TEST_LIST_ENTRY(Test_ModelInit_RightAndLeftHeavy_Int8TileShape),
    // TEST_LIST_ENTRY(Test_ModelInit_RightAndLeftHeavy_Int16TileShape),

    TEST_LIST_ENTRY(Test_UniformTiling_LeftHeavy_BatchSize1),
    TEST_LIST_ENTRY(Test_UniformTiling_LeftHeavy_BatchSize1),
    TEST_LIST_ENTRY(Test_UniformTiling_RightHeavy_BatchSize1),
    TEST_LIST_ENTRY(Test_UniformTiling_Balanced_BatchSize1),
    TEST_LIST_ENTRY(Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1),
    TEST_LIST_ENTRY(Test_UniformTiling_LeftHeavy_BatchSize1_Int8TileShape),
    TEST_LIST_ENTRY(Test_UniformTiling_RightHeavy_BatchSize1_Int8TileShape),
    TEST_LIST_ENTRY(Test_UniformTiling_Balanced_BatchSize1_Int8TileShape),
    TEST_LIST_ENTRY(
        Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1_Int8TileShape),
    TEST_LIST_ENTRY(Test_UniformTiling_LeftHeavy_BatchSize1_Int16TileShape),
    TEST_LIST_ENTRY(Test_UniformTiling_RightHeavy_BatchSize1_Int16TileShape),
    TEST_LIST_ENTRY(Test_UniformTiling_Balanced_BatchSize1_Int16TileShape),
    TEST_LIST_ENTRY(
        Test_UniformTiling_LeftfAndRighttHeavy_BatchSize1_Int16TileShape),
    TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize1),
    TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize1),
    TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize1),
    TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize2),
    TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize2),
    TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize2),
    TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4),
    TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4),
    TEST_LIST_ENTRY(Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4),
    TEST_LIST_ENTRY(
        Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4_Int8TileShape),
    TEST_LIST_ENTRY(
        Test_UniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4_Int16TileShape),
    TEST_LIST_ENTRY(
        Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4_Int8TileShape),
    TEST_LIST_ENTRY(
        Test_UniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4_Int16TileShape),
    TEST_LIST_ENTRY(
        Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4_Int8TileShape),
    TEST_LIST_ENTRY(
        Test_UniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4_Int16TileShape),
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
    TEST_LIST_ENTRY(Test_TileSize1_Letters_Int8Type),
    TEST_LIST_ENTRY(Test_TileSize2_Letters_Int8Type),
    TEST_LIST_ENTRY(Test_TileSize3_Letters_Int8Type),
    TEST_LIST_ENTRY(Test_TileSize4_Letters_Int8Type),
    TEST_LIST_ENTRY(Test_TileSize8_Letters_Int8Type),
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

    // // Sparse tests
    TEST_LIST_ENTRY(Test_SparseCodeGeneration_LeftHeavy_BatchSize1_I32ChildIdx),
    TEST_LIST_ENTRY(
        Test_SparseCodeGeneration_RightHeavy_BatchSize1_I32ChildIdx),
    TEST_LIST_ENTRY(
        Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize1_I32ChildIdx),
    TEST_LIST_ENTRY(Test_SparseCodeGeneration_LeftHeavy_BatchSize2_I32ChildIdx),
    TEST_LIST_ENTRY(
        Test_SparseCodeGeneration_RightHeavy_BatchSize2_I32ChildIdx),
    TEST_LIST_ENTRY(
        Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize2_I32ChildIdx),
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
    TEST_LIST_ENTRY(
        Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape),
    TEST_LIST_ENTRY(
        Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape),
    TEST_LIST_ENTRY(
        Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize),
    TEST_LIST_ENTRY(
        Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1),
    TEST_LIST_ENTRY(
        Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize),
    TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1),
    TEST_LIST_ENTRY(
        Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape),
    TEST_LIST_ENTRY(
        Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape),
    TEST_LIST_ENTRY(
        Test_SparseUniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4),
    TEST_LIST_ENTRY(
        Test_SparseUniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4),
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
    TEST_LIST_ENTRY(Test_SparseScalar_Letters_Int8Type),
    TEST_LIST_ENTRY(Test_SparseTileSize8_Letters_Int8Type),
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
    TEST_LIST_ENTRY(
        Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2_XGBoostSchedule),
    TEST_LIST_ENTRY(
        Test_CodeGeneration_LeftHeavy_BatchSize8_CacheInputSchedule),
    TEST_LIST_ENTRY(
        Test_CodeGeneration_RightHeavy_BatchSize2_CacheInputSchedule),
    TEST_LIST_ENTRY(
        Test_CodeGeneration_AddRightAndLeftHeavyTrees_BatchSize2_CacheInputSchedule),
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
    TEST_LIST_ENTRY(Test_SparseTileSize8_Letters_TiledSchedule),

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
    TEST_LIST_ENTRY(Test_SparseScalar_Letters_OneTreeAtATimeSchedule),
    TEST_LIST_ENTRY(Test_SparseTileSize8_Letters_OneTreeAtATimeSchedule),
    TEST_LIST_ENTRY(Test_SparseTileSize8_Letters_TiledSchedule),
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
    TEST_LIST_ENTRY(
        Test_RandomXGBoostJSONs_1Tree_BatchSize4_EqualDepth_TileSize8),
    TEST_LIST_ENTRY(
        Test_RandomXGBoostJSONs_2Trees_BatchSize4_EqualDepth_TileSize8),
    TEST_LIST_ENTRY(
        Test_RandomXGBoostJSONs_4Trees_BatchSize4_EqualDepth_TileSize8),
    TEST_LIST_ENTRY(Test_TileSize8_Abalone_TestInputs_MakeLeavesSameDepth),
    TEST_LIST_ENTRY(Test_TileSize8_AirlineOHE_TestInputs_MakeLeavesSameDepth),
    TEST_LIST_ENTRY(Test_TileSize8_Airline_TestInputs_MakeLeavesSameDepth),
    TEST_LIST_ENTRY(Test_TileSize8_Epsilon_TestInputs_MakeLeavesSameDepth),
    TEST_LIST_ENTRY(Test_TileSize8_Higgs_TestInputs_MakeLeavesSameDepth),
    TEST_LIST_ENTRY(Test_TileSize8_Year_TestInputs_MakeLeavesSameDepth),
    TEST_LIST_ENTRY(Test_TileSize8_CovType_TestInputs_MakeLeavesSameDepth),

    TEST_LIST_ENTRY(Test_TileSize8_Abalone_TestInputs_ReorderTrees),

    // Split Schedule
    TEST_LIST_ENTRY(Test_TileSize8_Abalone_TestInputs_SwapAndSplitTreeIndex),
    TEST_LIST_ENTRY(Test_TileSize8_AirlineOHE_TestInputs_SwapAndSplitTreeIndex),
    TEST_LIST_ENTRY(Test_TileSize8_Airline_TestInputs_SwapAndSplitTreeIndex),
    TEST_LIST_ENTRY(Test_TileSize8_Epsilon_TestInputs_SwapAndSplitTreeIndex),
    TEST_LIST_ENTRY(Test_TileSize8_Higgs_TestInputs_SwapAndSplitTreeIndex),
    TEST_LIST_ENTRY(Test_TileSize8_Year_TestInputs_SwapAndSplitTreeIndex),
    TEST_LIST_ENTRY(Test_TileSize8_Abalone_TestInputs_SplitTreeLoopSchedule),
    TEST_LIST_ENTRY(Test_TileSize8_AirlineOHE_TestInputs_SplitTreeLoopSchedule),
    TEST_LIST_ENTRY(Test_TileSize8_Airline_TestInputs_SplitTreeLoopSchedule),
    TEST_LIST_ENTRY(Test_TileSize8_Epsilon_TestInputs_SplitTreeLoopSchedule),
    TEST_LIST_ENTRY(Test_TileSize8_Higgs_TestInputs_SplitTreeLoopSchedule),
    TEST_LIST_ENTRY(Test_TileSize8_Year_TestInputs_SplitTreeLoopSchedule),

#ifdef OMP_SUPPORT
    TEST_LIST_ENTRY(Test_TileSize8_Abalone_TestInputs_ParallelBatch),
    TEST_LIST_ENTRY(Test_TileSize8_Airline_TestInputs_ParallelBatch),
    TEST_LIST_ENTRY(Test_TileSize8_AirlineOHE_TestInputs_ParallelBatch),
    TEST_LIST_ENTRY(Test_TileSize8_Covtype_TestInputs_ParallelBatch),
    TEST_LIST_ENTRY(Test_TileSize8_Letters_TestInputs_ParallelBatch),
    TEST_LIST_ENTRY(Test_TileSize8_Epsilon_TestInputs_ParallelBatch),
    TEST_LIST_ENTRY(Test_TileSize8_Higgs_TestInputs_ParallelBatch),
    TEST_LIST_ENTRY(Test_TileSize8_Year_TestInputs_ParallelBatch),
#endif // OMP_SUPPORT

    // Pipelining + Unrolling tests
    TEST_LIST_ENTRY(
        Test_RandomXGBoostJSONs_1Tree_BatchSize8_TileSize2_4Pipelined),
    TEST_LIST_ENTRY(Test_RandomXGBoostJSONs_4Trees_BatchSize4_4Pipelined),
    TEST_LIST_ENTRY(Test_TileSize3_Letters_2Pipelined_Int8Type),
    TEST_LIST_ENTRY(Test_TileSize4_Letters_3Pipelined_Int8Type),
    TEST_LIST_ENTRY(Test_TileSize8_Letters_5Pipelined_Int8Type),
    TEST_LIST_ENTRY(Test_SparseTileSize8_4Pipelined_Bosch),
    TEST_LIST_ENTRY(Test_TileSize8_Abalone_4Pipelined_TestInputs),
    TEST_LIST_ENTRY(Test_TileSize8_CovType_4Pipelined_TestInputs),
    TEST_LIST_ENTRY(Test_SparseTileSize8_Pipeline4_Airline),
    TEST_LIST_ENTRY(Test_SparseTileSize8_Pipelined4_AirlineOHE),
    TEST_LIST_ENTRY(Test_SparseTileSize8_Pipelined_Year),
    TEST_LIST_ENTRY(Test_SparseTileSize8_Pipelined_Higgs),
    TEST_LIST_ENTRY(Test_SparseTileSize8_Pipelined_Epsilon),

    TEST_LIST_ENTRY(Test_TileSize8_Abalone_4PipelinedTrees_TestInputs),
    TEST_LIST_ENTRY(Test_TileSize8_Abalone_PipelinedTreesPeeling_TestInputs),

    // Hybrid Tiling
    TEST_LIST_ENTRY(Test_WalkPeeling_BalancedTree_TileSize2),
    TEST_LIST_ENTRY(
        Test_HybridTilingAndPeeling_RandomXGBoostJSONs_4Tree_FloatBatchSize4),
    TEST_LIST_ENTRY(
        Test_HybridTilingAndPeeling_RandomXGBoostJSONs_1Tree_FloatBatchSize4),
    TEST_LIST_ENTRY(
        Test_HybridTilingAndPeeling_RandomXGBoostJSONs_2Tree_FloatBatchSize4),
    TEST_LIST_ENTRY(
        Test_UniformAndHybridTilingAndPeeling_RandomXGBoostJSONs_2Tree_FloatBatchSize4),
    TEST_LIST_ENTRY(
        Test_UniformAndHybridTilingAndPeeling_RandomXGBoostJSONs_4Tree_FloatBatchSize4),
    TEST_LIST_ENTRY(Test_PeeledHybridProbabilisticTiling_TileSize8_Year),
    TEST_LIST_ENTRY(Test_PeeledHybridProbabilisticTiling_TileSize8_Letters),
    TEST_LIST_ENTRY(Test_PeeledHybridProbabilisticTiling_TileSize8_Epsilon),
    TEST_LIST_ENTRY(Test_PeeledHybridProbabilisticTiling_TileSize8_Higgs),
    TEST_LIST_ENTRY(Test_PeeledHybridProbabilisticTiling_TileSize8_AirlineOHE),
    TEST_LIST_ENTRY(Test_PeeledHybridProbabilisticTiling_TileSize8_Covtype),
    TEST_LIST_ENTRY(Test_PeeledHybridProbabilisticTiling_TileSize8_Airline),
    TEST_LIST_ENTRY(Test_PeeledHybridProbabilisticTiling_TileSize8_Abalone),

#ifdef TREEBEARD_GPU_SUPPORT
    // GPU model buffer initialization tests (scalar)
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftHeavy_Scalar_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_RightHeavy_Scalar_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_Balanced_Scalar_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftAndRightHeavy_Scalar_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftHeavy_Scalar_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_RightHeavy_Scalar_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_Balanced_Scalar_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftAndRightHeavy_Scalar_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftHeavy_Scalar_FloatInt16),
    TEST_LIST_ENTRY(Test_GPUModelInit_RightHeavy_Scalar_FloatInt16),
    TEST_LIST_ENTRY(Test_GPUModelInit_Balanced_Scalar_FloatInt16),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftAndRightHeavy_Scalar_FloatInt16),

    TEST_LIST_ENTRY(Test_GPUModelInit_LeftHeavy_Reorg_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_RightHeavy_Reorg_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_Balanced_Reorg_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftAndRightHeavy_Reorg_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftHeavy_Reorg_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_RightHeavy_Reorg_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_Balanced_Reorg_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftAndRightHeavy_Reorg_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftHeavy_Reorg_FloatInt16),
    TEST_LIST_ENTRY(Test_GPUModelInit_RightHeavy_Reorg_FloatInt16),
    TEST_LIST_ENTRY(Test_GPUModelInit_Balanced_Reorg_FloatInt16),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftAndRightHeavy_Reorg_FloatInt16),

    // Basic Array Scalar GPU Codegen Tests
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Balanced_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_GPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_LeftHeavy_FloatInt16_BatchSize32),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_RightHeavy_FloatInt16_BatchSize32),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Balanced_FloatInt16_BatchSize32),
    TEST_LIST_ENTRY(
        Test_GPUCodeGeneration_LeftAndRightHeavy_FloatInt16_BatchSize32),

    // Basic scalar sparse GPU codegen tests
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_Balanced_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_LeftHeavy_FloatInt16_ChI16_BatchSize32),
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_RightHeavy_FloatInt16_ChI16_BatchSize32),
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_Balanced_FloatInt16_ChI16_BatchSize32),
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_LeftAndRightHeavy_FloatInt16_ChI16_BatchSize32),

    // Basic reorg forest tests
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_Balanced_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_LeftHeavy_FloatInt16_BatchSize32),
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_RightHeavy_FloatInt16_BatchSize32),
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_LeftAndRightHeavy_FloatInt16_BatchSize32),
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_LeftRightAndBalanced_FloatInt16_BatchSize32),

    // Basic GPU caching tests
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftRightAndBalanced),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftHeavy),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftHeavy_F32I16),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftRightAndBalanced_F32I16),

    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftHeavy_ReorgRep),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftRightAndBalanced_Reorg),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftHeavy_ReorgRep_F32I16),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftRightAndBalanced_Reorg_F32I16),

    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftHeavy_SparseRep),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftRightAndBalanced_SparseRep),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftHeavy_SparseRep_F32I16),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftRightAndBalanced_SparseRep_F32I16),

    TEST_LIST_ENTRY(Test_InputSharedMem_LeftHeavy),
    TEST_LIST_ENTRY(Test_InputSharedMem_RightHeavy),
    TEST_LIST_ENTRY(Test_InputSharedMem_LeftRightAndBalanced),

    // Simple GPU Tiling tests
    TEST_LIST_ENTRY(Test_TiledSparseGPU_LeftHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledSparseGPU_RightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledSparseGPU_Balanced_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledSparseGPU_LeftAndRightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledSparseGPU_LeftHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledSparseGPU_RightHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledSparseGPU_Balanced_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledSparseGPU_LeftAndRightHeavy_FltI16_B32_TSz2),

    TEST_LIST_ENTRY(Test_TiledArrayGPU_LeftHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledArrayGPU_RightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledArrayGPU_Balanced_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledArrayGPU_LeftAndRightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledArrayGPU_LeftHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledArrayGPU_RightHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledArrayGPU_Balanced_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledArrayGPU_LeftAndRightHeavy_FltI16_B32_TSz2),

    // Tiling + Caching
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_LeftHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_RightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_Balanced_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_LeftAndRightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(
        Test_TiledCachedArrayGPU_LeftRightAndBalanced_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_LeftHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_RightHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_Balanced_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_LeftAndRightHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(
        Test_TiledCachedArrayGPU_LeftRightAndBalanced_FltI16_B32_TSz2),

    TEST_LIST_ENTRY(Test_TiledCachedSparseGPU_LeftHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedSparseGPU_RightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedSparseGPU_Balanced_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(
        Test_TiledCachedSparseGPU_LeftAndRightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedSparseGPU_LeftHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedSparseGPU_RightHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedSparseGPU_Balanced_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(
        Test_TiledCachedSparseGPU_LeftAndRightHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(
        Test_TiledCachedSparseGPU_LeftRightAndBalanced_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(
        Test_TiledCachedSparseGPU_LeftRightAndBalanced_FltI16_B32_TSz2),

    // GPU Synthetic XGB tests -- scalar
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Reorg_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Reorg_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Reorg_Scalar_f32i16),

    // GPU Synthetic XGB tests -- tile4
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Sparse_Tile4),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Sparse_Tile4),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Sparse_Tile4),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Sparse_Tile4_f32i16),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Sparse_Tile4_f32i16),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Sparse_Tile4_f32i16),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Array_Tile4),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Array_Tile4),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Array_Tile4),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Array_Tile4_f32i16),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Array_Tile4_f32i16),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Array_Tile4_f32i16),

    // GPU Synthetic XGB tests -- Shared Forest -- Scalar
    TEST_LIST_ENTRY(Test_GPU_SharedForest_1TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_2TreeXGB_Reorg_Scalar),
    // TEST_LIST_ENTRY(Test_GPU_SharedForest_4TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_1TreeXGB_Reorg_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_2TreeXGB_Reorg_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_4TreeXGB_Reorg_Scalar_f32i16),

    TEST_LIST_ENTRY(Test_GPU_SharedForest_1TreeXGB_Array_Scalar),
    // TEST_LIST_ENTRY(Test_GPU_SharedForest_2TreeXGB_Array_Scalar),
    // TEST_LIST_ENTRY(Test_GPU_SharedForest_4TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_1TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_2TreeXGB_Array_Scalar_f32i16),
    // TEST_LIST_ENTRY(Test_GPU_SharedForest_4TreeXGB_Array_Scalar_f32i16),

    TEST_LIST_ENTRY(Test_GPU_SharedForest_1TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_2TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_4TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_1TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_2TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_4TreeXGB_Sparse_Scalar_f32i16),

    // GPU Synthetic XGB tests -- Cache Partial Forest -- Scalar
    // NOTE: These schedules are different than Tahoe's partial shared
    // forest schedule
    TEST_LIST_ENTRY(Test_GPU_CachePartialForest1Tree_2TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_CachePartialForest2Trees_4TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_CachePartialForest2Trees_2TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_CachePartialForest2Trees_4TreeXGB_Sparse_Scalar_f32i16),

    TEST_LIST_ENTRY(Test_GPU_CachePartialForest1Tree_2TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(Test_GPU_CachePartialForest2Trees_4TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_CachePartialForest2Trees_2TreeXGB_Reorg_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_CachePartialForest2Trees_4TreeXGB_Reorg_Scalar_f32i16),

    TEST_LIST_ENTRY(Test_GPU_CachePartialForest1Tree_2TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(Test_GPU_CachePartialForest1Tree_4TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_CachePartialForest2Trees_2TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_CachePartialForest2Trees_4TreeXGB_Array_Scalar_f32i16),

    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Abalone_TileSize1_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Abalone_TileSize2_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Abalone_TileSize4_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Abalone_TileSize8_BasicSchedule),

    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Airline_TileSize1_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Airline_TileSize2_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Airline_TileSize4_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Airline_TileSize8_BasicSchedule),

    TEST_LIST_ENTRY(Test_GPUCodeGeneration_AirlineOHE_TileSize1_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_AirlineOHE_TileSize2_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_AirlineOHE_TileSize4_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_AirlineOHE_TileSize8_BasicSchedule),

    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Bosch_TileSize1_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Bosch_TileSize2_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Bosch_TileSize4_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Bosch_TileSize8_BasicSchedule),

    TEST_LIST_ENTRY(Test_GPUCodeGeneration_CovType_TileSize1_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_CovType_TileSize2_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_CovType_TileSize4_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_CovType_TileSize8_BasicSchedule),

    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Epsilon_TileSize1_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Epsilon_TileSize2_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Epsilon_TileSize4_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Epsilon_TileSize8_BasicSchedule),

    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Higgs_TileSize1_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Higgs_TileSize2_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Higgs_TileSize4_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Higgs_TileSize8_BasicSchedule),

    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Letters_TileSize1_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Letters_TileSize2_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Letters_TileSize4_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Letters_TileSize8_BasicSchedule),

    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Year_TileSize1_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Year_TileSize2_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Year_TileSize4_BasicSchedule),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Year_TileSize8_BasicSchedule),
#endif // TREEBEARD_GPU_SUPPORT

    // Parallelize across trees
    TEST_LIST_ENTRY(Test_TreePar_LeftRightAndBalanced_DblI32),
    TEST_LIST_ENTRY(Test_NestedTreePar_LeftRightAndBalanced_DblI32),
    TEST_LIST_ENTRY(Test_AtomicReduction_TwiceLeftRightAndBalanced_DblI32),
    TEST_LIST_ENTRY(Test_VectorReduction_TwiceLeftRightAndBalanced_DblI32),
    TEST_LIST_ENTRY(Test_SparseTileSize8_Abalone_TestInputs_4ParallelTreeSets),
    TEST_LIST_ENTRY(Test_SparseTileSize8_Airline_TestInputs_4ParallelTreeSets),
    TEST_LIST_ENTRY(
        Test_SparseTileSize8_AirlineOHE_TestInputs_4ParallelTreeSets),
    TEST_LIST_ENTRY(Test_SparseTileSize8_Covtype_TestInputs_4ParallelTreeSets),
    TEST_LIST_ENTRY(Test_SparseTileSize8_Letters_TestInputs_4ParallelTreeSets),
    TEST_LIST_ENTRY(Test_SparseTileSize8_Epsilon_TestInputs_4ParallelTreeSets),
    TEST_LIST_ENTRY(Test_SparseTileSize8_Higgs_TestInputs_4ParallelTreeSets),
    TEST_LIST_ENTRY(Test_SparseTileSize8_Year_TestInputs_4ParallelTreeSets),

    TEST_LIST_ENTRY(
        Test_SparseTileSize8_Abalone_TestInputs_4ParallelTreeSets_VectorReduce),
    TEST_LIST_ENTRY(
        Test_SparseTileSize8_Airline_TestInputs_4ParallelTreeSets_VectorReduce),
    TEST_LIST_ENTRY(
        Test_SparseTileSize8_AirlineOHE_TestInputs_4ParallelTreeSets_VectorReduce),
    TEST_LIST_ENTRY(
        Test_SparseTileSize8_Covtype_TestInputs_4ParallelTreeSets_VectorReduce),
    // This test fails because vector reduction can only handle perfect
    // multiples of vector size
    //  TEST_LIST_ENTRY(
    //     Test_SparseTileSize8_Letters_TestInputs_4ParallelTreeSets_VectorReduce),
    TEST_LIST_ENTRY(
        Test_SparseTileSize8_Epsilon_TestInputs_4ParallelTreeSets_VectorReduce),
    TEST_LIST_ENTRY(
        Test_SparseTileSize8_Higgs_TestInputs_4ParallelTreeSets_VectorReduce),
    TEST_LIST_ENTRY(
        Test_SparseTileSize8_Year_TestInputs_4ParallelTreeSets_VectorReduce),

    TEST_LIST_ENTRY(
        Test_SparseTileSize8_Abalone_TestInputs_4ParallelTreeSets_AtomicReduce),
    TEST_LIST_ENTRY(
        Test_SparseTileSize8_Airline_TestInputs_4ParallelTreeSets_AtomicReduce),
    TEST_LIST_ENTRY(
        Test_SparseTileSize8_AirlineOHE_TestInputs_4ParallelTreeSets_AtomicReduce),
    TEST_LIST_ENTRY(
        Test_SparseTileSize8_Covtype_TestInputs_4ParallelTreeSets_AtomicReduce),
    TEST_LIST_ENTRY(
        Test_SparseTileSize8_Letters_TestInputs_4ParallelTreeSets_AtomicReduce),
    TEST_LIST_ENTRY(
        Test_SparseTileSize8_Epsilon_TestInputs_4ParallelTreeSets_AtomicReduce),
    TEST_LIST_ENTRY(
        Test_SparseTileSize8_Higgs_TestInputs_4ParallelTreeSets_AtomicReduce),
    TEST_LIST_ENTRY(
        Test_SparseTileSize8_Year_TestInputs_4ParallelTreeSets_AtomicReduce),

#ifdef TREEBEARD_GPU_SUPPORT
    // GPU Parallelize across trees
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_TwiceLeftRightBalanced_TahoeShdInpMultiRow_FltI16_B32),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_LeftRightAndBalanced_TahoeShdInpMultiRow_FltI16_B32),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_LeftRightAndBalanced_TahoeShdInp_FltI16_B32),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_TwiceLeftRightBalanced_IterShdPartialForest_FltI16_B32),
    TEST_LIST_ENTRY(Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Reorg_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Reorg_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Reorg_Scalar_f32i16),
    // GPU Tree parallelization tests - Tile size 4
    TEST_LIST_ENTRY(
        Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Tile4),
    TEST_LIST_ENTRY(
        Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Tile4_f32i16),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_TwiceLeftRightBalanced_iterCachedPartialForest_NoCache_SharedReduce_FltI16_B64),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_TwiceLeftRightBalanced_iterCachedPartialForest_NoCache_SpecializedTreeLoop_FltI16_B64),
    // Tree Parallelization Multi-class tests
    TEST_LIST_ENTRY(
        Test_GPUCodeGeneration_Covtype_SparseRep_f32i16_B32_iterativeCachedPartialForestStrategy_NoCache),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleCachedTrees),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleCachedRows),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleBasic),
    TEST_LIST_ENTRY(
        Test_GPUCodeGeneration_Covtype_SparseRep_f32i16_B32_iterativeCachedPartialForestStrategy_NoCache_SharedReduce),

    TEST_LIST_ENTRY(Test_ScalarGPU_Airline_AutoScheduleBasic),
    TEST_LIST_ENTRY(Test_ScalarGPU_Abalone_AutoScheduleBasic),
    TEST_LIST_ENTRY(
        Test_GPUCodeGeneration_Covtype_SparseRep_f32i16_B512_AutoSched_SharedReduce),
    TEST_LIST_ENTRY(
        Test_GPUCodeGeneration_Letters_SparseRep_f32i16_B512_AutoSched_SharedReduce),
    TEST_LIST_ENTRY(
        Test_GPUCodeGeneration_Abalone_SparseRep_f32i16_B32_iterativeCachedPartialForestStrategy_NoCache_SharedReduce),
#endif // TREEBEARD_GPU_SUPPORT
    TEST_LIST_ENTRY(
        Test_TileSize8_Abalone_TestInputs_CPUAutoSchedule_TreeParallel_f32i16),
    TEST_LIST_ENTRY(
        Test_TileSize8_Airline_TestInputs_CPUAutoSchedule_TreeParallel_f32i16),
    TEST_LIST_ENTRY(
        Test_TileSize8_AirlineOHE_TestInputs_CPUAutoSchedule_TreeParallel_f32i16),
    TEST_LIST_ENTRY(
        Test_TileSize8_Covtype_TestInputs_CPUAutoSchedule_TreeParallel_f32i16),
    TEST_LIST_ENTRY(
        Test_TileSize8_Epsilon_TestInputs_CPUAutoSchedule_TreeParallel_f32i16),
    TEST_LIST_ENTRY(
        Test_TileSize8_Higgs_TestInputs_CPUAutoSchedule_TreeParallel_f32i16),
    TEST_LIST_ENTRY(
        Test_TileSize8_Letters_TestInputs_CPUAutoSchedule_TreeParallel_f32i16),
    TEST_LIST_ENTRY(
        Test_TileSize8_Year_TestInputs_CPUAutoSchedule_TreeParallel_f32i16),
};

#else  // RUN_ALL_TESTS

TestDescriptor testList[] = {
    TEST_LIST_ENTRY(
        Test_GPUCodeGeneration_Letters_SparseRep_f32i16_B512_AutoSched_SharedReduce),
};
#endif // RUN_ALL_TESTS

#ifdef TREEBEARD_GPU_SUPPORT
TestDescriptor gpuTestList[] = {
    // GPU model buffer initialization tests (scalar)
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftHeavy_Scalar_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_RightHeavy_Scalar_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_Balanced_Scalar_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftAndRightHeavy_Scalar_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftHeavy_Scalar_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_RightHeavy_Scalar_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_Balanced_Scalar_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftAndRightHeavy_Scalar_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftHeavy_Scalar_FloatInt16),
    TEST_LIST_ENTRY(Test_GPUModelInit_RightHeavy_Scalar_FloatInt16),
    TEST_LIST_ENTRY(Test_GPUModelInit_Balanced_Scalar_FloatInt16),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftAndRightHeavy_Scalar_FloatInt16),

    TEST_LIST_ENTRY(Test_GPUModelInit_LeftHeavy_Reorg_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_RightHeavy_Reorg_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_Balanced_Reorg_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftAndRightHeavy_Reorg_DoubleInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftHeavy_Reorg_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_RightHeavy_Reorg_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_Balanced_Reorg_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftAndRightHeavy_Reorg_FloatInt),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftHeavy_Reorg_FloatInt16),
    TEST_LIST_ENTRY(Test_GPUModelInit_RightHeavy_Reorg_FloatInt16),
    TEST_LIST_ENTRY(Test_GPUModelInit_Balanced_Reorg_FloatInt16),
    TEST_LIST_ENTRY(Test_GPUModelInit_LeftAndRightHeavy_Reorg_FloatInt16),

    // Basic Array Scalar GPU Codegen Tests
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Balanced_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_GPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_LeftHeavy_FloatInt16_BatchSize32),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_RightHeavy_FloatInt16_BatchSize32),
    TEST_LIST_ENTRY(Test_GPUCodeGeneration_Balanced_FloatInt16_BatchSize32),
    TEST_LIST_ENTRY(
        Test_GPUCodeGeneration_LeftAndRightHeavy_FloatInt16_BatchSize32),

    // Basic scalar sparse GPU codegen tests
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_Balanced_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_LeftHeavy_FloatInt16_ChI16_BatchSize32),
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_RightHeavy_FloatInt16_ChI16_BatchSize32),
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_Balanced_FloatInt16_ChI16_BatchSize32),
    TEST_LIST_ENTRY(
        Test_SparseGPUCodeGeneration_LeftAndRightHeavy_FloatInt16_ChI16_BatchSize32),

    // Basic reorg forest tests
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_LeftHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_RightHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_Balanced_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_LeftAndRightHeavy_DoubleInt32_BatchSize32),
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_LeftHeavy_FloatInt16_BatchSize32),
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_RightHeavy_FloatInt16_BatchSize32),
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_LeftAndRightHeavy_FloatInt16_BatchSize32),
    TEST_LIST_ENTRY(
        Test_ReorgGPUCodeGeneration_LeftRightAndBalanced_FloatInt16_BatchSize32),

    // Basic GPU caching tests
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftRightAndBalanced),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftHeavy),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftHeavy_F32I16),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftRightAndBalanced_F32I16),

    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftHeavy_ReorgRep),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftRightAndBalanced_Reorg),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftHeavy_ReorgRep_F32I16),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftRightAndBalanced_Reorg_F32I16),

    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftHeavy_SparseRep),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftRightAndBalanced_SparseRep),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftHeavy_SparseRep_F32I16),
    TEST_LIST_ENTRY(Test_SimpleSharedMem_LeftRightAndBalanced_SparseRep_F32I16),

    TEST_LIST_ENTRY(Test_InputSharedMem_LeftHeavy),
    TEST_LIST_ENTRY(Test_InputSharedMem_RightHeavy),
    TEST_LIST_ENTRY(Test_InputSharedMem_LeftRightAndBalanced),

    // Simple GPU Tiling tests
    TEST_LIST_ENTRY(Test_TiledSparseGPU_LeftHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledSparseGPU_RightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledSparseGPU_Balanced_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledSparseGPU_LeftAndRightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledSparseGPU_LeftHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledSparseGPU_RightHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledSparseGPU_Balanced_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledSparseGPU_LeftAndRightHeavy_FltI16_B32_TSz2),

    TEST_LIST_ENTRY(Test_TiledArrayGPU_LeftHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledArrayGPU_RightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledArrayGPU_Balanced_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledArrayGPU_LeftAndRightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledArrayGPU_LeftHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledArrayGPU_RightHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledArrayGPU_Balanced_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledArrayGPU_LeftAndRightHeavy_FltI16_B32_TSz2),

    // Tiling + Caching
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_LeftHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_RightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_Balanced_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_LeftAndRightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(
        Test_TiledCachedArrayGPU_LeftRightAndBalanced_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_LeftHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_RightHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_Balanced_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedArrayGPU_LeftAndRightHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(
        Test_TiledCachedArrayGPU_LeftRightAndBalanced_FltI16_B32_TSz2),

    TEST_LIST_ENTRY(Test_TiledCachedSparseGPU_LeftHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedSparseGPU_RightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedSparseGPU_Balanced_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(
        Test_TiledCachedSparseGPU_LeftAndRightHeavy_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedSparseGPU_LeftHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedSparseGPU_RightHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(Test_TiledCachedSparseGPU_Balanced_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(
        Test_TiledCachedSparseGPU_LeftAndRightHeavy_FltI16_B32_TSz2),
    TEST_LIST_ENTRY(
        Test_TiledCachedSparseGPU_LeftRightAndBalanced_DblI32_B32_TSz2),
    TEST_LIST_ENTRY(
        Test_TiledCachedSparseGPU_LeftRightAndBalanced_FltI16_B32_TSz2),

    // GPU Synthetic XGB tests -- scalar
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Reorg_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Reorg_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Reorg_Scalar_f32i16),

    // GPU Synthetic XGB tests -- tile4
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Sparse_Tile4),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Sparse_Tile4),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Sparse_Tile4),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Sparse_Tile4_f32i16),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Sparse_Tile4_f32i16),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Sparse_Tile4_f32i16),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Array_Tile4),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Array_Tile4),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Array_Tile4),
    TEST_LIST_ENTRY(Test_GPU_1TreeXGB_Array_Tile4_f32i16),
    TEST_LIST_ENTRY(Test_GPU_2TreeXGB_Array_Tile4_f32i16),
    TEST_LIST_ENTRY(Test_GPU_4TreeXGB_Array_Tile4_f32i16),

    // GPU Synthetic XGB tests -- Shared Forest -- Scalar
    TEST_LIST_ENTRY(Test_GPU_SharedForest_1TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_2TreeXGB_Reorg_Scalar),
    // TEST_LIST_ENTRY(Test_GPU_SharedForest_4TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_1TreeXGB_Reorg_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_2TreeXGB_Reorg_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_4TreeXGB_Reorg_Scalar_f32i16),

    TEST_LIST_ENTRY(Test_GPU_SharedForest_1TreeXGB_Array_Scalar),
    // TEST_LIST_ENTRY(Test_GPU_SharedForest_2TreeXGB_Array_Scalar),
    // TEST_LIST_ENTRY(Test_GPU_SharedForest_4TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_1TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_2TreeXGB_Array_Scalar_f32i16),
    // TEST_LIST_ENTRY(Test_GPU_SharedForest_4TreeXGB_Array_Scalar_f32i16),

    TEST_LIST_ENTRY(Test_GPU_SharedForest_1TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_2TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_4TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_1TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_2TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_SharedForest_4TreeXGB_Sparse_Scalar_f32i16),

    // GPU Synthetic XGB tests -- Cache Partial Forest -- Scalar
    // NOTE: These schedules are different than Tahoe's partial shared forest
    // schedule
    TEST_LIST_ENTRY(Test_GPU_CachePartialForest1Tree_2TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_CachePartialForest2Trees_4TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_CachePartialForest2Trees_2TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_CachePartialForest2Trees_4TreeXGB_Sparse_Scalar_f32i16),

    TEST_LIST_ENTRY(Test_GPU_CachePartialForest1Tree_2TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(Test_GPU_CachePartialForest2Trees_4TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_CachePartialForest2Trees_2TreeXGB_Reorg_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_CachePartialForest2Trees_4TreeXGB_Reorg_Scalar_f32i16),

    TEST_LIST_ENTRY(Test_GPU_CachePartialForest1Tree_2TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(Test_GPU_CachePartialForest1Tree_4TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_CachePartialForest2Trees_2TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_CachePartialForest2Trees_4TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_TwiceLeftRightBalanced_TahoeShdInpMultiRow_FltI16_B32),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_LeftRightAndBalanced_TahoeShdInpMultiRow_FltI16_B32),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_LeftRightAndBalanced_TahoeShdInp_FltI16_B32),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_TwiceLeftRightBalanced_IterShdPartialForest_FltI16_B32),
    TEST_LIST_ENTRY(Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Array_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_TahoeSharedDataStrategy_2TreeXGB_Reorg_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_TahoeSharedDataStrategy_4TreeXGB_Reorg_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Array_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Scalar_f32i16),
    TEST_LIST_ENTRY(
        Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Reorg_Scalar),
    TEST_LIST_ENTRY(
        Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Reorg_Scalar_f32i16),
    // GPU Tree parallelization tests - Tile size 4
    TEST_LIST_ENTRY(
        Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Tile4),
    TEST_LIST_ENTRY(
        Test_GPU_iterativeCachedPartialForestStrategy_4TreeXGB_Sparse_Tile4_f32i16),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_TwiceLeftRightBalanced_iterCachedPartialForest_NoCache_SharedReduce_FltI16_B64),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_TwiceLeftRightBalanced_iterCachedPartialForest_NoCache_SpecializedTreeLoop_FltI16_B64),
    // Tree Parallelization Multi-class tests
    TEST_LIST_ENTRY(
        Test_GPUCodeGeneration_Covtype_SparseRep_f32i16_B32_iterativeCachedPartialForestStrategy_NoCache),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleCachedTrees),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleCachedRows),
    TEST_LIST_ENTRY(
        Test_ScalarSparseGPU_TwiceLeftRightBalanced_AutoScheduleBasic),
    TEST_LIST_ENTRY(
        Test_GPUCodeGeneration_Covtype_SparseRep_f32i16_B32_iterativeCachedPartialForestStrategy_NoCache_SharedReduce),
};
#endif // TREEBEARD_GPU_SUPPORT

TestDescriptor sanityTestList[] = {
    TEST_LIST_ENTRY(Test_TileSize8_Abalone_TestInputs_MakeLeavesSameDepth),
    TEST_LIST_ENTRY(Test_TileSize8_AirlineOHE_TestInputs_MakeLeavesSameDepth),
    TEST_LIST_ENTRY(Test_TileSize8_Airline_TestInputs_MakeLeavesSameDepth),
    TEST_LIST_ENTRY(Test_TileSize8_Epsilon_TestInputs_MakeLeavesSameDepth),
    TEST_LIST_ENTRY(Test_TileSize8_Higgs_TestInputs_MakeLeavesSameDepth),
    TEST_LIST_ENTRY(Test_TileSize8_Year_TestInputs_MakeLeavesSameDepth),
    TEST_LIST_ENTRY(Test_TileSize8_CovType_TestInputs_MakeLeavesSameDepth),
    TEST_LIST_ENTRY(Test_TileSize8_Abalone_TestInputs_ParallelBatch),
    TEST_LIST_ENTRY(Test_TileSize8_Airline_TestInputs_ParallelBatch),
    TEST_LIST_ENTRY(Test_TileSize8_AirlineOHE_TestInputs_ParallelBatch),
    TEST_LIST_ENTRY(Test_TileSize8_Covtype_TestInputs_ParallelBatch),
    TEST_LIST_ENTRY(Test_TileSize8_Epsilon_TestInputs_ParallelBatch),
    TEST_LIST_ENTRY(Test_TileSize8_Higgs_TestInputs_ParallelBatch),
    TEST_LIST_ENTRY(Test_TileSize8_Year_TestInputs_ParallelBatch),
    TEST_LIST_ENTRY(Test_WalkPeeling_BalancedTree_TileSize2),
    TEST_LIST_ENTRY(
        Test_HybridTilingAndPeeling_RandomXGBoostJSONs_4Tree_FloatBatchSize4),
    TEST_LIST_ENTRY(
        Test_HybridTilingAndPeeling_RandomXGBoostJSONs_1Tree_FloatBatchSize4),
    TEST_LIST_ENTRY(
        Test_HybridTilingAndPeeling_RandomXGBoostJSONs_2Tree_FloatBatchSize4),
    TEST_LIST_ENTRY(
        Test_UniformAndHybridTilingAndPeeling_RandomXGBoostJSONs_2Tree_FloatBatchSize4),
    TEST_LIST_ENTRY(
        Test_UniformAndHybridTilingAndPeeling_RandomXGBoostJSONs_4Tree_FloatBatchSize4),
    TEST_LIST_ENTRY(Test_PeeledHybridProbabilisticTiling_TileSize8_Year),
    TEST_LIST_ENTRY(Test_PeeledHybridProbabilisticTiling_TileSize8_Epsilon),
    TEST_LIST_ENTRY(Test_PeeledHybridProbabilisticTiling_TileSize8_Higgs),
    TEST_LIST_ENTRY(Test_PeeledHybridProbabilisticTiling_TileSize8_AirlineOHE),
    TEST_LIST_ENTRY(Test_PeeledHybridProbabilisticTiling_TileSize8_Covtype),
    TEST_LIST_ENTRY(Test_PeeledHybridProbabilisticTiling_TileSize8_Airline),
    TEST_LIST_ENTRY(Test_PeeledHybridProbabilisticTiling_TileSize8_Abalone),

    // Remove extra hop tests
    TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Abalone),
    TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Airline),
    TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_AirlineOHE),
    TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Epsilon),
    TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Higgs),
    TEST_LIST_ENTRY(Test_SparseProbabilisticTiling_TileSize8_Year),
    TEST_LIST_ENTRY(Test_TileSize8_Higgs_TestInputs),
    TEST_LIST_ENTRY(Test_TileSize8_Year_TestInputs),
    TEST_LIST_ENTRY(
        Test_SparseUniformTiling_RandomXGBoostJSONs_1Tree_BatchSize4),
    TEST_LIST_ENTRY(
        Test_SparseUniformTiling_RandomXGBoostJSONs_2Trees_BatchSize4),
    TEST_LIST_ENTRY(
        Test_SparseUniformTiling_RandomXGBoostJSONs_4Trees_BatchSize4),
    TEST_LIST_ENTRY(
        Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int16TileShape),
    TEST_LIST_ENTRY(
        Test_SparseTiledCodeGeneration_LeftHeavy_BatchSize1_Int8TileShape),
    TEST_LIST_ENTRY(
        Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int8TileSize),
    TEST_LIST_ENTRY(
        Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1),
    TEST_LIST_ENTRY(
        Test_SparseTiledCodeGeneration_LeftAndRightHeavy_BatchSize1_Int16TileSize),
    TEST_LIST_ENTRY(Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1),
    TEST_LIST_ENTRY(
        Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int16TileShape),
    TEST_LIST_ENTRY(
        Test_SparseTiledCodeGeneration_RightHeavy_BatchSize1_Int8TileShape),
    TEST_LIST_ENTRY(
        Test_SparseCodeGeneration_RightHeavy_BatchSize1_I32ChildIdx),
    TEST_LIST_ENTRY(
        Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize1_I32ChildIdx),
    TEST_LIST_ENTRY(Test_SparseCodeGeneration_LeftHeavy_BatchSize2_I32ChildIdx),
    TEST_LIST_ENTRY(
        Test_SparseCodeGeneration_RightHeavy_BatchSize2_I32ChildIdx),
    TEST_LIST_ENTRY(
        Test_SparseCodeGeneration_RightAndLeftHeavy_BatchSize2_I32ChildIdx),
    TEST_LIST_ENTRY(Test_TileSize8_Airline),
    TEST_LIST_ENTRY(Test_TileSize8_Bosch),
    TEST_LIST_ENTRY(Test_TileSize8_Epsilon),
    TEST_LIST_ENTRY(Test_TileSize8_Higgs),
    TEST_LIST_ENTRY(Test_TileSize8_Year),
    TEST_LIST_ENTRY(Test_TileSize3_Abalone),
    TEST_LIST_ENTRY(Test_TileSize4_Abalone),
    TEST_LIST_ENTRY(
        Test_RandomXGBoostJSONs_1Tree_BatchSize8_TileSize2_4Pipelined),
    TEST_LIST_ENTRY(Test_SparseTileSize8_4Pipelined_Bosch),
    TEST_LIST_ENTRY(Test_TileSize8_Abalone_4Pipelined_TestInputs),
};

const size_t numTests = sizeof(testList) / sizeof(testList[0]);
const size_t numSanityTests =
    sizeof(sanityTestList) / sizeof(sanityTestList[0]);
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

bool RunTest(TestDescriptor test, TestArgs_t &args, size_t testNum) {
  std::string errStr;
  std::cout << white << testNum << ". Running test " << blue << test.m_testName
            << reset << ".... ";
  bool pass = false;
  // try
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  { pass = test.m_testFunc(args); }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  // catch s(...)
  // {
  // 	std::exception_ptr eptr = std::current_exception();
  // 	std::cout << "Crashed with exception ";
  // 	PrintExceptionInfo(eptr);
  // 	pass = false;
  // }
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count();
  std::cout << (pass ? green + "Passed" : red + "Failed")
            << white + " (Duration : " << duration << " ms)" << reset
            << std::endl;
  return pass;
}

void RunTestsImpl(TestDescriptor *testsToRun, size_t numberOfTests) {
  bool overallPass = true;

  std::cout << "Running Treebeard Tests " << std::endl << std::endl;
  int32_t numPassed = 0;
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  for (size_t i = 0; i < numberOfTests; ++i) {
    TestArgs_t args;

    // Disable sparse code generation by default
    decisionforest::UseSparseTreeRepresentation = false;
    mlir::decisionforest::ForestJSONReader::GetInstance().SetChildIndexBitWidth(
        -1);

    bool pass = RunTest(testsToRun[i], args, i + 1);
    numPassed += pass ? 1 : 0;
    overallPass = overallPass && pass;
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto totalTime =
      std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();

  std::cout << std::endl
            << boldBlue << underline << numPassed << "/" << numberOfTests
            << reset << white << " tests passed.";
  std::cout << underline
            << (overallPass ? boldGreen + "\nTest Suite Passed."
                            : boldRed + "\nTest Suite Failed.")
            << reset;
  std::cout << std::endl
            << "Total time taken : " << totalTime << " seconds." << std::endl
            << std::endl;
}


// Function to create a TestDescriptor dynamically
#define RED   "\033[31m"
#define RESET "\033[0m"

TestDescriptor createTestDescriptor(const std::string &testName) {
    auto it = testFuncMap.find(testName);
    if (it != testFuncMap.end()) {
        // Test found in the map, return the TestDescriptor
        return {testName, it->second};
    } else {
        // Print the error message in red and abort the program
        std::cerr << RED << "Error: "<< RESET <<"Test not found: " + testName  << std::endl;
        std::abort();  // Stops the program immediately
    }
}

void RunIndividualTests(const std::string &individualTestName) {
  // Create the TestDescriptor dynamically
  TestDescriptor testDesc = createTestDescriptor(individualTestName);

  // You can now use the test descriptor to run the test
  TestArgs_t args;
  bool pass = RunTest(testDesc, args, 1);
  std::cout << underline
            << (pass ? boldGreen + "\nTest Passed."
                     : boldRed + "\nTest Failed.")
            << reset;
}

void RunTests() { RunTestsImpl(testList, numTests); }

void RunSanityTests() { RunTestsImpl(sanityTestList, numSanityTests); }

} // namespace test
} // namespace TreeBeard