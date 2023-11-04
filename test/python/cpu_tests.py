import os
import numpy
import treebeard
from functools import partial
from test_utils import RunTestOnSingleModelTestInputsJIT, treebeard_repo_dir, RunSingleTestJIT, RunSingleTestJIT_TBContext, RunSingleTestJIT_ScheduleManipulation, RunAllTests

def SetStatsProfileCSVPath(options, modelName):
  profilesDir = os.path.join(os.path.join(treebeard_repo_dir, "xgb_models"), "profiles") 
  csvPath = os.path.join(profilesDir, modelName + ".test.csv")
  options.SetStatsProfileCSVPath(csvPath)
  return options

def RunPipeliningTests():
  invertLoopsTileSize8Options = treebeard.CompilerOptions(200, 8)
  # invertLoopsTileSize8Options.SetOneTreeAtATimeSchedule()
  invertLoopsTileSize8Options.SetPipelineWidth(8)
  invertLoopsTileSize8Options.SetReorderTreesByDepth(True)
  invertLoopsTileSize8Options.SetMakeAllLeavesSameDepth(1)
  
  invertLoopsTileSize8MulticlassOptions = treebeard.CompilerOptions(200, 8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)
  # invertLoopsTileSize8MulticlassOptions.SetOneTreeAtATimeSchedule();
  invertLoopsTileSize8MulticlassOptions.SetPipelineWidth(8)
  invertLoopsTileSize8MulticlassOptions.SetReorderTreesByDepth(True)
  invertLoopsTileSize8MulticlassOptions.SetMakeAllLeavesSameDepth(1)

  RunAllTests("one-tree-pipeline8-sparse", invertLoopsTileSize8Options, invertLoopsTileSize8MulticlassOptions, RunSingleTestJIT)

def RunParallelTests():
  batchSize = 200
  num_cores = 4
  tile_size = 8
  pipeline_width = 8
  invertLoopsTileSize8Options = treebeard.CompilerOptions(batchSize, tile_size)
  invertLoopsTileSize8Options.SetPipelineWidth(pipeline_width)
  invertLoopsTileSize8Options.SetReorderTreesByDepth(True)
  invertLoopsTileSize8Options.SetMakeAllLeavesSameDepth(1)
  invertLoopsTileSize8Options.SetNumberOfCores(num_cores)

  invertLoopsTileSize8MulticlassOptions = treebeard.CompilerOptions(batchSize, tile_size)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)
  invertLoopsTileSize8MulticlassOptions.SetPipelineWidth(pipeline_width)
  invertLoopsTileSize8MulticlassOptions.SetReorderTreesByDepth(True)
  invertLoopsTileSize8MulticlassOptions.SetMakeAllLeavesSameDepth(1)
  invertLoopsTileSize8MulticlassOptions.SetNumberOfCores(num_cores)

  RunAllTests("one-tree-par-4cores", invertLoopsTileSize8Options, invertLoopsTileSize8MulticlassOptions, RunSingleTestJIT)

def RunProbBasedTilingTests():
  probTilingOptions = treebeard.CompilerOptions(200, 8)
  probTilingOptions.SetTilingType(2) # prob tiling
  probTilingOptions.SetReorderTreesByDepth(True)

  probTilingMulticlassOptions = treebeard.CompilerOptions(200, 8)
  probTilingMulticlassOptions.SetReturnTypeWidth(8)
  probTilingMulticlassOptions.SetReturnTypeIsFloatType(False)

  assert RunTestOnSingleModelTestInputsJIT("abalone", SetStatsProfileCSVPath(probTilingOptions, "abalone"), "default-probtiling-sparse")
  assert RunTestOnSingleModelTestInputsJIT("airline", SetStatsProfileCSVPath(probTilingOptions, "airline"), "default-probtiling-sparse")
  assert RunTestOnSingleModelTestInputsJIT("airline-ohe", SetStatsProfileCSVPath(probTilingOptions, "airline-ohe"), "default-probtiling-sparse")
  assert RunTestOnSingleModelTestInputsJIT("covtype", SetStatsProfileCSVPath(probTilingMulticlassOptions, "covtype"), "default-probtiling-sparse", numpy.int8)
  assert RunTestOnSingleModelTestInputsJIT("epsilon", SetStatsProfileCSVPath(probTilingOptions, "epsilon"), "default-probtiling-sparse")
  assert RunTestOnSingleModelTestInputsJIT("higgs", SetStatsProfileCSVPath(probTilingOptions, "higgs"), "default-probtiling-sparse")
  assert RunTestOnSingleModelTestInputsJIT("letters", SetStatsProfileCSVPath(probTilingMulticlassOptions, "letters"), "default-probtiling-sparse", numpy.int8)
  assert RunTestOnSingleModelTestInputsJIT("year_prediction_msd", SetStatsProfileCSVPath(probTilingOptions, "year_prediction_msd"), "default-probtiling-sparse")

def RunBasicTests():
  defaultTileSize8Options = treebeard.CompilerOptions(200, 8)
  defaultTileSize8MulticlassOptions = treebeard.CompilerOptions(200, 8)
  defaultTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  defaultTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)

  RunAllTests("default-array", defaultTileSize8Options, defaultTileSize8MulticlassOptions, RunSingleTestJIT)

  invertLoopsTileSize8Options = treebeard.CompilerOptions(200, 8)
  invertLoopsTileSize8Options.SetOneTreeAtATimeSchedule()

  invertLoopsTileSize8MulticlassOptions = treebeard.CompilerOptions(200, 8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)
  invertLoopsTileSize8MulticlassOptions.SetOneTreeAtATimeSchedule()
  
  RunAllTests("one-tree-array", invertLoopsTileSize8Options, invertLoopsTileSize8MulticlassOptions, RunSingleTestJIT)

  treebeard.SetEnableSparseRepresentation(1)
  
  RunAllTests("one-tree-sparse", invertLoopsTileSize8Options, invertLoopsTileSize8MulticlassOptions, RunSingleTestJIT)
  RunProbBasedTilingTests()
  
  treebeard.SetEnableSparseRepresentation(0)

def RunTBContextTests():
  defaultTileSize8Options = treebeard.CompilerOptions(200, 8)
  defaultTileSize8MulticlassOptions = treebeard.CompilerOptions(200, 8)
  defaultTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  defaultTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)

  arrayRepSingleTestRunner = partial(RunSingleTestJIT_TBContext, representation="array", inputType="xgboost_json")
  sparseRepSingleTestRunner = partial(RunSingleTestJIT_TBContext, representation="sparse", inputType="xgboost_json")

  RunAllTests("default-array-tbcontext", defaultTileSize8Options, defaultTileSize8MulticlassOptions, arrayRepSingleTestRunner)

  RunAllTests("default-sparse-tbcontext", defaultTileSize8Options, defaultTileSize8MulticlassOptions, sparseRepSingleTestRunner)

  invertLoopsTileSize8Options = treebeard.CompilerOptions(200, 8)
  invertLoopsTileSize8Options.SetOneTreeAtATimeSchedule()

  invertLoopsTileSize8MulticlassOptions = treebeard.CompilerOptions(200, 8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  invertLoopsTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)
  invertLoopsTileSize8MulticlassOptions.SetOneTreeAtATimeSchedule();

  RunAllTests("one-tree-array-tbcontext", invertLoopsTileSize8Options, invertLoopsTileSize8MulticlassOptions, arrayRepSingleTestRunner)

  RunAllTests("one-tree-sparse-tbcontext", invertLoopsTileSize8Options, invertLoopsTileSize8MulticlassOptions, sparseRepSingleTestRunner)

def TileBatchLoopSchedule(schedule: treebeard.Schedule):
  batchIndex = schedule.GetBatchIndex()
  outerIndex = schedule.NewIndexVariable("b0")
  innerIndex = schedule.NewIndexVariable("b1")
  schedule.Tile(batchIndex, outerIndex, innerIndex, tileSize=4)

def TileTreeLoopSchedule(schedule: treebeard.Schedule):
  treeIndex = schedule.GetTreeIndex()
  outerIndex = schedule.NewIndexVariable("b0")
  innerIndex = schedule.NewIndexVariable("b1")
  schedule.Tile(treeIndex, outerIndex, innerIndex, tileSize=4)

def OneTreeAtATimeSchedule(schedule: treebeard.Schedule):
  tree_index = schedule.GetTreeIndex()
  batch_index = schedule.GetBatchIndex()
  index_order = [tree_index, batch_index]
  schedule.Reorder(index_order)

def run_custom_schedule(test_name, rep, schedule_manipulator):
  defaultTileSize8Options = treebeard.CompilerOptions(200, 8)
  defaultTileSize8MulticlassOptions = treebeard.CompilerOptions(200, 8)
  defaultTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  defaultTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)

  arrayRepSingleTestRunner_TileBatch = partial(RunSingleTestJIT_ScheduleManipulation,
                                               representation=rep,
                                               inputType="xgboost_json",
                                               scheduleManipulator=schedule_manipulator)

  RunAllTests(test_name, defaultTileSize8Options, defaultTileSize8MulticlassOptions, arrayRepSingleTestRunner_TileBatch)

def RunTileBatchLoopTests():
  run_custom_schedule("tiled_batch-array-tbcontext", "array", TileBatchLoopSchedule)

def RunTileTreeLoopTests():
  run_custom_schedule("tiled_tree-array-tbcontext", "array", TileTreeLoopSchedule)

def RuneOneTreeAtATimeTests():
  run_custom_schedule("one_tree_schedule-array-tbcontext", "array", OneTreeAtATimeSchedule)

def ScheduleTest():
  RuneOneTreeAtATimeTests()
  RunTileBatchLoopTests()
  RunTileTreeLoopTests()

def run_all_tests():
  ScheduleTest()
  RunTBContextTests()
  RunBasicTests()

  treebeard.SetEnableSparseRepresentation(1)

  RunPipeliningTests()
  RunParallelTests()

  treebeard.SetEnableSparseRepresentation(0)

if __name__ == "__main__":
  run_all_tests()