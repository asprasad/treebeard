import os
import numpy
import treebeard
from functools import partial
from test_utils import RunSingleGPUTestJIT, RunAllTests, RunSelectedTests, RunSingleGPUTestAutoSchedule, RunSingleGPUTestAutoTuneHeuristic

def SimpleGPUSchedule(schedule: treebeard.Schedule, rows_per_threadblock: int):
  batchIndex = schedule.GetBatchIndex()
  threadBlockIndex = schedule.NewIndexVariable("b0")
  threadIndex = schedule.NewIndexVariable("b1")
  schedule.Tile(batchIndex, threadBlockIndex, threadIndex, tileSize=rows_per_threadblock)
  threadBlockIndex.set_gpu_dimension(treebeard.GPUConstruct.Grid, treebeard.Dimension.X)
  threadIndex.set_gpu_dimension(treebeard.GPUConstruct.ThreadBlock, treebeard.Dimension.X)

def OneRowAtATimeGPUSchedule(schedule: treebeard.Schedule, rows_per_threadblock: int, rows_per_thread: int):
  batchIndex = schedule.GetBatchIndex()
  threadBlockIndex = schedule.NewIndexVariable("b0")
  threadIndexTemp = schedule.NewIndexVariable("b1_temp")
  threadIndex = schedule.NewIndexVariable("b1_outer")
  perThreadIndex = schedule.NewIndexVariable("b1_inner")
  schedule.Tile(batchIndex, threadBlockIndex, threadIndexTemp, tileSize=rows_per_threadblock)
  schedule.Tile(threadIndexTemp, threadIndex, perThreadIndex, tileSize=rows_per_thread)
  threadBlockIndex.set_gpu_dimension(treebeard.GPUConstruct.Grid, treebeard.Dimension.X)
  threadIndex.set_gpu_dimension(treebeard.GPUConstruct.ThreadBlock, treebeard.Dimension.X)

def OneTreeAtATimeGPUSchedule(schedule: treebeard.Schedule, rows_per_threadblock: int, rows_per_thread: int):
  batchIndex = schedule.GetBatchIndex()
  treeIndex = schedule.GetTreeIndex()
  threadBlockIndex = schedule.NewIndexVariable("b0")
  threadIndexTemp = schedule.NewIndexVariable("b1_temp")
  threadIndex = schedule.NewIndexVariable("b1_outer")
  perThreadIndex = schedule.NewIndexVariable("b1_inner")
  schedule.Tile(batchIndex, threadBlockIndex, threadIndexTemp, tileSize=rows_per_threadblock)
  schedule.Tile(threadIndexTemp, threadIndex, perThreadIndex, tileSize=rows_per_thread)
  schedule.Reorder([threadBlockIndex, threadIndex, treeIndex, perThreadIndex])
  threadBlockIndex.set_gpu_dimension(treebeard.GPUConstruct.Grid, treebeard.Dimension.X)
  threadIndex.set_gpu_dimension(treebeard.GPUConstruct.ThreadBlock, treebeard.Dimension.X)

def OneTreeAtATimeGPUSchedule_CachedRow(schedule: treebeard.Schedule, rows_per_threadblock: int, rows_per_thread: int):
  batchIndex = schedule.GetBatchIndex()
  treeIndex = schedule.GetTreeIndex()
  threadBlockIndex = schedule.NewIndexVariable("b0")
  threadIndexTemp = schedule.NewIndexVariable("b1_temp")
  threadIndex = schedule.NewIndexVariable("b1_outer")
  perThreadIndex = schedule.NewIndexVariable("b1_inner")
  schedule.Tile(batchIndex, threadBlockIndex, threadIndexTemp, tileSize=rows_per_threadblock)
  schedule.Tile(threadIndexTemp, threadIndex, perThreadIndex, tileSize=rows_per_thread)
  schedule.Reorder([threadBlockIndex, threadIndex, treeIndex, perThreadIndex])
  schedule.Cache(threadBlockIndex)
  threadBlockIndex.set_gpu_dimension(treebeard.GPUConstruct.Grid, treebeard.Dimension.X)
  threadIndex.set_gpu_dimension(treebeard.GPUConstruct.ThreadBlock, treebeard.Dimension.X)

def OneTreeAtATimeGPUSchedule_CachedTree(schedule: treebeard.Schedule, rows_per_threadblock: int, rows_per_thread: int):
  batchIndex = schedule.GetBatchIndex()
  treeIndex = schedule.GetTreeIndex()
  threadBlockIndex = schedule.NewIndexVariable("b0")
  threadIndexTemp = schedule.NewIndexVariable("b1_temp")
  threadIndex = schedule.NewIndexVariable("b1_outer")
  perThreadIndex = schedule.NewIndexVariable("b1_inner")
  schedule.Tile(batchIndex, threadBlockIndex, threadIndexTemp, tileSize=rows_per_threadblock)
  schedule.Tile(threadIndexTemp, threadIndex, perThreadIndex, tileSize=rows_per_thread)
  schedule.Reorder([threadBlockIndex, threadIndex, treeIndex, perThreadIndex])
  schedule.Cache(treeIndex)
  threadBlockIndex.set_gpu_dimension(treebeard.GPUConstruct.Grid, treebeard.Dimension.X)
  threadIndex.set_gpu_dimension(treebeard.GPUConstruct.ThreadBlock, treebeard.Dimension.X)

def OneTreeAtATimeGPUSchedule_CachedTreeAndRow(schedule: treebeard.Schedule, rows_per_threadblock: int, rows_per_thread: int):
  batchIndex = schedule.GetBatchIndex()
  treeIndex = schedule.GetTreeIndex()
  threadBlockIndex = schedule.NewIndexVariable("b0")
  threadIndexTemp = schedule.NewIndexVariable("b1_temp")
  threadIndex = schedule.NewIndexVariable("b1_outer")
  perThreadIndex = schedule.NewIndexVariable("b1_inner")
  schedule.Tile(batchIndex, threadBlockIndex, threadIndexTemp, tileSize=rows_per_threadblock)
  schedule.Tile(threadIndexTemp, threadIndex, perThreadIndex, tileSize=rows_per_thread)
  schedule.Reorder([threadBlockIndex, threadIndex, treeIndex, perThreadIndex])
  schedule.Cache(treeIndex)
  schedule.Cache(threadBlockIndex)
  threadBlockIndex.set_gpu_dimension(treebeard.GPUConstruct.Grid, treebeard.Dimension.X)
  threadIndex.set_gpu_dimension(treebeard.GPUConstruct.ThreadBlock, treebeard.Dimension.X)

def OneTreeAtATimeGPUSchedule_CacheMultipleTrees(schedule: treebeard.Schedule, rows_per_threadblock: int, rows_per_thread: int, trees_to_cache: int):
  batchIndex = schedule.GetBatchIndex()
  treeIndex = schedule.GetTreeIndex()
  threadBlockIndex = schedule.NewIndexVariable("b0")
  threadIndexTemp = schedule.NewIndexVariable("b1_temp")
  threadIndex = schedule.NewIndexVariable("b1_outer")
  perThreadIndex = schedule.NewIndexVariable("b1_inner")
  schedule.Tile(batchIndex, threadBlockIndex, threadIndexTemp, tileSize=rows_per_threadblock)
  schedule.Tile(threadIndexTemp, threadIndex, perThreadIndex, tileSize=rows_per_thread)

  tree_inner = schedule.NewIndexVariable("t1_inner")
  tree_outer = schedule.NewIndexVariable("t1_outer")
  schedule.Tile(treeIndex, tree_outer, tree_inner, tileSize=trees_to_cache)

  schedule.Reorder([threadBlockIndex, threadIndex, tree_outer, tree_inner, perThreadIndex])
  schedule.Cache(tree_inner)
  threadBlockIndex.set_gpu_dimension(treebeard.GPUConstruct.Grid, treebeard.Dimension.X)
  threadIndex.set_gpu_dimension(treebeard.GPUConstruct.ThreadBlock, treebeard.Dimension.X)

def SplitTreesAcrossThreadBlocksGPUSchedule(schedule: treebeard.Schedule, rows_per_threadblock: int, trees_per_thread: int):
  batchIndex = schedule.GetBatchIndex()
  treeIndex = schedule.GetTreeIndex()
  threadBlockIndex = schedule.NewIndexVariable("b0")
  perThreadRowIndex = schedule.NewIndexVariable("b1")
  schedule.Tile(batchIndex, threadBlockIndex, perThreadRowIndex, tileSize=rows_per_threadblock)
  threadBlockIndex.set_gpu_dimension(treebeard.GPUConstruct.Grid, treebeard.Dimension.X)

  parallelTreeIndex = schedule.NewIndexVariable("t0")
  perThreadTreeIndex = schedule.NewIndexVariable("t1")
  schedule.Tile(treeIndex, parallelTreeIndex, perThreadTreeIndex, tileSize=trees_per_thread)
  parallelTreeIndex.set_gpu_dimension(treebeard.GPUConstruct.ThreadBlock, treebeard.Dimension.X)

  schedule.Reorder([threadBlockIndex, parallelTreeIndex, perThreadTreeIndex, perThreadRowIndex])

def SplitTreesAcrossThreadsAndSpecializeGPUSchedule(schedule: treebeard.Schedule, rows_per_threadblock: int, trees_per_thread: int):
  batchIndex = schedule.GetBatchIndex()
  treeIndex = schedule.GetTreeIndex()
  threadBlockIndex = schedule.NewIndexVariable("b0")
  perThreadRowIndex = schedule.NewIndexVariable("b1")
  schedule.Tile(batchIndex, threadBlockIndex, perThreadRowIndex, tileSize=rows_per_threadblock)
  threadBlockIndex.set_gpu_dimension(treebeard.GPUConstruct.Grid, treebeard.Dimension.X)

  parallelTreeIndex = schedule.NewIndexVariable("t0")
  perThreadTreeIndex = schedule.NewIndexVariable("t1")
  schedule.Tile(treeIndex, parallelTreeIndex, perThreadTreeIndex, tileSize=trees_per_thread)
  parallelTreeIndex.set_gpu_dimension(treebeard.GPUConstruct.ThreadBlock, treebeard.Dimension.X)
  schedule.Reorder([threadBlockIndex, parallelTreeIndex, perThreadTreeIndex, perThreadRowIndex])

  schedule.Specialize(parallelTreeIndex)

def run_custom_schedule(test_name, rep, schedule_manipulator, tile_size=1, batch_size=200):
  defaultTileSize8Options = treebeard.CompilerOptions(batch_size, tile_size)
  defaultTileSize8MulticlassOptions = treebeard.CompilerOptions(batch_size, tile_size)
  defaultTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  defaultTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)

  arrayRepSingleTestRunner_TileBatch = partial(RunSingleGPUTestJIT,
                                               representation=rep,
                                               inputType="xgboost_json",
                                               scheduleManipulator=schedule_manipulator)

  RunAllTests(test_name, defaultTileSize8Options, defaultTileSize8MulticlassOptions, arrayRepSingleTestRunner_TileBatch)

def run_custom_schedule_selected(test_name, rep, schedule_manipulator, tile_size=1, batch_size=200, selected_tests=[]):
  defaultTileSize8Options = treebeard.CompilerOptions(batch_size, tile_size)
  defaultTileSize8MulticlassOptions = treebeard.CompilerOptions(batch_size, tile_size)
  defaultTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  defaultTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)

  arrayRepSingleTestRunner_TileBatch = partial(RunSingleGPUTestJIT,
                                               representation=rep,
                                               inputType="xgboost_json",
                                               scheduleManipulator=schedule_manipulator)

  RunSelectedTests(test_name, defaultTileSize8Options, defaultTileSize8MulticlassOptions, arrayRepSingleTestRunner_TileBatch, selected_tests)

def run_auto_schedule_test(test_name, rep, gpuAutoScheduleOptions, tile_size=1, batch_size=200):
  defaultTileSize8Options = treebeard.CompilerOptions(batch_size, tile_size)
  defaultTileSize8Options.SetMakeAllLeavesSameDepth(1)
  defaultTileSize8MulticlassOptions = treebeard.CompilerOptions(batch_size, tile_size)
  defaultTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  defaultTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)
  defaultTileSize8MulticlassOptions.SetMakeAllLeavesSameDepth(1)

  arrayRepSingleTestRunner_TileBatch = partial(RunSingleGPUTestAutoSchedule,
                                               representation=rep,
                                               inputType="xgboost_json",
                                               gpuAutoScheduleOptions=gpuAutoScheduleOptions)

  RunAllTests(test_name, defaultTileSize8Options, defaultTileSize8MulticlassOptions, arrayRepSingleTestRunner_TileBatch)

def run_auto_tune_heuristic_test(test_name, tile_size=1, batch_size=256):
  defaultTileSize8Options = treebeard.CompilerOptions(batch_size, tile_size)
  defaultTileSize8Options.SetMakeAllLeavesSameDepth(1)
  defaultTileSize8MulticlassOptions = treebeard.CompilerOptions(batch_size, tile_size)
  defaultTileSize8MulticlassOptions.SetReturnTypeWidth(8)
  defaultTileSize8MulticlassOptions.SetReturnTypeIsFloatType(False)
  defaultTileSize8MulticlassOptions.SetMakeAllLeavesSameDepth(1)

  arrayRepSingleTestRunner_TileBatch = partial(RunSingleGPUTestAutoTuneHeuristic,
                                               inputType="xgboost_json")

  RunAllTests(test_name, defaultTileSize8Options, defaultTileSize8MulticlassOptions, arrayRepSingleTestRunner_TileBatch)

def RunSimpleGPUScheduleTests():
  gpu_schedule_20rows = partial(SimpleGPUSchedule, rows_per_threadblock=20)
  run_custom_schedule("simple_gpu_schedule-array", "gpu_array", gpu_schedule_20rows)
  run_custom_schedule("simple_gpu_schedule-sparse", "gpu_sparse", gpu_schedule_20rows)
  run_custom_schedule("simple_gpu_schedule-reorg", "gpu_reorg", gpu_schedule_20rows)  

def RunOneRowAtATimeGPUScheduleTests():
  gpu_schedule = partial(OneRowAtATimeGPUSchedule, rows_per_threadblock=40, rows_per_thread=2)
  run_custom_schedule("row-at-a-time_gpu_schedule-array", "gpu_array", gpu_schedule)
  run_custom_schedule("row-at-a-time_schedule-sparse", "gpu_sparse", gpu_schedule)
  run_custom_schedule("row-at-a-time_schedule-reorg", "gpu_reorg", gpu_schedule)  

def RunOneTreeAtATimeGPUScheduleTests():
  gpu_schedule = partial(OneTreeAtATimeGPUSchedule, rows_per_threadblock=40, rows_per_thread=2)
  run_custom_schedule("tree-at-a-time_gpu_schedule-array", "gpu_array", gpu_schedule)
  run_custom_schedule("tree-at-a-time_schedule-sparse", "gpu_sparse", gpu_schedule)
  run_custom_schedule("tree-at-a-time_schedule-reorg", "gpu_reorg", gpu_schedule)  

def RunOneTreeAtATimeCachedRowGPUScheduleTests():
  # "airline-ohe", "epsilon" omitted due to shared memory size constraints
  selected_tests = ["abalone", "airline", "covtype", "higgs", "letters", "year_prediction_msd",]
  gpu_schedule = partial(OneTreeAtATimeGPUSchedule_CachedRow, rows_per_threadblock=40, rows_per_thread=2)
  run_custom_schedule_selected("tree-at-a-time-cache-row_schedule-array", "gpu_array", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("tree-at-a-time-cache-row_schedule-sparse", "gpu_sparse", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("tree-at-a-time-cache-row_schedule-reorg", "gpu_reorg", gpu_schedule, selected_tests=selected_tests)

def RunOneTreeAtATimeCachedTreeGPUScheduleTests():
  selected_tests = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs",  "letters", "year_prediction_msd",]
  gpu_schedule = partial(OneTreeAtATimeGPUSchedule_CachedTree, rows_per_threadblock=40, rows_per_thread=2)
  run_custom_schedule_selected("tree-at-a-time-cache-tree_schedule-array", "gpu_array", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("tree-at-a-time-cache-tree_schedule-sparse", "gpu_sparse", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("tree-at-a-time-cache-tree_schedule-reorg", "gpu_reorg", gpu_schedule, selected_tests=selected_tests)

def RunOneTreeAtATimeCacheRowAndTreeGPUScheduleTests():
  # "airline-ohe", "epsilon" omitted due to shared memory size constraints
  selected_tests = ["abalone", "airline", "covtype", "higgs", "letters", "year_prediction_msd",]
  gpu_schedule = partial(OneTreeAtATimeGPUSchedule_CachedTreeAndRow, rows_per_threadblock=40, rows_per_thread=2)
  run_custom_schedule_selected("tree-at-a-time-cache-row-and-tree_schedule-array", "gpu_array", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("tree-at-a-time-cache-row-and-tree_schedule-sparse", "gpu_sparse", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("tree-at-a-time-cache-row-and-tree_schedule-reorg", "gpu_reorg", gpu_schedule, selected_tests=selected_tests)

def RunOneTreeAtATimeCache4TreesGPUScheduleTests():
  selected_tests = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs",  "letters", "year_prediction_msd",]
  gpu_schedule = partial(OneTreeAtATimeGPUSchedule_CacheMultipleTrees, rows_per_threadblock=40, rows_per_thread=2, trees_to_cache=4)
  run_custom_schedule_selected("tree-at-a-time-cache-4-tree_schedule-array", "gpu_array", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("tree-at-a-time-cache-4-tree_schedule-sparse", "gpu_sparse", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("tree-at-a-time-cache-4-tree_schedule-reorg", "gpu_reorg", gpu_schedule, selected_tests=selected_tests)

def RunParallelizeOnTreeGPUScheduleTests():
  selected_tests = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs", "year_prediction_msd",]
  gpu_schedule = partial(SplitTreesAcrossThreadBlocksGPUSchedule, rows_per_threadblock=20, trees_per_thread=10)
  run_custom_schedule_selected("par-tree-tree-at-a-time_gpu_schedule-array", "gpu_array", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("par-tree-tree-at-a-time_schedule-sparse", "gpu_sparse", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("par-tree-tree-at-a-time_schedule-reorg", "gpu_reorg", gpu_schedule, selected_tests=selected_tests)

  selected_tests = ["letters"]
  gpu_schedule = partial(SplitTreesAcrossThreadBlocksGPUSchedule, rows_per_threadblock=20, trees_per_thread=500)
  run_custom_schedule_selected("par-tree-tree-at-a-time_gpu_schedule-array", "gpu_array", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("par-tree-tree-at-a-time_schedule-sparse", "gpu_sparse", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("par-tree-tree-at-a-time_schedule-reorg", "gpu_reorg", gpu_schedule, selected_tests=selected_tests)

def RunParallelizeOnTreeAndSpecializeGPUScheduleTests():
  selected_tests = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs", "year_prediction_msd",]
  gpu_schedule = partial(SplitTreesAcrossThreadsAndSpecializeGPUSchedule, rows_per_threadblock=20, trees_per_thread=10)
  run_custom_schedule_selected("par-tree-tree-at-a-time_gpu_schedule-array", "gpu_array", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("par-tree-tree-at-a-time_schedule-sparse", "gpu_sparse", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("par-tree-tree-at-a-time_schedule-reorg", "gpu_reorg", gpu_schedule, selected_tests=selected_tests)

  selected_tests = ["letters"]
  gpu_schedule = partial(SplitTreesAcrossThreadsAndSpecializeGPUSchedule, rows_per_threadblock=20, trees_per_thread=500)
  run_custom_schedule_selected("par-tree-tree-at-a-time_gpu_schedule-array", "gpu_array", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("par-tree-tree-at-a-time_schedule-sparse", "gpu_sparse", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("par-tree-tree-at-a-time_schedule-reorg", "gpu_reorg", gpu_schedule, selected_tests=selected_tests)

def RunGPUAutoScheduleTestsNoCacheNoUnroll():
  batchSize = 200
  gpuAutoScheduleOptions = treebeard.GPUAutoScheduleOptions()
  gpuAutoScheduleOptions.NumberOfRowsPerThreadBlock(20)
  gpuAutoScheduleOptions.NumberOfRowsPerThread(2)
  gpuAutoScheduleOptions.UnrollTreeWalks(False)
  gpuAutoScheduleOptions.CacheRows(False)
  gpuAutoScheduleOptions.CacheTrees(False)
  gpuAutoScheduleOptions.NumberOfTreeThreads(1)
  gpuAutoScheduleOptions.NumberOfTreesAtATime(1)

  # selected_tests = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs",  "letters", "year_prediction_msd",]
  run_auto_schedule_test("no-cache-autoschedule-array", "gpu_array", gpuAutoScheduleOptions, batch_size=batchSize)
  run_auto_schedule_test("no-cache-autoschedule-sparse", "gpu_sparse", gpuAutoScheduleOptions, batch_size=batchSize)
  run_auto_schedule_test("no-cache-autoschedule-reorg", "gpu_reorg", gpuAutoScheduleOptions, batch_size=batchSize)

def RunGPUAutoScheduleTestsNoCacheUnroll():
  batchSize = 200
  gpuAutoScheduleOptions = treebeard.GPUAutoScheduleOptions()
  gpuAutoScheduleOptions.NumberOfRowsPerThreadBlock(20)
  gpuAutoScheduleOptions.NumberOfRowsPerThread(2)
  gpuAutoScheduleOptions.UnrollTreeWalks(True)
  gpuAutoScheduleOptions.CacheRows(False)
  gpuAutoScheduleOptions.CacheTrees(False)
  gpuAutoScheduleOptions.NumberOfTreeThreads(1)
  gpuAutoScheduleOptions.NumberOfTreesAtATime(1)

  # selected_tests = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs",  "letters", "year_prediction_msd",]
  run_auto_schedule_test("no-cache-autoschedule-unroll-array", "gpu_array", gpuAutoScheduleOptions, batch_size=batchSize)
  run_auto_schedule_test("no-cache-autoschedule-unroll-sparse", "gpu_sparse", gpuAutoScheduleOptions, batch_size=batchSize)
  run_auto_schedule_test("no-cache-autoschedule-unroll-reorg", "gpu_reorg", gpuAutoScheduleOptions, batch_size=batchSize)

def RunGPUAutoScheduleTestsNoCacheUnrollParallelTrees():
  batchSize = 200
  gpuAutoScheduleOptions = treebeard.GPUAutoScheduleOptions()
  gpuAutoScheduleOptions.NumberOfRowsPerThreadBlock(20)
  gpuAutoScheduleOptions.NumberOfRowsPerThread(2)
  gpuAutoScheduleOptions.UnrollTreeWalks(True)
  gpuAutoScheduleOptions.CacheRows(False)
  gpuAutoScheduleOptions.CacheTrees(False)
  gpuAutoScheduleOptions.NumberOfTreeThreads(4)
  gpuAutoScheduleOptions.NumberOfTreesAtATime(1)

  # selected_tests = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs",  "letters", "year_prediction_msd",]
  run_auto_schedule_test("no-cache-autoschedule-unroll-treepar-array", "gpu_array", gpuAutoScheduleOptions, batch_size=batchSize)
  run_auto_schedule_test("no-cache-autoschedule-unroll-treepar-sparse", "gpu_sparse", gpuAutoScheduleOptions, batch_size=batchSize)
  run_auto_schedule_test("no-cache-autoschedule-unroll-treepar-reorg", "gpu_reorg", gpuAutoScheduleOptions, batch_size=batchSize)

def RunGPUAutoScheduleTestsUnrollAndInterleaveParallelTrees():
  batchSize = 200
  gpuAutoScheduleOptions = treebeard.GPUAutoScheduleOptions()
  gpuAutoScheduleOptions.NumberOfRowsPerThreadBlock(20)
  gpuAutoScheduleOptions.NumberOfRowsPerThread(1)
  gpuAutoScheduleOptions.UnrollTreeWalks(True)
  gpuAutoScheduleOptions.CacheRows(False)
  gpuAutoScheduleOptions.CacheTrees(False)
  gpuAutoScheduleOptions.NumberOfTreeThreads(4)
  gpuAutoScheduleOptions.NumberOfTreesAtATime(1)
  gpuAutoScheduleOptions.TreeWalkInterleaveFactor(2)

  # selected_tests = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs",  "letters", "year_prediction_msd",]
  run_auto_schedule_test("no-cache-autoschedule-unroll-interleave-treepar-array", "gpu_array", gpuAutoScheduleOptions, batch_size=batchSize)
  run_auto_schedule_test("no-cache-autoschedule-unroll-interleave-treepar-sparse", "gpu_sparse", gpuAutoScheduleOptions, batch_size=batchSize)
  run_auto_schedule_test("no-cache-autoschedule-unroll-interleave-treepar-reorg", "gpu_reorg", gpuAutoScheduleOptions, batch_size=batchSize)

def RunGPUAutoTuneHeuristicTests():
  run_auto_tune_heuristic_test("auto-tune-heuristic")

def run_all_tests():
  RunSimpleGPUScheduleTests()
  RunOneRowAtATimeGPUScheduleTests()
  RunOneTreeAtATimeGPUScheduleTests()
  RunOneTreeAtATimeCachedRowGPUScheduleTests()
  RunParallelizeOnTreeGPUScheduleTests()
  RunParallelizeOnTreeAndSpecializeGPUScheduleTests()
  RunOneTreeAtATimeCachedTreeGPUScheduleTests()
  RunOneTreeAtATimeCacheRowAndTreeGPUScheduleTests()
  RunOneTreeAtATimeCache4TreesGPUScheduleTests()
  RunGPUAutoScheduleTestsNoCacheNoUnroll()
  RunGPUAutoScheduleTestsNoCacheUnroll()
  RunGPUAutoScheduleTestsNoCacheUnrollParallelTrees()
  RunGPUAutoScheduleTestsUnrollAndInterleaveParallelTrees()
  RunGPUAutoTuneHeuristicTests()

if __name__ == "__main__":
  run_all_tests()


