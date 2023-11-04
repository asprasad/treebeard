import os
import numpy
import treebeard
from functools import partial
from test_utils import RunSingleGPUTestJIT, RunAllTests, RunSelectedTests

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
  run_custom_schedule_selected("tree-at-a-time-cache-row_gpu_schedule-array", "gpu_array", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("tree-at-a-time-cache-row_schedule-sparse", "gpu_sparse", gpu_schedule, selected_tests=selected_tests)
  run_custom_schedule_selected("tree-at-a-time-cache_schedule-reorg", "gpu_reorg", gpu_schedule, selected_tests=selected_tests)

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

def run_all_tests():
  RunSimpleGPUScheduleTests()
  RunOneRowAtATimeGPUScheduleTests()
  RunOneTreeAtATimeGPUScheduleTests()
  RunOneTreeAtATimeCachedRowGPUScheduleTests()
  RunParallelizeOnTreeGPUScheduleTests()
  RunParallelizeOnTreeAndSpecializeGPUScheduleTests()

if __name__ == "__main__":
  run_all_tests()


