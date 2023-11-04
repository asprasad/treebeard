import os
import numpy
import treebeard
from functools import partial
from test_utils import treebeard_repo_dir, RunSingleGPUTestJIT, RunAllTests

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

def run_all_tests():
  RunSimpleGPUScheduleTests()
  RunOneRowAtATimeGPUScheduleTests()

if __name__ == "__main__":
  run_all_tests()


