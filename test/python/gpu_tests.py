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

def run_custom_schedule(test_name, rep, schedule_manipulator, tile_size=1):
  defaultTileSize8Options = treebeard.CompilerOptions(200, tile_size)
  defaultTileSize8MulticlassOptions = treebeard.CompilerOptions(200, tile_size)
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

def run_all_tests():
  RunSimpleGPUScheduleTests()

if __name__ == "__main__":
  run_all_tests()


