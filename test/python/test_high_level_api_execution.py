import os
import json
import subprocess
import sys
import tempfile
import unittest

import numpy
import xgboost


treebeard_repo_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(treebeard_repo_dir, "src", "python"))

import treebeard  # pyright: ignore[reportMissingImports]

try:
  import onnx
except ImportError:
  onnx = None


def load_fixture(csv_path, num_rows=13):
  data = numpy.loadtxt(csv_path, delimiter=",", max_rows=num_rows)
  return (numpy.asarray(data[:, :-1], dtype=numpy.float32, order="C"),
          numpy.asarray(data[:, -1], dtype=numpy.float32))


def has_available_gpu():
  if not treebeard.treebeardAPI.gpu_support:
    return False
  try:
    nvidia_smi = subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
    if nvidia_smi.returncode == 0 and "GPU" in nvidia_smi.stdout:
      return True
  except (FileNotFoundError, subprocess.TimeoutExpired):
    pass
  try:
    rocminfo = subprocess.run(
        ["rocminfo"], capture_output=True, text=True, timeout=5)
    return rocminfo.returncode == 0 and "Agent" in rocminfo.stdout
  except (FileNotFoundError, subprocess.TimeoutExpired):
    return False


GPU_AVAILABLE = has_available_gpu()


def tile_batch_schedule(schedule):
  batch = schedule.GetBatchIndex()
  batch_outer = schedule.NewIndexVariable("batch_outer")
  batch_inner = schedule.NewIndexVariable("batch_inner")
  schedule.Tile(batch, batch_outer, batch_inner, tileSize=4)


def vectorize_batch_schedule(schedule):
  batch = schedule.GetBatchIndex()
  tree = schedule.GetTreeIndex()
  batch_outer = schedule.NewIndexVariable("batch_outer")
  batch_inner = schedule.NewIndexVariable("batch_inner")
  schedule.Tile(batch, batch_outer, batch_inner, tileSize=4)
  schedule.Reorder([tree, batch_inner])
  schedule.Simdize(batch_inner)


def reorder_tree_before_batch_schedule(schedule):
  schedule.Reorder([schedule.GetTreeIndex(), schedule.GetBatchIndex()])


def gpu_batch_schedule(schedule):
  batch = schedule.GetBatchIndex()
  block = schedule.NewIndexVariable("block")
  thread = schedule.NewIndexVariable("thread")
  schedule.Tile(batch, block, thread, tileSize=8)
  block.set_gpu_dimension(treebeard.GPUConstruct.Grid, treebeard.Dimension.X)
  thread.set_gpu_dimension(
      treebeard.GPUConstruct.ThreadBlock, treebeard.Dimension.X)


def gpu_tree_reduction_schedule(schedule, reduction):
  batch = schedule.GetBatchIndex()
  tree = schedule.GetTreeIndex()
  block = schedule.NewIndexVariable("block")
  row = schedule.NewIndexVariable("row")
  tree_thread = schedule.NewIndexVariable("tree_thread")
  tree_inner = schedule.NewIndexVariable("tree_inner")

  schedule.Tile(batch, block, row, tileSize=8)
  schedule.Tile(tree, tree_thread, tree_inner, tileSize=4)
  schedule.Reorder([block, tree_thread, tree_inner, row])
  block.set_gpu_dimension(treebeard.GPUConstruct.Grid, treebeard.Dimension.X)
  tree_thread.set_gpu_dimension(
      treebeard.GPUConstruct.ThreadBlock, treebeard.Dimension.X)
  reduction(schedule, tree_thread)


def atomic_gpu_reduction(schedule, index):
  schedule.AtomicReduce(index)


def vector_gpu_reduction(schedule, index):
  schedule.VectorReduce(index, width=2)


def shared_gpu_reduction(schedule, index):
  schedule.SharedReduce(index)


class HighLevelAPIExecutionTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.xgboost_model_path = os.path.join(
        treebeard_repo_dir, "xgb_models", "abalone_xgb_model_1.7.6.json")
    cls.xgboost_csv_path = os.path.join(
        treebeard_repo_dir, "xgb_models",
        "abalone_xgb_model_save.json.test.sampled.csv")

  def load_xgboost_model(self):
    model = xgboost.XGBRegressor()
    model.load_model(self.xgboost_model_path)
    return model

  def assert_fixture_outputs(self, kernel, csv_path):
    inputs, expected = load_fixture(csv_path)
    actual = kernel.run(inputs)

    self.assertEqual(actual.shape, expected.shape)
    numpy.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  def assert_xgboost_schedule(self, schedule, *, target="cpu",
                              representation="sparse", batch_size=8):
    kernel = treebeard.compile(
        self.load_xgboost_model(), batch_size=batch_size, tile_size=1,
        target=target, representation=representation, schedule=schedule)
    self.addCleanup(kernel.close)
    self.assert_fixture_outputs(kernel, self.xgboost_csv_path)

  def test_compile_and_run_xgboost_estimator(self):
    kernel = treebeard.compile(
        self.load_xgboost_model(), batch_size=8, tile_size=1)
    self.addCleanup(kernel.close)

    self.assert_fixture_outputs(kernel, self.xgboost_csv_path)

  def test_unsupported_xgboost_objective_raises(self):
    with open(self.xgboost_model_path, encoding="utf-8") as model_file:
      model = json.load(model_file)
    model["learner"]["objective"]["name"] = "rank:pairwise"

    with tempfile.TemporaryDirectory() as directory:
      model_path = os.path.join(directory, "unsupported.json")
      with open(model_path, "w", encoding="utf-8") as model_file:
        json.dump(model, model_file)
      with self.assertRaisesRegex(
          treebeard.UnsupportedModelError, "rank:pairwise"):
        treebeard.compile(model_path, batch_size=8)

  def test_compile_and_run_with_batch_tiling(self):
    self.assert_xgboost_schedule(tile_batch_schedule)

  def test_compile_and_run_with_vectorization(self):
    self.assert_xgboost_schedule(vectorize_batch_schedule)

  def test_compile_and_run_with_tree_before_batch_loop_order(self):
    self.assert_xgboost_schedule(reorder_tree_before_batch_schedule)

  @unittest.skipUnless(
      GPU_AVAILABLE, "requires a GPU-enabled runtime and available GPU")
  def test_compile_and_run_on_gpu(self):
    self.assert_xgboost_schedule(
        gpu_batch_schedule, target="gpu", representation="gpu_array",
        batch_size=32)

  @unittest.skipUnless(
      GPU_AVAILABLE, "requires a GPU-enabled runtime and available GPU")
  def test_compile_and_run_on_gpu_with_atomic_reduction(self):
    self.assert_xgboost_schedule(
        lambda schedule: gpu_tree_reduction_schedule(
            schedule, atomic_gpu_reduction),
        target="gpu", representation="gpu_array", batch_size=32)

  @unittest.skipUnless(
      GPU_AVAILABLE, "requires a GPU-enabled runtime and available GPU")
  def test_compile_and_run_on_gpu_with_vector_reduction(self):
    self.assert_xgboost_schedule(
        lambda schedule: gpu_tree_reduction_schedule(
            schedule, vector_gpu_reduction),
        target="gpu", representation="gpu_array", batch_size=32)

  @unittest.skipUnless(
      GPU_AVAILABLE, "requires a GPU-enabled runtime and available GPU")
  def test_compile_and_run_on_gpu_with_shared_reduction(self):
    self.assert_xgboost_schedule(
        lambda schedule: gpu_tree_reduction_schedule(
            schedule, shared_gpu_reduction),
        target="gpu", representation="gpu_array", batch_size=32)

  @unittest.skipUnless(
      os.environ.get("TREEBEARD_ENABLE_ONNX_TESTS") == "1",
      "requires an ONNX-enabled build and TREEBEARD_ENABLE_ONNX_TESTS=1")
  @unittest.skipIf(onnx is None, "requires the Python onnx package")
  def test_compile_and_run_loaded_onnx_model(self):
    model_path = os.path.join(
        treebeard_repo_dir, "onnx_models", "abalone_onnx_model_save.onnx")
    model = onnx.load(model_path)

    kernel = treebeard.compile(model, batch_size=8, tile_size=1)
    self.addCleanup(kernel.close)

    self.assertEqual(kernel.num_features, 8)
    self.assert_fixture_outputs(kernel, model_path + ".csv")


if __name__ == "__main__":
  unittest.main()