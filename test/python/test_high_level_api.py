import importlib.util
import os
import sys
import types
import unittest
from unittest import mock
from enum import IntEnum

import numpy


def load_treebeard_module():
  runtime_module = types.ModuleType("treebeard_runtime_api")

  class TreebeardErrorCode(IntEnum):
    SUCCESS = 0
    UNSUPPORTED_MODEL = 1
    UNSUPPORTED_CONFIGURATION = 2
    COMPILATION = 3
    RUNTIME = 4

  class FakeTreebeardAPI:
    def __init__(self):
      self.runtime_lib = object()

  runtime_module.TreebeardAPI = FakeTreebeardAPI
  runtime_module.TreebeardErrorCode = TreebeardErrorCode
  old_runtime_module = sys.modules.get("treebeard_runtime_api")
  sys.modules["treebeard_runtime_api"] = runtime_module
  try:
    treebeard_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "python", "treebeard.py")
    spec = importlib.util.spec_from_file_location(
        "treebeard_high_level_api_test_module", treebeard_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
  finally:
    if old_runtime_module is None:
      del sys.modules["treebeard_runtime_api"]
    else:
      sys.modules["treebeard_runtime_api"] = old_runtime_module


treebeard = load_treebeard_module()


class FakeRunner:
  def __init__(self, batch_size=4, row_size=2):
    self.batchSize = batch_size
    self.rowSize = row_size
    self.inputs = None

  def RunInferenceOnMultipleBatches(self, inputs, result_type):
    self.inputs = inputs
    return numpy.arange(inputs.shape[0], dtype=result_type)


class HighLevelAPITests(unittest.TestCase):
  def test_native_errors_map_to_public_exception_types(self):
    expected_types = {
      treebeard.TreebeardErrorCode.UNSUPPORTED_MODEL:
        treebeard.UnsupportedModelError,
      treebeard.TreebeardErrorCode.UNSUPPORTED_CONFIGURATION:
        treebeard.UnsupportedConfigurationError,
      treebeard.TreebeardErrorCode.COMPILATION: treebeard.CompilationError,
      treebeard.TreebeardErrorCode.RUNTIME:
        treebeard.RuntimeUnavailableError,
        99: treebeard.TreebeardError,
    }

    for code, expected_type in expected_types.items():
      with self.subTest(code=code):
        native_error = RuntimeError("native failure")
        native_error.code = code
        translated = treebeard._translate_native_error(native_error)
        self.assertIsInstance(translated, expected_type)
        self.assertEqual(str(translated), "native failure")

  def test_shared_reduce_forwards_to_runtime_api(self):
    schedule = treebeard.Schedule(123)
    index = treebeard.IndexVariable("tree_thread", 456)

    with mock.patch.object(
      treebeard.treebeardAPI, "Schedule_SharedReduce",
      create=True) as shared_reduce:
      schedule.SharedReduce(index)

    shared_reduce.assert_called_once_with(123, 456)

  def test_kernel_pads_partial_batch_and_trims_results(self):
    runner = FakeRunner()
    kernel = treebeard.TreebeardKernel(
        runner, object(), mock.Mock(), numpy.float32, numpy.float32)
    self.addCleanup(kernel.close)

    results = kernel.run(numpy.ones((5, 2), dtype=numpy.float64))

    numpy.testing.assert_array_equal(results, numpy.arange(5, dtype=numpy.float32))
    self.assertEqual(runner.inputs.shape, (8, 2))
    self.assertEqual(runner.inputs.dtype, numpy.float32)
    self.assertTrue(runner.inputs.flags.c_contiguous)

  def test_kernel_rejects_wrong_feature_count(self):
    kernel = treebeard.TreebeardKernel(
        FakeRunner(), object(), mock.Mock(), numpy.float32, numpy.float32)
    self.addCleanup(kernel.close)

    with self.assertRaisesRegex(ValueError, "expected 2 features, got 3"):
      kernel.run(numpy.ones((4, 3)))

  def test_materialize_onnx_model_infers_static_feature_count(self):
    dimension = types.SimpleNamespace(dim_value=7)
    shape = types.SimpleNamespace(dim=[dimension])
    tensor_type = types.SimpleNamespace(shape=shape)
    input_type = types.SimpleNamespace(tensor_type=tensor_type)
    model_input = types.SimpleNamespace(type=input_type)

    class FakeONNXModel:
      graph = types.SimpleNamespace(input=[model_input])

      def SerializeToString(self):
        return b"onnx-model"

    with treebeard.tempfile.TemporaryDirectory() as directory:
      artifacts = types.SimpleNamespace(name=directory)
      model_path, filetype, num_features = treebeard._materialize_model(
          FakeONNXModel(), artifacts)

      self.assertEqual(filetype, "onnx_file")
      self.assertEqual(num_features, 7)
      with open(model_path, "rb") as model_file:
        self.assertEqual(model_file.read(), b"onnx-model")

  def test_compile_materializes_xgboost_model_and_applies_schedule(self):
    events = []
    runner = FakeRunner()

    class FakeModel:
      def get_booster(self):
        events.append("get_booster")
        return self

      def save_model(self, path):
        events.append(("save_model", path))
        with open(path, "w", encoding="ascii") as model_file:
          model_file.write("{}")

    class FakeContext:
      def __init__(self, model_path, globals_path, options,
                   gpu_auto_schedule_options):
        events.append(("context", model_path, globals_path))

      def SetInputFiletype(self, filetype):
        events.append(("filetype", filetype))

      def SetRepresentationType(self, representation):
        events.append(("representation", representation))

      def BuildHIRRepresentation(self):
        events.append("build")

      def GetSchedule(self):
        events.append("get_schedule")
        return "schedule"

      def ConstructInferenceRunnerFromHIR(self):
        events.append("compile_cpu")
        return runner

    def apply_schedule(schedule):
      events.append(("schedule", schedule))

    with mock.patch.object(treebeard, "TreebeardContext", FakeContext), \
         mock.patch.object(treebeard, "CompilerOptions", mock.Mock):
      kernel = treebeard.compile(FakeModel(), schedule=apply_schedule)

    self.assertIs(kernel._inference_runner, runner)
    self.assertIn(("filetype", "xgboost_json"), events)
    self.assertLess(events.index("build"), events.index(("schedule", "schedule")))
    self.assertLess(events.index(("schedule", "schedule")),
                    events.index("compile_cpu"))
    self.assertTrue(os.path.exists(events[1][1]))
    kernel.close()


if __name__ == "__main__":
  unittest.main()