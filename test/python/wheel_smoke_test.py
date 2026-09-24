import json
import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import onnx
import treebeard


def run_smoke_test(xgboost_fixture_dir, onnx_fixture_dir):
  data = np.loadtxt(
    xgboost_fixture_dir / "abalone_xgb_model_save.json.test.sampled.csv",
      delimiter=",")[:13]
  kernel = treebeard.compile(
    xgboost_fixture_dir / "abalone_xgb_model_1.7.6.json",
      batch_size=8,
      tile_size=1)
  np.testing.assert_allclose(
      kernel.run(data[:, :-1]), data[:, -1], rtol=1e-5, atol=1e-5)
  kernel.close()

  model_path = xgboost_fixture_dir / "abalone_xgb_model_1.7.6.json"
  model = json.loads(model_path.read_text(encoding="utf-8"))
  model["learner"]["objective"]["name"] = "rank:pairwise"
  with tempfile.TemporaryDirectory() as directory:
    unsupported_path = pathlib.Path(directory) / "unsupported.json"
    unsupported_path.write_text(json.dumps(model), encoding="utf-8")
    try:
      treebeard.compile(unsupported_path, batch_size=8)
    except treebeard.UnsupportedModelError:
      pass
    else:
      raise AssertionError("unsupported objective did not raise")

  onnx_data = np.loadtxt(
      onnx_fixture_dir / "abalone_onnx_model_save.onnx.csv",
      delimiter=",")[:13]
  kernel = treebeard.compile(
      onnx.load(onnx_fixture_dir / "abalone_onnx_model_save.onnx"),
      batch_size=8,
      tile_size=1)
  np.testing.assert_allclose(
      kernel.run(onnx_data[:, :-1]), onnx_data[:, -1],
      rtol=1e-5, atol=1e-5)
  kernel.close()

  notice = pathlib.Path(treebeard.__file__).with_name(
      "THIRD_PARTY_NOTICES.md")
  if not notice.is_file():
    raise AssertionError("third-party notices are missing from the wheel")


def run_execution_tests():
  os.environ["TREEBEARD_ENABLE_ONNX_TESTS"] = "1"
  import test_high_level_api_execution

  suite = unittest.defaultTestLoader.loadTestsFromModule(
      test_high_level_api_execution)
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  if not result.wasSuccessful():
    raise SystemExit(1)


if __name__ == "__main__":
  xgboost_fixture_dir = pathlib.Path(sys.argv[1]).resolve()
  onnx_fixture_dir = (
      pathlib.Path(sys.argv[2]).resolve()
      if len(sys.argv) > 2 else xgboost_fixture_dir)
  run_smoke_test(
      xgboost_fixture_dir,
      onnx_fixture_dir)
  run_execution_tests()