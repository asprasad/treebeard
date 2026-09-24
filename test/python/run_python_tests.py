import cpu_tests
import gpu_tests
import unittest
from test_high_level_api_execution import GPU_AVAILABLE

cpu_tests.run_all_tests()
if GPU_AVAILABLE:
	gpu_tests.run_all_tests()
else:
	print("Skipping GPU tests: no GPU-enabled runtime and available GPU")

high_level_api_tests = unittest.defaultTestLoader.loadTestsFromNames([
	"test_high_level_api",
	"test_high_level_api_execution",
])
result = unittest.TextTestRunner(verbosity=2).run(high_level_api_tests)
if not result.wasSuccessful():
	raise SystemExit(1)