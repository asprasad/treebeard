import os
import json

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filepath))))

def get_benchmark_names(dir_name):
  # get all the json files in the directory dir_name
  files = os.listdir(dir_name)
  # filter out files that end with treebeard-globals.json
  files = [f for f in files if not f.endswith("treebeard-globals.json")]
  json_files = [os.path.join(dir_name, f) for f in files if f.endswith(".json")]
  return json_files

if __name__ == "__main__":
  benchmark_dir = os.path.join(treebeard_repo_dir, "xgb_models/test/GPUBenchmarks")
  benchmarks = get_benchmark_names(benchmark_dir)
  batch_size = 512
  num_trials = 5

  executor_script_path = os.path.join(treebeard_repo_dir, "test/python/RAPIDs/rapids_compare_random_models.py")
  # iterate the benchmarks 3 at a time
  for i in range(0, len(benchmarks), 3):
    # get the 3 benchmarks
    benchmark_map = { "benchmarks": benchmarks[i:i+3] }
    benchmark_json = json.dumps(benchmark_map)
    # run the compare script
    cmd = f"python -u {executor_script_path} --benchmarks '{benchmark_json}' --batch_size {batch_size} --num_trials {num_trials}"
    # print(cmd, flush=True)
    os.system(cmd)
    # break
