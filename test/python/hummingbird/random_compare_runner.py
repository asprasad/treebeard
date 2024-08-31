import argparse
import os
import json

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filepath))))

def get_tree_depth(model_path):
    return int(model_path.split("/")[-1].split("_")[4])

def get_num_features(model_path):
    return int(model_path.split("/")[-1].split("_")[3])

def get_benchmark_names(dir_name, batch_size):
  # get all the json files in the directory dir_name
  files = os.listdir(dir_name)
  # filter out files that end with treebeard-globals.json
  files = [f for f in files if not f.endswith("treebeard-globals.json")]
  json_files = [os.path.join(dir_name, f) for f in files if f.endswith(".json")]
  depth = 8 if batch_size == 512 else 6
  json_files = [f for f in json_files if get_tree_depth(f) == depth]
  return json_files

if __name__ == "__main__":
  benchmark_dir = os.path.join(treebeard_repo_dir, "xgb_models/test/GPUBenchmarks")

  arg_parser = argparse.ArgumentParser(description="")
  arg_parser.add_argument("-b", "--batch_size", help="Batch size", type=int)
  arg_parser.add_argument("-n", "--num_trials", help="Number of trials to run", type=int)

  # Parse the command line arguments
  args = arg_parser.parse_args()

  batch_size = args.batch_size
  num_trials = args.num_trials

  benchmarks = get_benchmark_names(benchmark_dir, batch_size)
  executor_script_path = os.path.join(treebeard_repo_dir, "test/python/hummingbird/hb_compare_random_models.py")
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
