import os
import sys
import pandas as pd
import numpy as np
import math
import time
import subprocess

def parse_hb_cmp_result(result_lines):
    # remove the last line from the result_lines
    result_lines = result_lines[:-1]

    results = []
    for line in result_lines[:-1]:
        cells = line.split(" ")
        benchmark = cells[0]
        batch_size = int(cells[1])
        hb_jit_time = float(cells[2])
        hb_torch_time = float(cells[3])
        hb_tvm_time = float(cells[4])
        tb_total_time = float(cells[5])
        jit_speedup = float(cells[6])
        torch_speedup = float(cells[7])
        tvm_speedup = float(cells[8])
        res = { "benchmark": benchmark, "batch_size": batch_size, 
                "hb_jit_time": hb_jit_time, "hb_torch_time": hb_torch_time, "hb_tvm_time": hb_tvm_time,
                "tb_total_time": tb_total_time, "hb_jit_speedup": jit_speedup, 
                "hb_torch_speedup": torch_speedup, "hb_tvm_speedup": tvm_speedup }
        results.append(res)
    avg_speedups = result_lines[-1].split(" ")
    avg_jit_speedup = float(avg_speedups[0])
    avg_torch_speedup = float(avg_speedups[1])
    avg_tvm_speedup = float(avg_speedups[2])
    return results, avg_jit_speedup, avg_torch_speedup, avg_tvm_speedup

if __name__ == "__main__":
    filepath = os.path.abspath(__file__)
    treebeard_test_dir = os.path.dirname(os.path.dirname(os.path.dirname(filepath)))

    hb_script_path = os.path.join(treebeard_test_dir, "python", "hummingbird", "hb_test.py")
    stderr_file = os.path.join(treebeard_test_dir, "python", "hummingbird", "stderr.txt")

    # delete stderr file if it exists
    if os.path.exists(stderr_file):
        os.remove(stderr_file)

    BATCH_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
    # benchmarks = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs", "letters", "year_prediction_msd"]

    benchmark_results = []
    summary_results = []

    for batch_size in BATCH_SIZES:
        command = ["python", hb_script_path, "--batch_size", str(batch_size), "--num_trials", "3"]
        results = subprocess.run(command, capture_output=True, text=True)
        # print(results.stdout, results.stderr)
        result_str = results.stdout
        results_arr = result_str.split("\n")
        assert len(results_arr) == 10
        hb_results, avg_jit_speedup, avg_torch_speedup, avg_tvm_speedup = parse_hb_cmp_result(results_arr)

        # append stderr to stderr_file
        with open(stderr_file, "a") as f:
           f.write(results.stderr)
   
        benchmark_results.extend(hb_results)
        
        summary = {"batch_size": batch_size, "avg_jit_speedup": avg_jit_speedup, "avg_torch_speedup": avg_torch_speedup, "avg_tvm_speedup": avg_tvm_speedup}
        summary_results.append(summary)

        print(hb_results)
        print(summary)

    results_df = pd.DataFrame(benchmark_results)
    summary_df = pd.DataFrame(summary_results)

    print(results_df.to_string())
    print(summary_df.to_string())

    # import test_inputs
