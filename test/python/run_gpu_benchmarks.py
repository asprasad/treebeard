import os
import sys
import pandas as pd
import numpy
import math
import time
import subprocess

def parse_rapids_cmp_result(result_lines):
    # remove the last line from the result_lines
    result_lines = result_lines[:-1]

    results = []
    for line in result_lines[:-1]:
        cells = line.split(" ")
        benchmark = cells[0]
        batch_size = int(cells[1])
        rapids_total_time = float(cells[2])
        tb_total_time = float(cells[3])
        total_speedup = float(cells[4])
        rapids_kernel_time = float(cells[5])
        tb_kernel_time = float(cells[6])
        tb_kernel_speedup = float(cells[7])
        res = { "benchmark": benchmark, "batch_size": batch_size, 
                "rapids_total_time": rapids_total_time, "tb_total_time": tb_total_time, 
                "total_speedup": total_speedup, "rapids_kernel_time": rapids_kernel_time, 
                "tb_kernel_time": tb_kernel_time, "tb_kernel_speedup": tb_kernel_speedup }
        results.append(res)
    avg_speedups = result_lines[-1].split(" ")
    avg_total_speedup = float(avg_speedups[0])
    avg_kernel_speedup = float(avg_speedups[1])
    return results, avg_total_speedup, avg_kernel_speedup

def run_tahoe_benchmark(benchmark_name, batch_size):
    model_filename = benchmark_name + "_xgb_model_save.json.txt"
    model_filepath = os.path.join(tahoe_benchmarks_dir, model_filename)

    data_filename = model_filename + ".test.sampled.txt"
    data_filepath = os.path.join(tahoe_benchmarks_dir, str(batch_size), data_filename)

    command = [tahoe_exe_path, model_filepath, data_filepath]
    results = subprocess.run(command, capture_output=True, text=True)
    tahoe_results = results.stdout.split("\n")
    lines = [line for line in tahoe_results if line.startswith('*******')]
    assert len(lines) == 1
    total_time = float(lines[0].split(" ")[1])
    return total_time * 1e-6

filepath = os.path.abspath(__file__)
treebeard_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(filepath)))
treebeard_runtime_dir = os.path.join(os.path.join(treebeard_repo_dir, "src"), "python")

rapids_script_path = os.path.join(treebeard_repo_dir, "test", "python", "RAPIDs", "benchmark_rapids.py")
tahoe_dir = "/home/ashwin/ML/Tahoe/"
tahoe_exe_path = os.path.join(tahoe_dir, "Tahoe")
tahoe_benchmarks_dir = os.path.join(tahoe_dir, "Tahoe_expts", "treebeard_models")

stderr_file = os.path.join(treebeard_repo_dir, "test", "python", "stderr.txt")

# delete stderr file if it exists
if os.path.exists(stderr_file):
    os.remove(stderr_file)

# BATCH_SIZES = [1024]
# result_str = """abalone 1024 2.1401665262168363e-07 7.367709518543312e-08 2.904792216401026 1.3802461104378815e-07 4.4079706101190474e-08 3.1312507104048155
# airline 1024 1.7729210889055617e-07 4.796799094904037e-08 3.6960503323748854 1.0664142402155058e-07 1.447172619047619e-08 7.368949814136966
# airline-ohe 1024 5.751978495113907e-07 4.594702650571153e-07 1.2518717602756941 1.5172519765439488e-07 6.015438988095239e-08 2.5222630959214163
# covtype 1024 2.343086747541314e-07 1.0006761710558618e-07 2.3415034906537344 1.547806563654116e-07 4.8272646949404765e-08 3.2063842806805143
# epsilon 1024 1.2773282582028992e-06 9.070381167389098e-07 1.4082409929974058 1.160485026914449e-07 3.318940662202381e-08 3.496552499809909
# higgs 1024 1.80386317272981e-07 5.513540513458706e-08 3.2716965955478696 1.0822334193757603e-07 1.501976376488095e-08 7.205395746011843
# letters 1024 2.0306216486330545e-06 1.8288768894438233e-06 1.110310737892578 1.9674869586846658e-06 1.797064732142857e-06 1.0948336604094353
# year_prediction_msd 1024 1.8247137112276894e-07 9.04764359196027e-08 2.0167833676042775 1.0545942045393444e-07 1.5578962053571427e-08 6.769348310323293
# 2.060310493923643 3.696283608714957
# """

BATCH_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
benchmarks = ["abalone", "airline", "airline-ohe", "covtype", "epsilon", "higgs", "letters", "year_prediction_msd"]

benchmark_results = []
summary_results = []

for batch_size in BATCH_SIZES:
    command = ["python", rapids_script_path, "--batch_size", str(batch_size), "--num_trials", "3"]
    results = subprocess.run(command, capture_output=True, text=True)
    result_str = results.stdout
    rapids_results = result_str.split("\n")
    assert len(rapids_results) == 10
    rapids_results, avg_total_speedup, avg_kernel_speedup = parse_rapids_cmp_result(rapids_results)

    # append stderr to stderr_file
    with open(stderr_file, "a") as f:
       f.write(results.stderr)

    tahoe_speedups = []
    for (i, benchmark) in zip(range(0, len(benchmarks)), benchmarks):
        tahoe_total_time = run_tahoe_benchmark(benchmark, batch_size)
        rapids_results[i]["tahoe_total_time"] = tahoe_total_time
        speedup = tahoe_total_time / rapids_results[i]["tb_kernel_time"]
        rapids_results[i]["tahoe_speedup"] = speedup 
        tahoe_speedups.append(speedup)
    
    # find tahoe geomean speedup
    tahoe_avg_speedup = numpy.prod(tahoe_speedups)**(1.0/len(tahoe_speedups))

    benchmark_results.extend(rapids_results)
    
    summary = {"batch_size": batch_size, "avg_total_speedup": avg_total_speedup, 
                "avg_kernel_speedup": avg_kernel_speedup, "tahoe_avg_speedup": tahoe_avg_speedup}
    summary_results.append(summary)

    print(rapids_results)
    print(summary)

results_df = pd.DataFrame(benchmark_results)
summary_df = pd.DataFrame(summary_results)

print(results_df.to_string())
print(summary_df.to_string())