import sys

# Get the input file name as the first argument and the output file name as the second argument
input_file = "/home/ashwin/mlir-build/llvm-project/mlir/examples/treebeard/results/thedeep/gpu/exploration/20240229/exploration_b512-2k_stderr_letters.txt"
output_file = input_file.replace(".txt", "_filtered.txt")

# Open the input file, read it line by line and split the string by ","
with open(input_file, "r") as file:
    lines = file.readlines()
    # Filter only the lines that contain 13 parts when split by ","
    lines = [line for line in lines if len(line.split(",")) == 13]

# Write the resulting lines to the output file
with open(output_file, "w") as file:
    file.writelines(lines)
    
