import os

filename = '/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/results/xgboost/holmes/Native/20220217/RawResults_Float_Sparse_OneTreeSched_TestIP.txt'
# doubleCSV = '/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/results/xgboost/holmes/Native/Reordering/DoubleResults_O0_LargeTiles.csv'
floatCSV = '/home/ashwin/mlir-build/llvm-project/mlir/examples/tree-heavy/results/xgboost/holmes/Native/20220217/FloatResults_Sparse_OneTreeSched_testIP.csv'

results = dict()
results["double"] = dict()
results["float"] = dict()

raw_results = open(filename, "r")

tile_size = 1
batch_size = 1
type_string = ""
benchmark_names = []

for line in raw_results:
  line = line.strip()
  tokens = line.split()
  print (tokens)
  if tokens[0] == "Type":
    type_string = tokens[1]
    batch_size = int(tokens[3])
    tile_size = int(tokens[5])
    if not batch_size in results[type_string].keys():
      results[type_string][batch_size] = dict()
    if not tile_size in results[type_string][batch_size].keys():
      results[type_string][batch_size][tile_size] = dict()
    benchmark_names = []
  else:
    results[type_string][batch_size][tile_size][tokens[0]] = int(tokens[1])
    benchmark_names.append(tokens[0])

print (results)

def WriteResultsToCSV(csv_name, results, typeStr):
  csv_file = open(csv_name, "w")

  header = "Tile Size, Batch Size"
  for benchmark in benchmark_names:
    header = header + ", " + benchmark

  csv_file.write(header + "\n")

  doubleResults = results[typeStr]
  for batchSize, batchSizeResult in doubleResults.items():
    for tileSize, tileSizeResult in batchSizeResult.items():
      rowStr = str(tileSize) + ", " + str(batchSize)
      for benchmark in benchmark_names:
        rowStr = rowStr + ", " + str(tileSizeResult[benchmark])
      csv_file.write(rowStr + "\n")

# WriteResultsToCSV(doubleCSV, results, "double")
WriteResultsToCSV(floatCSV, results, "float")