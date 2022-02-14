import sys
import os
import matplotlib.pyplot as plt
import math
import numpy

stats_file = sys.argv[1]
print("Analyzing file ", stats_file)

tree_stats = dict()

with open(stats_file, 'r') as f:
  first_line = f.readline()
  numbers = first_line.split(",")
  num_trees = int(numbers[0])
  num_inputs = int(numbers[1])

  for i in range(num_trees):
    line = f.readline()
    numbers = line.split(",")
    hit_counts = [int(numbers[i]) for i in range(0, len(numbers), 2)]
    depths = [int(numbers[i]) for i in range(1, len(numbers), 2)]
    tree_stats[i] = (hit_counts, depths)

# For a given percentage of inputs (say p%), plot fraction of leaves on x-axis vs how many
# trees would cover p% of inputs with x% of leaves (i.e at x=0.5, how many trees in the model
# would be able to cover the hits of at least p% of input rows using just half their leaves)

def CanCover(tree_num : int, input_fraction : float, leaf_fraction : float) -> bool :
  hit_counts = tree_stats[tree_num][0]
  num_leaves = int(math.ceil(len(hit_counts) * leaf_fraction))
  num_hits = int(math.ceil(num_inputs * input_fraction))
  actual_hits = 0
  for i in range(num_leaves):
    actual_hits += hit_counts[i]
  return actual_hits >= num_hits

def ComputeSinglePoint(input_fraction : float, leaf_fraction : float) -> float :
  num_valid_trees = 0
  for i in range(num_trees):
    if CanCover(i, input_fraction, leaf_fraction):
      num_valid_trees += 1
  return float(num_valid_trees)/float(num_trees)

def ComputeSingleCurve(input_fraction : float) -> list[float]:
  leaf_fractions = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  tree_fractions = []
  for leaf_fraction in leaf_fractions:
    tree_fractions.append(ComputeSinglePoint(input_fraction, leaf_fraction))
  return tree_fractions, leaf_fractions

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(numpy.arange(0, 1, 0.1), rotation='vertical')
ax.set_yticks(numpy.arange(0, 1., 0.05))

input_fractions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for input_fraction in input_fractions:
  tree_fractions, leaf_fractions = ComputeSingleCurve(input_fraction)
  plt.plot(leaf_fractions, tree_fractions, "-o", label="Input Fraction " + str(input_fraction) )

plt.xlabel("Fraction of Leaves Used")
plt.ylabel("Fraction of Trees")
plt.title(os.path.basename(stats_file))
plt.legend()
plt.grid(linestyle="dotted")
plt.savefig(stats_file + ".png")