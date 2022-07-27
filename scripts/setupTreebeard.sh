#!/bin/bash
echo "Cloning Treebeard git repo ..."
cd llvm-project/mlir/examples/
git clone https://github.com/asprasad/tree-heavy.git
mkdir tree-heavy/build
cd tree-heavy/build
git checkout micro-artifact
bash ../scripts/gen.sh
cmake --build .
