#!/bin/bash
echo "Cloning Treebeard git repo ..."
cd llvm-project/mlir/examples/
git clone https://github.com/asprasad/treebeard.git
mkdir treebeard/build
cd treebeard/build
bash ../scripts/gen.sh
cmake --build .
