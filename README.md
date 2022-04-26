# Treebeard 
An optimizing compiler for decision tree ensemble inference.

# To build the Treebeard project
1. Setup a build of [MLIR](https://mlir.llvm.org/getting_started/). See section below for MLIR commit to use.
2. Clone this repository into <path-to-llvm-repo>/llvm-project/mlir/examples/
3. Open a terminal and change directory into <path-to-llvm-repo>/llvm-project/mlir/examples/tree-heavy.
```bash    
    mkdir build && cd build
    bash ../scripts/gen.sh [cmake path] [mlir build directory name] 
    # Eg : bash ../scripts/gen.sh /snap/bin/cmake build (if your mlir build is in a directory called "build")
    cmake --build .
```
4. The "cmake path" and "mlir build directory name" are optional. If cmake path is not specified above, the "cmake" binary in the path is used. The default mlir build directory name is "build".

# MLIR Version
The current version of Treebeard is tested with the following LLVM commit:
```
commit fc3a260a0fddf2bd2ee18cec53ebe65635ceb2dc (HEAD -> main, origin/main, origin/HEAD)
Author: Vitaly Buka <vitalybuka@google.com>
Date:   Tue Dec 7 00:16:28 2021 -0800

    [sanitizer] Don't lock for StackStore::Allocated()
```