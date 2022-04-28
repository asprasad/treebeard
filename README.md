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
commit b6b8d34554a4d85ec064463b54a27e073c42beeb (HEAD -> main, origin/main, origin/HEAD)
Author: Peixin-Qiao <qiaopeixin@huawei.com>
Date:   Thu Apr 28 09:40:30 2022 +0800

    [flang] Add lowering stubs for OpenMP/OpenACC declarative constructs
    
    This patch provides the basic infrastructure for lowering declarative
    constructs for OpenMP and OpenACC.
    
    This is part of the upstreaming effort from the fir-dev branch in [1].
    [1] https://github.com/flang-compiler/f18-llvm-project
    
    Reviewed By: kiranchandramohan, shraiysh, clementval
    
    Differential Revision: https://reviews.llvm.org/D124225
```