# tree-heavy 
A compiler for gradient boosting tree inference.

# To build the tree-heavy project
1. Setup a build of [MLIR](https://mlir.llvm.org/getting_started/)
2. Clone this repository into <path-to-llvm-repo>/llvm-project/mlir/examples/
3. Open a terminal and change directory into <path-to-llvm-repo>/llvm-project/mlir/examples/tree-heavy.
```bash    
    mkdir build && cd build
    bash ../scripts/gen.sh [cmake path] [mlir build directory name] 
    # Eg : bash ../scripts/gen.sh /snap/bin/cmake build (if your mlir build is in a directory called "build")
    bash ../scripts/build.sh [cmake path]
```
4. The "cmake path" and "mlir build directory name" are optional. If cmake path is not specified above, the "cmake" binary in the path is used. The default mlir build directory name is "build".
