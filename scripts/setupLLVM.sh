#!/bin/bash
echo "Cloning LLVM git repo ..."
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
git checkout fc3a260a0fddf2bd2ee18cec53ebe65635ceb2dc
echo "Building LLVM ..."
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="llvm;clang;lld;mlir;openmp" -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
cmake --build .
cmake --build . --target check-mlir
