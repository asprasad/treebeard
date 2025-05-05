# Clone and build LLVM
git clone https://github.com/asprasad/llvm-project.git
cd llvm-project/
git checkout release/16.x
mkdir build
cd build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="llvm;clang;lld;mlir;openmp" -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON -DMLIR_ENABLE_CUDA_RUNNER=ON -DMLIR_INCLUDE_INTEGRATION_TESTS=ON
cmake --build .

# Clone and build Treebeard
cd ../mlir/examples
git clone https://github.com/asprasad/treebeard.git
cd treebeard

# Checkout branch with changes made for silvanforge
git checkout ashwin/onnx-documentation

mkdir build
cd build
bash ../scripts/gen.sh -o
cmake --build .


