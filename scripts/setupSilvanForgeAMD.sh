# Clone and build LLVM
git clone https://github.com/asprasad/llvm-project.git
cd llvm-project/
git checkout release/16.x
mkdir build
cd build
if command -v ninja > /dev/null 2>&1; then
    cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="llvm;clang;lld;mlir;openmp" -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON -DMLIR_ENABLE_ROCM_RUNNER=ON -DMLIR_INCLUDE_INTEGRATION_TESTS=ON
    cmake --build .
else
    cmake ../llvm -DLLVM_ENABLE_PROJECTS="llvm;clang;lld;mlir;openmp" -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON -DMLIR_ENABLE_ROCM_RUNNER=ON -DMLIR_INCLUDE_INTEGRATION_TESTS=ON
    cmake --build . -j16
fi
# Clone and build Treebeard
cd ../mlir/examples
git clone https://github.com/asprasad/treebeard.git
cd treebeard

# Checkout branch with changes made for silvanforge
git checkout silvanforge

mkdir build
cd build
bash ../scripts/gen_gpu_amd.sh
if command -v ninja > /dev/null 2>&1; then
    cmake --build .
else
    cmake --build . -j16
fi


