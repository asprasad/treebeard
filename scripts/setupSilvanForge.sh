# Clone and build LLVM
git clone https://github.com/asprasad/llvm-project.git
cd llvm-project/
git checkout release/16.x

if [ "$BUILD_MODE" = "debug" ]; then
    BUILD_DIR="debug_build"
    CONFIG="Debug"
else
    BUILD_DIR="build"
    CONFIG="Release"
fi

mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="llvm;clang;lld;mlir;openmp" -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" -DCMAKE_BUILD_TYPE=$CONFIG -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON -DMLIR_ENABLE_CUDA_RUNNER=ON -DMLIR_INCLUDE_INTEGRATION_TESTS=ON
cmake --build .

# Clone and build Treebeard
cd ../mlir/examples

if [ ! -d "treebeard" ]; then
    git clone https://github.com/asprasad/treebeard.git
else
    echo "'treebeard' directory already exists. Skipping clone."
fi

cd treebeard

# Checkout branch with changes made for silvanforge
git checkout silvanforge

mkdir -p $BUILD_DIR
cd $BUILD_DIR
bash ../scripts/gen_gpu.sh
cmake --build .

# Clone and build Tahoe
cd ../../

if [ ! -d "Tahoe" ]; then
    git clone https://github.com/sampathrg/Tahoe.git
else
    echo "'Tahoe' directory already exists. Skipping clone."
fi

cd Tahoe
git checkout tahoe-expts
make
