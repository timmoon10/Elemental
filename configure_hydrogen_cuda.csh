#!/bin/csh

set path = ( /usr/workspace/wsb/brain/utils/toss2/cmake-3.9.6/bin  $path )

# Sets up the stuff LC is nice enough to provide
use gcc-4.9.3p
#module load cudatoolkit/8.0

if (`hostname | grep surface -c` == 1) then
	set NCCL_DIR="/usr/workspace/wsb/brain/nccl2/nccl_2.1.15-1+cuda9.1_x86_64/"
        setenv CMAKE_PREFIX_PATH /usr/global/tools/mpi/sideinstalls/$SYS_TYPE/mvapich2-2.3/install-gcc-4.9.3-cuda-9.1
else if (`hostname | grep pascal -c` == 1) then
	set NCCL_DIR="/usr/workspace/wsb/brain/nccl2/nccl_2.1.15-1+cuda9.1_x86_64/"
else if (`hostname | grep ray -c` == 1) then
	set NCCL_DIR="/usr/workspace/wsb/brain/nccl2/nccl_2.0.5-3+cuda8.0_ppc64el/"
endif
#set ALUMINUM_DIR="/p/lscratche/ayoo/al/Aluminum/install/lib64/cmake/aluminum"


#
#
# Adds modern CMake, ninja, and CUDA-aware MPI to the path
#set path = ( /usr/workspace/wsb/brain/utils/toss2/cmake-3.9.6/bin /usr/workspace/wsb/brain/utils/toss2/ninja/bin /usr/global/tools/mpi/sideinstalls/chaos_5_x86_64_ib/mvapich2-2.2/install-gcc-cuda/bin $path )

# Adds the CUDA-aware MPI library to the LD_LIBRARY path to ensure preference
#setenv LD_LIBRARY_PATH /usr/global/tools/mpi/sideinstalls/chaos_5_x86_64_ib/mvapich2-2.2/install-gcc-cuda/lib:${ALUMINUM_DIR}/lib:${NCCL_DIR}/lib:${LD_LIBRARY_PATH}

# Trick CMake into picking the right MPI
#setenv CMAKE_PREFIX_PATH /usr/global/tools/mpi/sideinstalls/chaos_5_x86_64_ib/mvapich2-2.2/install-gcc-cuda

cmake -DCMAKE_C_COMPILER=`which gcc` \
    -DCMAKE_CXX_COMPILER=`which g++` \
    -DBUILD_SHARED_LIBS=YES \
    -DCMAKE_INSTALL_PREFIX=test_install_hydrogen \
    -DHydrogen_GENERAL_LAPACK_FALLBACK=ON \
    -DHydrogen_ENABLE_CUDA=ON \
    -DHydrogen_ENABLE_CUB=OFF  \
    -DCMAKE_CXX_FLAGS="-g -O0" \
    -DCMAKE_CUDA_FLAGS="-g -O0" \
    -DUSE_ALUMINUM=ON \
    -DUSE_NCCL=ON \
    -DAluminum_DIR=/p/lscratchf/ayoo/al/Aluminum/install/lib64/cmake/aluminum \
    -DNCCL_DIR=/usr/workspace/wsb/brain/nccl2/nccl_2.1.15-1+cuda9.1_x86_64 \
    ..
