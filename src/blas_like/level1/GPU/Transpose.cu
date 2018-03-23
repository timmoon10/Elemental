#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/blas_like/level1/GPU/Fill.hpp>

#define BLOCK_DIM 32

namespace El
{
namespace
{
/**
 * \brief Transpose kernel using shared memory.
 *
 * Assumes column-major ordering! This borrows from and expands upon
 * this example code:
 * http://www.nvidia.com/content/cudazone/cuda_sdk/Linear_Algebra.html#transpose
 *
 * \param dest (device) pointer to output matrix memory
 * \param dest_ldim The leading dimension of the output matrix. Must
 *     be at least \c width.
 * \param src (device) pointer to input matrix memory
 * \param height The number of rows in the source matrix
 * \param width The number of columns in the source matrix
 * \param ldim The leading dimension of the source matrix
 */
template <typename T>
__global__ void transpose_kernel(
    T *dest, unsigned dest_ldim,
    T const* src, unsigned height, unsigned width, unsigned ldim)
{
    __shared__ T block[BLOCK_DIM][BLOCK_DIM+1];

    // Copy a block of the matrix into shared memory
    unsigned columnIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned rowIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if ((columnIndex < width) && (rowIndex < height))
    {
        unsigned idx = rowIndex + columnIndex * ldim;
        block[threadIdx.y][threadIdx.x] = src[idx];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    columnIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    rowIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if ((columnIndex < height) && (rowIndex < width))
    {
        unsigned idx = columnIndex*dest_ldim + rowIndex;
        dest[idx] = block[threadIdx.x][threadIdx.y];
    }
}
}// namespace <anon>

template <typename T>
void Transpose_GPU_impl(
    T* dest, unsigned dest_ldim,
    T const* src, unsigned height, unsigned width, unsigned ldim)
{
    dim3 grid{Max<unsigned>(width/BLOCK_DIM,1),
            Max<unsigned>(height/BLOCK_DIM,1), 1};
    dim3 threads{BLOCK_DIM,BLOCK_DIM,1};
    transpose_kernel<<<grid,threads>>>(
        dest, dest_ldim, src, height, width, ldim);
    cudaThreadSynchronize(); // FIXME Sync here or...??
}

template void Transpose_GPU_impl(
    float*, unsigned, float const*, unsigned, unsigned, unsigned);
template void Transpose_GPU_impl(
    double*, unsigned, double const*, unsigned, unsigned, unsigned);
}// namespace El
