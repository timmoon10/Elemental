#include "El-lite.hpp"
#include "El/core/imports/cublas.hpp"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace El
{

namespace cublas
{

namespace
{

class cuBLAS_Manager
{
    cublasHandle_t cublas_handle_;
public:
    operator cublasHandle_t() { return cublas_handle_; }

    cuBLAS_Manager()
    {
        if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS)
            RuntimeError("cublasCreate() Error!");
    }

    ~cuBLAS_Manager()
    {
        if (cublasDestroy(cublas_handle_) != CUBLAS_STATUS_SUCCESS)
            std::terminate();
    }
};// class cuBLAS_Manager

inline cublasOperation_t CharTocuBLASOp(char c)
{
    switch (c)
    {
    case 'N':
        return CUBLAS_OP_N;
        break;
    case 'T':
        return CUBLAS_OP_T;
        break;
    case 'C':
        return CUBLAS_OP_C;
        break;
    default:
        RuntimeError("cuBLAS: Unknown operation type.");
        return CUBLAS_OP_N; // Compiler yells about not returning anything...
    }
}

}// namespace <anon>

  //cuBLAS_Manager manager_;

#define ADD_GEMM_IMPL(ScalarType, TypeChar)                             \
    void Gemm(                                                          \
        char transA, char transB, int m, int n, int k,                  \
        ScalarType const& alpha,                                        \
        ScalarType const* A, int ALDim,                                 \
        ScalarType const* B, int BLDim,                                 \
        ScalarType const& beta,                                         \
        ScalarType* C, int CLDim )                                      \
    {                                                                   \
      cuBLAS_Manager manager_; \
        auto ret = cublas ## TypeChar ## gemm(                          \
            manager_,                                                   \
            CharTocuBLASOp(transA), CharTocuBLASOp(transB),             \
            m, n, k, &alpha, A, ALDim, B, BLDim, &beta, C, CLDim);      \
        if (ret != CUBLAS_STATUS_SUCCESS)                               \
            RuntimeError("cuBLAS::Gemm failed!");                       \
        cudaThreadSynchronize();/* FIXME */                             \
    }
ADD_GEMM_IMPL(float, S)
ADD_GEMM_IMPL(double, D)

}// namespace cublas

void InitializeCUDA(int,char*[])
{
    int device_count;
    auto error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess)
        RuntimeError("CUDA initialize error: ", cudaGetErrorString(error));

    if (device_count < 1)
        RuntimeError("No CUDA devices found!");

    int device_id = 0;

    cudaDeviceProp deviceProp;
    error = cudaGetDeviceProperties(&deviceProp, device_id);

    if (error != cudaSuccess)
        RuntimeError("CUDA initialize error: ", cudaGetErrorString(error));

    if (deviceProp.computeMode == cudaComputeModeProhibited)
        RuntimeError("Device 0 is in ComputeModeProhibited mode. Can't use.");

    cudaSetDevice(0);
}

}// namespace El
