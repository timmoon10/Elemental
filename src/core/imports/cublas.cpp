#include "El-lite.hpp"
#include "El/core/imports/cuda.hpp"
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
    operator cublasHandle_t()
    {
        if (!initialized_)
            initialize();
        return cublas_handle_;
    }

    cuBLAS_Manager() = default;

    void initialize()
    {
        if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS)
            RuntimeError("cublasCreate() Error!");
        initialized_ = true;
    }

    ~cuBLAS_Manager()
    {
        if (initialized_)
            if (cublasDestroy(cublas_handle_) != CUBLAS_STATUS_SUCCESS)
                std::terminate();
    }

    bool initialized_ = false;
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

cuBLAS_Manager manager_;

//
// BLAS 1
//
#define ADD_AXPY_IMPL(ScalarType, TypeChar)             \
    void Axpy(int n, ScalarType const& alpha,           \
              ScalarType const* X, int incx,            \
              ScalarType* Y, int incy)                  \
    {                                                   \
        EL_CHECK_CUBLAS(cublas ## TypeChar ## axpy(     \
            manager_, n, &alpha, X, incx, Y, incy));    \
    }

#define ADD_COPY_IMPL(ScalarType, TypeChar)             \
    void Copy(int n, ScalarType const* X, int incx,     \
              ScalarType* Y, int incy)                  \
    {                                                   \
      EL_CHECK_CUBLAS(cublas ## TypeChar ## copy(       \
            manager_, n, X, incx, Y, incy));            \
    }

//
// BLAS 2
//
#define ADD_GEMV_IMPL(ScalarType, TypeChar)                             \
    void Gemv(                                                          \
        char transA, int m, int n,                                      \
        ScalarType const& alpha,                                        \
        ScalarType const* A, int ALDim,                                 \
        ScalarType const* B, int BLDim,                                 \
        ScalarType const& beta,                                         \
        ScalarType* C, int CLDim )                                      \
    {                                                                   \
      EL_CHECK_CUBLAS(cublas ## TypeChar ## gemv(                       \
            manager_,                                                   \
            CharTocuBLASOp(transA),                                     \
            m, n, &alpha, A, ALDim, B, BLDim, &beta, C, CLDim));        \
    }

//
// BLAS 3
//
#define ADD_GEMM_IMPL(ScalarType, TypeChar)                             \
    void Gemm(                                                          \
        char transA, char transB, int m, int n, int k,                  \
        ScalarType const& alpha,                                        \
        ScalarType const* A, int ALDim,                                 \
        ScalarType const* B, int BLDim,                                 \
        ScalarType const& beta,                                         \
        ScalarType* C, int CLDim )                                      \
    {                                                                   \
        EL_CHECK_CUBLAS(cublas ## TypeChar ## gemm(                     \
            manager_,                                                   \
            CharTocuBLASOp(transA), CharTocuBLASOp(transB),             \
            m, n, k, &alpha, A, ALDim, B, BLDim, &beta, C, CLDim));     \
    }

//
// BLAS-like Extension
//
#define ADD_GEAM_IMPL(ScalarType, TypeChar)                             \
    void Geam(                                                          \
        char transA, char transB,                                       \
        int m, int n,                                                   \
        ScalarType const& alpha,                                        \
        ScalarType const* A, int ALDim,                                 \
        ScalarType const& beta,                                         \
        ScalarType const* B, int BLDim,                                 \
        ScalarType* C, int CLDim )                                      \
    {                                                                   \
      EL_CHECK_CUDA(cublas ## TypeChar ## geam(                         \
            manager_,                                                   \
            CharTocuBLASOp(transA), CharTocuBLASOp(transB),             \
            m, n, &alpha, A, ALDim, &beta, B, BLDim, C, CLDim));        \
    }

// BLAS 1
ADD_AXPY_IMPL(float, S)
ADD_AXPY_IMPL(double, D)

ADD_COPY_IMPL(float, S)
ADD_COPY_IMPL(double, D)

// BLAS 2
ADD_GEMV_IMPL(float, S)
ADD_GEMV_IMPL(double, D)

// BLAS 3
ADD_GEMM_IMPL(float, S)
ADD_GEMM_IMPL(double, D)

// BLAS-like extension
ADD_GEAM_IMPL(float, S)
ADD_GEAM_IMPL(double, D)

}// namespace cublas

void InitializeCUDA(int,char*[], int requested_device_id)
{
    int device_count;
    auto error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess)
        RuntimeError("CUDA initialize error: ", cudaGetErrorString(error));

    if (device_count < 1)
        RuntimeError("No CUDA devices found!");

    int device_id = requested_device_id;
    if(requested_device_id < 0) {
      device_id = 0;
    }

    cudaDeviceProp deviceProp;
    error = cudaGetDeviceProperties(&deviceProp, device_id);

    if (error != cudaSuccess)
        RuntimeError("CUDA initialize error: ", cudaGetErrorString(error));

    if (deviceProp.computeMode == cudaComputeModeProhibited)
        RuntimeError(std::string {} + "Device " + std::to_string(device_id)
                     + " is in ComputeModeProhibited mode. Can't use.");

    cudaSetDevice(device_id);
}

}// namespace El
