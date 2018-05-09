#ifndef HYDROGEN_IMPORTS_CUBLAS_HPP_
#define HYDROGEN_IMPORTS_CUBLAS_HPP_

#include "cuda.hpp"
#include <cublas_v2.h>

namespace El
{

/** \class CublasError
 *  \brief Exception class for cuBLAS errors.
 */
struct CublasError : std::runtime_error
{

    static std::string get_error_string_(cublasStatus_t status)
    {
        switch (status)
        {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "unknown cuBLAS error";
        }
    }

    std::string build_error_string_(
        cublasStatus_t status, char const* file, int line)
    {
        std::ostringstream oss;
        oss << "cuBLAS error (" << file << ":" << line << "): "
            << get_error_string_(status);
        return oss.str();
    }

    CublasError(cublasStatus_t status, char const* file, int line)
        : std::runtime_error{build_error_string_(status,file,line)}
    {}

};// struct cublasError

#define EL_FORCE_CHECK_CUBLAS(cublas_call)                              \
    do                                                                  \
    {                                                                   \
        /* Check for earlier asynchronous errors. */                    \
        EL_FORCE_CHECK_CUDA(cudaSuccess);                               \
        {                                                               \
            /* Make cuBLAS call and check for errors. */                \
            const cublasStatus_t status_CHECK_CUBLAS = (cublas_call);   \
            if (status_CHECK_CUBLAS != CUBLAS_STATUS_SUCCESS)           \
            {                                                           \
              cudaDeviceReset();                                        \
              throw CublasError(status_CHECK_CUBLAS,__FILE__,__LINE__); \
            }                                                           \
        }                                                               \
        {                                                               \
            /* Check for CUDA errors. */                                \
            cudaError_t status_CHECK_CUBLAS = cudaDeviceSynchronize();  \
            if (status_CHECK_CUBLAS == cudaSuccess)                     \
                status_CHECK_CUBLAS = cudaGetLastError();               \
            if (status_CHECK_CUBLAS != cudaSuccess)                     \
            {                                                           \
                cudaDeviceReset();                                      \
                throw CudaError(status_CHECK_CUBLAS,__FILE__,__LINE__,false); \
            }                                                           \
        }                                                               \
    } while (0)

#ifdef EL_RELEASE
#define EL_CHECK_CUBLAS(cublas_call) (cublas_call)
#else
#define EL_CHECK_CUBLAS(cublas_call) EL_FORCE_CHECK_CUBLAS(cublas_call)
#endif // #ifdef EL_RELEASE


namespace cublas
{

//
// BLAS 1 Routines
//

#define ADD_AXPY_DECL(ScalarType)                       \
    void Axpy(int n, ScalarType const& alpha,           \
              ScalarType const* X, int incx,            \
              ScalarType* Y, int incy);

#define ADD_COPY_DECL(ScalarType)                       \
    void Copy(int n, ScalarType const* X, int incx,     \
              ScalarType* Y, int incy);

//
// BLAS 2 Routines
//

#define ADD_GEMV_DECL(ScalarType)                                       \
    void Gemv(                                                          \
        char transA, BlasInt m, BlasInt n,                              \
        ScalarType const& alpha,                                        \
        ScalarType const* A, BlasInt ALDim,                             \
        ScalarType const* x, BlasInt xLDim,                             \
        ScalarType const& beta,                                         \
        ScalarType* y, BlasInt yLDim);

//
// BLAS 3 Routines
//

#define ADD_GEMM_DECL(ScalarType)                                       \
    void Gemm(                                                          \
        char transA, char transB, BlasInt m, BlasInt n, BlasInt k,      \
        ScalarType const& alpha,                                        \
        ScalarType const* A, BlasInt ALDim,                             \
        ScalarType const* B, BlasInt BLDim,                             \
        ScalarType const& beta,                                         \
        ScalarType* C, BlasInt CLDim);

//
// BLAS-like Extension Routines
//
#define ADD_GEAM_DECL(ScalarType)                 \
    void Geam(char transA, char transB,           \
              BlasInt m, BlasInt n,               \
              ScalarType const& alpha,            \
              ScalarType const* A, BlasInt ALDim, \
              ScalarType const& beta,             \
              ScalarType const* B, BlasInt BLDim, \
              ScalarType* C, BlasInt CLDim);

// BLAS 1
ADD_AXPY_DECL(float)
ADD_AXPY_DECL(double)
ADD_COPY_DECL(float)
ADD_COPY_DECL(double)

// BLAS 2
ADD_GEMV_DECL(float)
ADD_GEMV_DECL(double)

// BLAS 3
ADD_GEMM_DECL(float)
ADD_GEMM_DECL(double)

// BLAS-like Extension
ADD_GEAM_DECL(float)
ADD_GEAM_DECL(double)

}// namespace cublas
}// namespace El
#endif
