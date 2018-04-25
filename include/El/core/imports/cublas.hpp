#ifndef HYDROGEN_IMPORTS_CUBLAS_HPP_
#define HYDROGEN_IMPORTS_CUBLAS_HPP_

namespace El
{

/** \class CudaError
 *  \brief Exception class for CUDA errors.
 */
struct cublasError : std::runtime_error
{
    std::string build_error_string_(
        char const* file, int line)
    {
        std::ostringstream oss;
        oss << "cuBLAS error at " << file << ":" << line << "\n\n";
        return oss.str();
    }
    cublasError(char const* file, int line)
        : std::runtime_error{build_error_string_(file,line)}
    {}
};// struct cublasError

#define EL_FORCE_CHECK_CUBLAS(cublas_call)                              \
    do                                                                  \
    {                                                                   \
        const cublasStatus_t cublas_status = cublas_call;               \
        if (cublas_status != CUBLAS_STATUS_SUCCESS)                     \
        {                                                               \
            cudaDeviceReset();                                          \
            throw cublasError(__FILE__,__LINE__);                       \
        }                                                               \
    } while (0)


#ifdef EL_RELEASE
#define EL_CHECK_CUBLAS(cublas_call) cublas_call
#else
#define EL_CHECK_CUBLAS(cublas_call)                   \
    do                                                 \
    {                                                  \
        EL_FORCE_CHECK_CUBLAS(cublas_call);            \
        EL_FORCE_CHECK_CUDA(cudaDeviceSynchronize());  \
    } while (0)
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
