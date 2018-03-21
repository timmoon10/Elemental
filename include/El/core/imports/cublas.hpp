#ifndef HYDROGEN_IMPORTS_CUBLAS_HPP_
#define HYDROGEN_IMPORTS_CUBLAS_HPP_

namespace El
{

void InitializeCUDA(int,char*[]);

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

// BLAS 1
ADD_AXPY_DECL(float)
ADD_AXPY_DECL(double)
ADD_COPY_DECL(float)
ADD_COPY_DECL(double)

// BLAS 3
ADD_GEMM_DECL(float)
ADD_GEMM_DECL(double)

}// namespace cublas
}// namespace El
#endif
