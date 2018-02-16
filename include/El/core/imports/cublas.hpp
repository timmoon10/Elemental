#ifndef HYDROGEN_IMPORTS_CUBLAS_HPP_
#define HYDROGEN_IMPORTS_CUBLAS_HPP_

namespace El
{

void InitializeCUDA(int,char*[]);

namespace cublas
{

//
// LEVEL 3 Routines
//

#define ADD_GEMM_DECL(ScalarType)                                       \
    void Gemm(                                                          \
        char transA, char transB, BlasInt m, BlasInt n, BlasInt k,      \
        ScalarType const& alpha,                                        \
        ScalarType const* A, BlasInt ALDim,                             \
        ScalarType const* B, BlasInt BLDim,                             \
        ScalarType const& beta,                                         \
        ScalarType* C, BlasInt CLDim);

ADD_GEMM_DECL(float)
ADD_GEMM_DECL(double)

}// namespace cublas
}// namespace El
#endif
