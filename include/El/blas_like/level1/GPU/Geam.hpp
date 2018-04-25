#ifndef EL_BLAS_LIKE_LEVEL1_GPU_GEAM_HPP_
#define EL_BLAS_LIKE_LEVEL1_GPU_GEAM_HPP_

namespace El
{

// FIXME (tym 4/24/18) Replace this with direct calls to cublas::Geam
template <typename T, typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
inline void CublasGeam(char transA, char transB, size_t m, size_t n,
                T const& alpha, T const* A, size_t ALDim,
                T const& beta, T const* B, size_t BLDim,
                T* C, size_t CLDim);

template <>
inline void CublasGeam<float>(char transA, char transB, size_t m, size_t n,
                       float const& alpha, float const* A, size_t ALDim,
                       float const& beta, float const* B, size_t BLDim,
                       float* C, size_t CLDim)
{
    cublas::Geam(transA, transB, m, n,
                 alpha, A, ALDim, beta, B, BLDim, C, CLDim);
}
template <>
inline void CublasGeam<double>(char transA, char transB, size_t m, size_t n,
                        double const& alpha, double const* A, size_t ALDim,
                        double const& beta, double const* B, size_t BLDim,
                        double* C, size_t CLDim)
{
    cublas::Geam(transA, transB, m, n,
                 alpha, A, ALDim, beta, B, BLDim, C, CLDim);
}
template <typename T,
          typename=DisableIf<IsDeviceValidType<T,Device::GPU>>,
          typename=void>
inline void CublasGeam(char transA, char transB, size_t m, size_t n,
                T const& alpha, T const* A, size_t ALDim,
                T const& beta, T const* B, size_t BLDim,
                T* C, size_t CLDim)
{
    LogicError("Geam: Type not valid on GPU.");
}


}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_GPU_GEAM_HPP_
