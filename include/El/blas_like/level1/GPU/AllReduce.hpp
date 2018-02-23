#ifndef EL_BLAS_GPU_DECL_HPP_
#define EL_BLAS_GPU_DECL_HPP_

namespace El
{

template <typename T>
void AllReduce_GPU_impl(Matrix<T,Device::GPU>& A, mpi::Comm comm, mpi::Op op);

}// namespace El
#endif // EL_BLAS_GPU_DECL_HPP_
