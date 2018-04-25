#ifndef HYDROGEN_IMPORTS_CUDA_HPP_
#define HYDROGEN_IMPORTS_CUDA_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

namespace El
{

/** \class CudaError
 *  \brief Exception class for CUDA errors.
 */
struct CudaError : std::runtime_error
{
    std::string build_error_string_(
        cudaError_t cuda_error, char const* file, int line)
    {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line << "\n\n"
                  << "Error: " << cudaGetErrorString(cuda_error) << '\n';
        return oss.str();
    }
    CudaError(cudaError_t cuda_error, char const* file, int line)
        : std::runtime_error{build_error_string_(cuda_error,file,line)}
    {}
};// struct CudaError

#define EL_FORCE_CHECK_CUDA(cuda_call)                                  \
    do                                                                  \
    {                                                                   \
        const cudaError_t cuda_status = cuda_call;                      \
        if (cuda_status != cudaSuccess)                                 \
        {                                                               \
            cudaDeviceReset();                                          \
            throw CudaError(cuda_status,__FILE__,__LINE__);             \
        }                                                               \
    } while (0)

#ifdef EL_RELEASE
#define EL_CHECK_CUDA(cuda_call) cuda_call
#define EL_CHECK_CUDNN(cudnn_call) cudnn_call
#else
#define EL_CHECK_CUDA(cuda_call)                   \
  do {                                             \
    EL_FORCE_CHECK_CUDA(cuda_call);                \
    EL_FORCE_CHECK_CUDA(cudaDeviceSynchronize());  \
  } while (0)
#define EL_CHECK_CUDNN(cudnn_call)                \
  do {                                            \
    EL_FORCE_CHECK_CUDNN(cudnn_call);             \
    EL_FORCE_CHECK_CUDA(cudaDeviceSynchronize()); \
  } while (0)
#endif // #ifdef LBANN_DEBUG


void InitializeCUDA(int,char*[],int requested_device_id = -1);

}// namespace El

#endif // HYDROGEN_IMPORTS_CUDA_HPP_
