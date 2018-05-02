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
        cudaError_t cuda_error, char const* file, int line, bool async = false)
    {
        std::ostringstream oss;
        oss << ( async ? "Asynchronous CUDA error" : "CUDA error" )
            << " (" << file << ":" << line << "): "
            << cudaGetErrorString(cuda_error);
        return oss.str();
    }
  CudaError(cudaError_t cuda_error, char const* file, int line, bool async = false)
      : std::runtime_error{build_error_string_(cuda_error,file,line,async)}
    {}
};// struct CudaError

#define EL_FORCE_CHECK_CUDA(cuda_call)                                  \
    do                                                                  \
    {                                                                   \
        {                                                               \
            /* Check for earlier asynchronous errors. */                \
            cudaError_t status_CHECK_CUDA = cudaDeviceSynchronize();    \
            if (status_CHECK_CUDA == cudaSuccess)                       \
                status_CHECK_CUDA = cudaGetLastError();                 \
            if (status_CHECK_CUDA != cudaSuccess) {                     \
                cudaDeviceReset();                                      \
                throw CudaError(status_CHECK_CUDA,__FILE__,__LINE__,true); \
            }                                                           \
        }                                                               \
        {                                                               \
            /* Make CUDA call and check for errors. */                  \
            cudaError_t status_CHECK_CUDA = (cuda_call);                \
            if (status_CHECK_CUDA == cudaSuccess)                       \
                status_CHECK_CUDA = cudaDeviceSynchronize();            \
            if (status_CHECK_CUDA == cudaSuccess)                       \
                status_CHECK_CUDA = cudaGetLastError();                 \
            if (status_CHECK_CUDA != cudaSuccess) {                     \
                cudaDeviceReset();                                      \
                throw CudaError(status_CHECK_CUDA,__FILE__,__LINE__,false); \
            }                                                           \
        }                                                               \
    } while (0)

#ifdef EL_RELEASE
#define EL_CHECK_CUDA(cuda_call) (cuda_call)
#else
#define EL_CHECK_CUDA(cuda_call) EL_FORCE_CHECK_CUDA(cuda_call)
#endif // #ifdef EL_RELEASE


void InitializeCUDA(int,char*[],int requested_device_id = -1);


class GPUManager
{
public:
    static GPUManager* getInstance()
    {
        if (!instance_)
        {
            instance_.reset(new GPUManager());;
        }
        return instance_.get();
    }

    GPUManager(const GPUManager&) = delete;
    GPUManager& operator=(const GPUManager&) = delete;

    void set_device_id(int gpu_id) noexcept { device_id_ = gpu_id; }
    int get_device_id() const noexcept { return device_id_; }

private:
    static std::unique_ptr<GPUManager> instance_;
    int device_id_;

    GPUManager()
        : device_id_{-1}
    {}

};// class GPUManager

}// namespace El

#endif // HYDROGEN_IMPORTS_CUDA_HPP_
