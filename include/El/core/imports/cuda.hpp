#ifndef HYDROGEN_IMPORTS_CUDA_HPP_
#define HYDROGEN_IMPORTS_CUDA_HPP_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


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


void InitializeCUDA(int,char*[]);

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

    void set_local_device_id(int gpu_id) noexcept { device_id_ = gpu_id; }
    int get_local_device_id() const noexcept { return device_id_; }

    void create_local_stream() {
      cuda_stream_ = 0;
      /*EL_FORCE_CHECK_CUDA(cudaStreamCreate(&cuda_stream_));*/ }
    cudaStream_t get_local_stream() const noexcept { return cuda_stream_; }

    void create_local_cublas_handle() {
      cublasCreate(&cublas_handle_);
      cublasSetStream(cublas_handle_, cuda_stream_);
      cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_HOST);
      // EL_FORCE_CHECK_CUBLAS(cublasCreate(&cublas_handle_));
      // EL_FORCE_CHECK_CUBLAS(cublasSetStream(cublas_handle_, cuda_stream_));
      // EL_FORCE_CHECK_CUBLAS(cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
    }
    cublasHandle_t get_local_cublas_handle() const noexcept { return cublas_handle_; }

    void set_local_device_count(int num_devices) noexcept { device_count_ = num_devices; }
    int get_local_device_count() const noexcept { return device_count_; }

    /** Use implicit type conversion to return the cuBLAS handle */
    operator cublasHandle_t()
    {
        return cublas_handle_;
    }

    ~GPUManager()
    {
      if (cuda_stream_) {
        if (cudaStreamDestroy(cuda_stream_) != cudaSuccess) {
          std::terminate();
        }
      }
      if (cublas_handle_) {
        if (cublasDestroy(cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
          std::terminate();
        }
      }
    }

private:
    static std::unique_ptr<GPUManager> instance_;
    /** GPU device ID */
    int device_id_;
    /** CUDA stream. */
    cudaStream_t cuda_stream_;
    /** cuBLAS handle */
    cublasHandle_t cublas_handle_;
    int device_count_;

    GPUManager()
      : device_id_{-1}, cuda_stream_{nullptr}, cublas_handle_{nullptr},
        device_count_{0}
    {}

};// class GPUManager

}// namespace El

#endif // HYDROGEN_IMPORTS_CUDA_HPP_
