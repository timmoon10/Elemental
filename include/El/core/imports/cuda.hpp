#ifndef HYDROGEN_IMPORTS_CUDA_HPP_
#define HYDROGEN_IMPORTS_CUDA_HPP_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


namespace El
{

/** CudaError
 *  Exception class for CUDA errors.
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
}; // struct CudaError

#define EL_CUDA_SYNC(async)                                             \
    do                                                                  \
    {                                                                   \
        /* Synchronize GPU and check for errors. */                     \
        cudaError_t status_CUDA_SYNC = cudaDeviceSynchronize();         \
        if (status_CUDA_SYNC == cudaSuccess)                            \
            status_CUDA_SYNC = cudaGetLastError();                      \
        if (status_CUDA_SYNC != cudaSuccess) {                          \
            cudaDeviceReset();                                          \
            throw CudaError(status_CUDA_SYNC,__FILE__,__LINE__,async);  \
        }                                                               \
    }                                                                   \
    while( 0 )
#define EL_FORCE_CHECK_CUDA(cuda_call)                                  \
    do                                                                  \
    {                                                                   \
        /* Call CUDA API routine, synchronizing before and after to */  \
        /* check for errors. */                                         \
        EL_CUDA_SYNC(true);                                             \
        cudaError_t status_CHECK_CUDA = cuda_call ;                     \
        if( status_CHECK_CUDA != cudaSuccess ) {                        \
            cudaDeviceReset();                                          \
            throw CudaError(status_CHECK_CUDA,__FILE__,__LINE__,false); \
        }                                                               \
        EL_CUDA_SYNC(false);                                            \
    } while (0)
#define EL_LAUNCH_CUDA_KERNEL(kernel, Dg, Db, Ns, S, args)      \
    do                                                          \
    {                                                           \
        /* Dg is a dim3 specifying grid dimensions. */          \
        /* Db is a dim3 specifying block dimensions. */         \
        /* Ns is a size_t specifying dynamic memory. */         \
        /* S is a cudaStream_t specifying stream. */            \
        kernel <<< Dg, Db, Ns, S >>> args ;                     \
    }                                                           \
    while (0)
#define EL_FORCE_CHECK_CUDA_KERNEL(kernel, Dg, Db, Ns, S, args) \
    do                                                          \
    {                                                           \
        /* Launch CUDA kernel, synchronizing before */          \
        /* and after to check for errors. */                    \
        EL_CUDA_SYNC(true);                                     \
        EL_LAUNCH_CUDA_KERNEL(kernel, Dg, Db, Ns, S, args);     \
        EL_CUDA_SYNC(false);                                    \
    }                                                           \
    while (0)

#ifdef EL_RELEASE
#define EL_CHECK_CUDA( cuda_call ) cuda_call
#define EL_CHECK_CUDA_KERNEL(kernel, Dg, Db, Ns, S, args) \
  EL_LAUNCH_CUDA_KERNEL(kernel, Dg, Db, Ns, S, args)
#else
#define EL_CHECK_CUDA( cuda_call ) EL_FORCE_CHECK_CUDA( cuda_call )
#define EL_CHECK_CUDA_KERNEL(kernel, Dg, Db, Ns, S, args) \
  EL_FORCE_CHECK_CUDA_KERNEL(kernel, Dg, Db, Ns, S, args)
#endif // #ifdef EL_RELEASE

/** Initialize CUDA environment. */
void InitializeCUDA(int,char*[]);
/** Finalize CUDA environment. */
void FinalizeCUDA();

/** Singleton class to manage CUDA objects.
 *  This class also manages cuBLAS objects. Note that the CUDA device
 *  is set whenever the singleton instance is requested, i.e. in most
 *  of the static functions.
 */
class GPUManager
{
public:

    GPUManager( const GPUManager& ) = delete;
    GPUManager& operator=( const GPUManager& ) = delete;
    ~GPUManager();

    /** Create new singleton instance of CUDA manager. */
    static void Create( int device = 0 );
    /** Destroy singleton instance of CUDA manager. */
    static void Destroy();
    /** Get singleton instance of CUDA manager. */
    static GPUManager* Instance();
    /** Get number of visible CUDA devices. */
    static int NumDevices();
    /** Get currently active CUDA device. */
    static int Device();
    /** Set active CUDA device. */
    static void SetDevice( int device );
    /** Get CUDA stream. */
    static cudaStream_t Stream();
    /** Get cuBLAS handle. */
    static cublasHandle_t cuBLASHandle();

private:  

    /** Singleton instance. */
    static std::unique_ptr<GPUManager> instance_;

    /** Number of visible CUDA devices. */
    int numDevices_;
    /** Currently active CUDA device. */
    int device_;
    /** CUDA stream. */
    cudaStream_t stream_;
    /** cuBLAS handle */
    cublasHandle_t cublasHandle_;

    GPUManager( int device = 0 );

}; // class GPUManager

} // namespace El

#endif // HYDROGEN_IMPORTS_CUDA_HPP_
