#include "El-lite.hpp"
#include "El/core/imports/cuda.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace El
{

// Global static pointer used to ensure a single instance of the GPUManager class.
std::unique_ptr<GPUManager> GPUManager::instance_ = nullptr;

void InitializeCUDA( int argc, char* argv[] )
{

    // Instantiate CUDA manager
    GPUManager::Instance();

    // Choose device by parsing command-line arguments
    const int numDevices = GPUManager::NumDevices();
    int device = -1;
    cudaDeviceProp deviceProp;
    // if( argc > 0 ) { device = atoi(argv[0]); }

    // Choose device by parsing environment variables
    if( device < 0 )
    {
        char *env = nullptr;
        if( env == nullptr ) { env = getenv("SLURM_LOCALID"); }
        if( env == nullptr ) { env = getenv("MV2_COMM_WORLD_LOCAL_RANK"); }
        if( env == nullptr ) { env = getenv("OMPI_COMM_WORLD_LOCAL_RANK"); }
        if( env != nullptr )
        {

            // Allocate devices amongst ranks in round-robin fashion
            const int localRank = atoi(env);
            device = localRank % numDevices;

            // If device is shared amongst MPI ranks, check its
            // compute mode
            if( localRank >= numDevices )
            {
                EL_FORCE_CHECK_CUDA( cudaGetDeviceProperties( &deviceProp,
                                                              device ) );
                switch( deviceProp.computeMode )
                {
                case cudaComputeModeExclusive:
                    cudaDeviceReset();
                    RuntimeError("Attempted to share CUDA device ",device,
                                 " amongst multiple MPI ranks, ",
                                 "but it is set to cudaComputeModeExclusive");
                    break;
                case cudaComputeModeExclusiveProcess:
                    cudaDeviceReset();
                    RuntimeError("Attempted to share CUDA device ",device,
                                 " amongst multiple MPI ranks, ",
                                 "but it is set to cudaComputeModeExclusiveProcess");
                    break;
                }
            }

        }
    }

    // Set CUDA device
    if( device < 0 ) { device = 0; }
    GPUManager::SetDevice(device);

    // Check device compute mode
    EL_FORCE_CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device));
    if( deviceProp.computeMode == cudaComputeModeProhibited )
    {
        cudaDeviceReset();
        RuntimeError("CUDA Device ",device," is set with ComputeModeProhibited");
    }

}

GPUManager::GPUManager()
    : numDevices_{0}, device_{0}, stream_{0}, cublasHandle_{nullptr}
{

    // Make sure there are CUDA devices available
    EL_FORCE_CHECK_CUDA( cudaGetDeviceCount( &numDevices_ ) );
    if( numDevices_ < 1 )
    {
        RuntimeError("No CUDA devices found!");
    }

    // Initialize CUDA objects
    EL_CHECK_CUDA( cudaSetDevice( device_ ) );
    EL_CHECK_CUDA( cudaStreamCreate( &stream_ ) );
    
    // Initialize cuBLAS
    EL_FORCE_CHECK_CUBLAS( cublasCreate( &cublasHandle_ ) );
    EL_CHECK_CUBLAS( cublasSetStream( cublasHandle_, stream_ ) );
    EL_CHECK_CUBLAS( cublasSetPointerMode( cublasHandle_,
                                           CUBLAS_POINTER_MODE_HOST ) );

}

GPUManager::~GPUManager()
{
    if( cublasHandle_ != nullptr )
    {
        EL_CHECK_CUBLAS( cublasDestroy( cublasHandle_ ) );
    }
    if( stream_ != 0 )
    {
        EL_CHECK_CUDA( cudaStreamDestroy( stream_ ) );
    }
}

GPUManager* GPUManager::Instance()
{
    if( !instance_ )
    {
        instance_.reset( new GPUManager() );
    }
    EL_CHECK_CUDA( cudaSetDevice( instance_->device_ ) );
    return instance_.get();
}

int GPUManager::NumDevices()
{ return Instance()->numDevices_; }

int GPUManager::Device()
{ return Instance()->device_; }

void GPUManager::SetDevice( int device )
{
    auto* instance = Instance();
    if( device < 0 || device >= instance->numDevices_ )
    {
      RuntimeError("Attempted to set invalid CUDA device ",
                   "(requested device ",device,", ",
                   "but there are ",instance->numDevices_," devices)");
    }
    instance->device_ = device;
    EL_CHECK_CUDA( cudaSetDevice( instance->device_ ) );
}

cudaStream_t GPUManager::Stream()
{ return Instance()->stream_; }

cublasHandle_t GPUManager::cuBLASHandle()
{
    auto* instance = Instance();
    auto handle = instance->cublasHandle_;
    EL_CHECK_CUBLAS( cublasSetStream( handle, instance->stream_ ) );
    EL_CHECK_CUBLAS( cublasSetPointerMode( handle,
                                           CUBLAS_POINTER_MODE_HOST ) );
    return handle;
}

} // namespace El
