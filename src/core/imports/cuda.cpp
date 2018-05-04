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

    int numDevices = 0;
    EL_FORCE_CHECK_CUDA( cudaGetDeviceCount( &numDevices ) );
    int device = -1;
    cudaDeviceProp deviceProp;

    // Choose device by parsing command-line arguments
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

    // Instantiate CUDA manager
    if( device < 0 ) { device = 0; }
    GPUManager::Create( device );

    // Check device compute mode
    EL_FORCE_CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device));
    if( deviceProp.computeMode == cudaComputeModeProhibited )
    {
        cudaDeviceReset();
        RuntimeError("CUDA Device ",device," is set with ComputeModeProhibited");
    }

}

void FinalizeCUDA()
{ GPUManager::Destroy(); }

GPUManager::GPUManager(int device)
    : numDevices_{0}, device_{device}, stream_{nullptr}, cublasHandle_{nullptr}
{

    // Check if device is valid
    EL_CHECK_CUDA( cudaGetDeviceCount( &numDevices_ ) );
    if( device_ < 0 || device_ >= numDevices_ )
    {
        RuntimeError("Attempted to set invalid CUDA device ",
                     "(requested device ",device_,", ",
                     "but there are ",numDevices_," available devices)");
    }

    // Initialize CUDA and cuBLAS objects
    EL_CHECK_CUDA( cudaSetDevice( device_ ) );
    EL_CHECK_CUDA( cudaStreamCreate( &stream_ ) );
    EL_CHECK_CUBLAS( cublasCreate( &cublasHandle_ ) );
    EL_CHECK_CUBLAS( cublasSetStream( cublasHandle_, stream_ ) );
    EL_CHECK_CUBLAS( cublasSetPointerMode( cublasHandle_,
                                           CUBLAS_POINTER_MODE_HOST ) );

}

GPUManager::~GPUManager()
{
    EL_CHECK_CUDA( cudaSetDevice( device_ ) );
    if( cublasHandle_ != nullptr )
    {
        EL_CHECK_CUBLAS( cublasDestroy( cublasHandle_ ) );
    }
    if( stream_ != nullptr )
    {
        EL_CHECK_CUDA( cudaStreamDestroy( stream_ ) );
    }
}

void GPUManager::Create( int device )
{ instance_.reset( new GPUManager( device ) ); }

void GPUManager::Destroy()
{ instance_.release(); }

GPUManager* GPUManager::Instance()
{
    if( !instance_ ) { Create(); }
    EL_CHECK_CUDA( cudaSetDevice( instance_->device_ ) );
    return instance_.get();
}

int GPUManager::NumDevices()
{ return Instance()->numDevices_; }

int GPUManager::Device()
{ return Instance()->device_; }

void GPUManager::SetDevice( int device )
{
    if( instance_ && instance_->device_ != device ) { Destroy(); }
    if( !instance_ ) { Create( device ); }
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
