#include "El-lite.hpp"
#include "El/core/imports/cuda.hpp"
#ifdef HYDROGEN_HAVE_CUB
#include "El/core/imports/cub.hpp"
#endif // HYDROGEN_HAVE_CUB

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdlib>// getenv

namespace El
{

// Global static pointer used to ensure a single instance of the
// GPUManager class.
std::unique_ptr<GPUManager> GPUManager::instance_ = nullptr;

void InitializeCUDA(int argc, char* argv[])
{

    int numDevices = 0;
    int device = 0;
    cudaDeviceProp deviceProp;

    EL_FORCE_CHECK_CUDA_NOSYNC(
        cudaGetDeviceCount(&numDevices));
    switch (numDevices)
    {
    case 0: return;
    case 1: device = 0; break;
    default:

        // Get local rank (rank within compute node)
        int localRank = 0;
        char* env=nullptr;
        if (!env) { env = std::getenv("SLURM_LOCALID"); }
        if (!env) { env = std::getenv("MV2_COMM_WORLD_LOCAL_RANK"); }
        if (!env) { env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"); }
        if (env) { localRank = std::atoi(env); }

        // Try assigning GPUs to local ranks in round-robin fashion
        device = localRank % numDevices;

        // Check GPU compute mode
        EL_FORCE_CHECK_CUDA_NOSYNC(
            cudaGetDeviceProperties(&deviceProp, device));
        switch (deviceProp.computeMode)
        {
        case cudaComputeModeExclusive:
        case cudaComputeModeExclusiveProcess:
            if (localRank >= numDevices)
            {
                cudaDeviceReset();
                RuntimeError("CUDA device ",device," has a compute mode "
                             "that does not permit sharing amongst ",
                             "multiple MPI ranks");
            }
            else
            {
                // Let CUDA handle GPU assignments
                EL_FORCE_CHECK_CUDA_NOSYNC(cudaGetDevice(&device));
            }
            break;
        }

    }

    // Check GPU compute mode
    EL_FORCE_CHECK_CUDA_NOSYNC(
        cudaGetDeviceProperties(&deviceProp, device));
    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        cudaDeviceReset();
        RuntimeError("CUDA Device ",device,
                     " is set with ComputeModeProhibited");
    }

    // Instantiate CUDA manager
    GPUManager::Create(device);
}

void FinalizeCUDA()
{
#ifdef HYDROGEN_HAVE_CUB
    cub::DestroyMemoryPool();
#endif // HYDROGEN_HAVE_CUB
    GPUManager::Destroy();
}

GPUManager::GPUManager(int device)
    : numDevices_{0}, device_{device}, stream_{nullptr}, cublasHandle_{nullptr}
{
    // Check if device is valid
    EL_FORCE_CHECK_CUDA_NOSYNC(
        cudaGetDeviceCount(&numDevices_));
    if (device_ < 0 || device_ >= numDevices_)
    {
        RuntimeError("Attempted to set invalid CUDA device ",
                     "(requested device ",device_,", ",
                     "but there are ",numDevices_," available devices)");
    }

    // Initialize CUDA and cuBLAS objects
    EL_FORCE_CHECK_CUDA_NOSYNC(cudaSetDevice(device_));
    EL_FORCE_CHECK_CUDA(cudaStreamCreate(&stream_));
}

void GPUManager::InitializeCUBLAS()
{
    EL_FORCE_CHECK_CUBLAS(cublasCreate(&Instance()->cublasHandle_));
    EL_FORCE_CHECK_CUBLAS(cublasSetStream(cuBLASHandle(), Stream()));
    EL_FORCE_CHECK_CUBLAS(cublasSetPointerMode(cuBLASHandle(),
                                               CUBLAS_POINTER_MODE_HOST));
}

GPUManager::~GPUManager()
{
    try
    {
        EL_FORCE_CHECK_CUDA(cudaSetDevice(device_));
        if (cublasHandle_ != nullptr)
            EL_FORCE_CHECK_CUBLAS(cublasDestroy(cublasHandle_));

        if (stream_ != nullptr)
            EL_FORCE_CHECK_CUDA(cudaStreamDestroy(stream_));
    }
    catch (std::exception const& e)
    {
        std::cerr << "cudaFree error detected:\n\n"
                  << e.what() << std::endl
                  << "std::terminate() will be called."
                  << std::endl;
        std::terminate();
    }
}

void GPUManager::Create(int device)
{
    instance_.reset(new GPUManager(device));
}

void GPUManager::Destroy()
{
    instance_.reset();
}

GPUManager* GPUManager::Instance()
{
    if (!instance_)
        Create();

    EL_CHECK_CUDA(
        cudaSetDevice(instance_->device_));
    return instance_.get();
}

int GPUManager::NumDevices()
{
    return Instance()->numDevices_;
}

int GPUManager::Device()
{
    return Instance()->device_;
}

void GPUManager::SetDevice(int device)
{
    if (instance_ && instance_->device_ != device)
        Destroy();
    if (!instance_)
        Create(device);
}

cudaStream_t GPUManager::Stream()
{
    return Instance()->stream_;
}

void GPUManager::SynchronizeStream()
{
    EL_CHECK_CUDA(
        cudaSetDevice(Device()));
    EL_CHECK_CUDA(
        cudaStreamSynchronize(Stream()));
}

void GPUManager::SynchronizeDevice( bool checkError )
{
    EL_CHECK_CUDA(
        cudaSetDevice(Device()));
    if (checkError)
    {
        // Synchronize with error check
        EL_CUDA_SYNC(true);
    }
    else
    {
        // Synchronize with no error check in release build
        EL_CHECK_CUDA(
            cudaDeviceSynchronize());
    }
}

cublasHandle_t GPUManager::cuBLASHandle()
{
    auto* instance = Instance();
    auto handle = instance->cublasHandle_;
    EL_CHECK_CUBLAS(
        cublasSetStream(handle, instance->stream_));
    EL_CHECK_CUBLAS(
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    return handle;
}

} // namespace El
