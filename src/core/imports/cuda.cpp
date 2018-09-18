#include "El-lite.hpp"
#include "El/core/imports/cuda.hpp"
#ifdef HYDROGEN_HAVE_CUB
#include "El/core/imports/cub.hpp"
#endif // HYDROGEN_HAVE_CUB

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>

#include <cstdlib>// getenv

namespace El
{

// Global static pointer used to ensure a single instance of the
// GPUManager class.
std::unique_ptr<GPUManager> GPUManager::instance_ = nullptr;

void InitializeCUDA(int argc, char* argv[])
{

    unsigned int numDevices = 0;
    int device = 0;

    nvmlReturn_t r = nvmlInit();
    if (r != NVML_SUCCESS) { RuntimeError("NVML error"); }
    r = nvmlDeviceGetCount(&numDevices);
    if (r != NVML_SUCCESS) { RuntimeError("NVML error"); }
    r = nvmlShutdown();
    if (r != NVML_SUCCESS) { RuntimeError("NVML error"); }
    switch (numDevices)
    {
    case 0: return;
    case 1: device = 0; break;
    default:

        // Get local rank (rank within compute node)
        int localRank = 0;
        char* env = nullptr;
        if (!env) { env = std::getenv("SLURM_LOCALID"); }
        if (!env) { env = std::getenv("MV2_COMM_WORLD_LOCAL_RANK"); }
        if (!env) { env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"); }
        if (env) { localRank = std::atoi(env); }

        // Try assigning GPUs to local ranks in round-robin fashion
        device = localRank % numDevices;
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
    nvmlReturn_t r = nvmlInit();
    if (r != NVML_SUCCESS) { RuntimeError("NVML error"); }
    r = nvmlDeviceGetCount(&numDevices_);
    if (r != NVML_SUCCESS) { RuntimeError("NVML error"); }
    r = nvmlShutdown();
    if (r != NVML_SUCCESS) { RuntimeError("NVML error"); }
    if (device_ < 0 || (unsigned int) device_ >= numDevices_)
    {
        RuntimeError("Attempted to set invalid CUDA device ",
                     "(requested device ",device_,", ",
                     "but there are ",numDevices_," available devices)");
    }

    // Initialize CUDA and cuBLAS objects
    // Can use the runtime API without creating unneeded contexts now.
    // This will fail with a CUDA error if device_ is in prohibited mode or
    // if it is in process-exclusive mode and another process already has it.
    EL_FORCE_CHECK_CUDA_NOSYNC(cudaSetDevice(device_));
    EL_FORCE_CHECK_CUDA(cudaStreamCreate(&stream_));
    EL_FORCE_CHECK_CUDA(cudaEventCreate(&event_));
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
        // Destroy CUDA resources only when the device is still active
        cudaError_t e = cudaSetDevice(device_);
        if (e != cudaSuccess)
            return;

        if (cublasHandle_ != nullptr)
            EL_FORCE_CHECK_CUBLAS(cublasDestroy(cublasHandle_));

        if (stream_ != nullptr)
            EL_FORCE_CHECK_CUDA(cudaStreamDestroy(stream_));

        if (event_ != nullptr)
            EL_FORCE_CHECK_CUDA(cudaEventDestroy(event_));
    }
    catch (std::exception const& e)
    {
        std::cerr << "Error detected in ~GPUManager():\n\n"
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

Unsigned GPUManager::NumDevices()
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

cudaEvent_t GPUManager::Event()
{
    return Instance()->event_;
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
