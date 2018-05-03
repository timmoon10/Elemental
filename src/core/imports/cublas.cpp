#include "El-lite.hpp"
#include "El/core/imports/cuda.hpp"
#include "El/core/imports/cublas.hpp"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace El
{

namespace cublas
{

namespace
{

inline cublasOperation_t CharTocuBLASOp(char c)
{
    switch (c)
    {
    case 'N':
        return CUBLAS_OP_N;
        break;
    case 'T':
        return CUBLAS_OP_T;
        break;
    case 'C':
        return CUBLAS_OP_C;
        break;
    default:
        RuntimeError("cuBLAS: Unknown operation type.");
        return CUBLAS_OP_N; // Compiler yells about not returning anything...
    }
}

}// namespace <anon>

//
// BLAS 1
//
#define ADD_AXPY_IMPL(ScalarType, TypeChar)                  \
    void Axpy(int n, ScalarType const& alpha,                \
              ScalarType const* X, int incx,                 \
              ScalarType* Y, int incy)                       \
    {                                                        \
        GPUManager* gpu_manager = GPUManager::getInstance(); \
        EL_CHECK_CUBLAS(cublas ## TypeChar ## axpy(          \
            *gpu_manager, n, &alpha, X, incx, Y, incy));     \
    }

#define ADD_COPY_IMPL(ScalarType, TypeChar)                 \
    void Copy(int n, ScalarType const* X, int incx,         \
              ScalarType* Y, int incy)                      \
    {                                                       \
       GPUManager* gpu_manager = GPUManager::getInstance(); \
       EL_CHECK_CUBLAS(cublas ## TypeChar ## copy(          \
            *gpu_manager, n, X, incx, Y, incy));            \
    }

//
// BLAS 2
//
#define ADD_GEMV_IMPL(ScalarType, TypeChar)                             \
    void Gemv(                                                          \
        char transA, int m, int n,                                      \
        ScalarType const& alpha,                                        \
        ScalarType const* A, int ALDim,                                 \
        ScalarType const* B, int BLDim,                                 \
        ScalarType const& beta,                                         \
        ScalarType* C, int CLDim )                                      \
    {                                                                   \
      GPUManager* gpu_manager = GPUManager::getInstance();              \
      EL_CHECK_CUBLAS(cublas ## TypeChar ## gemv(                       \
            *gpu_manager,                                               \
            CharTocuBLASOp(transA),                                     \
            m, n, &alpha, A, ALDim, B, BLDim, &beta, C, CLDim));        \
    }

//
// BLAS 3
//
#define ADD_GEMM_IMPL(ScalarType, TypeChar)                             \
    void Gemm(                                                          \
        char transA, char transB, int m, int n, int k,                  \
        ScalarType const& alpha,                                        \
        ScalarType const* A, int ALDim,                                 \
        ScalarType const* B, int BLDim,                                 \
        ScalarType const& beta,                                         \
        ScalarType* C, int CLDim )                                      \
    {                                                                   \
         GPUManager* gpu_manager = GPUManager::getInstance();           \
         EL_CHECK_CUBLAS(cublas ## TypeChar ## gemm(                    \
            *gpu_manager,                                               \
            CharTocuBLASOp(transA), CharTocuBLASOp(transB),             \
            m, n, k, &alpha, A, ALDim, B, BLDim, &beta, C, CLDim));     \
    }

//
// BLAS-like Extension
//
#define ADD_GEAM_IMPL(ScalarType, TypeChar)                             \
    void Geam(                                                          \
        char transA, char transB,                                       \
        int m, int n,                                                   \
        ScalarType const& alpha,                                        \
        ScalarType const* A, int ALDim,                                 \
        ScalarType const& beta,                                         \
        ScalarType const* B, int BLDim,                                 \
        ScalarType* C, int CLDim )                                      \
    {                                                                   \
       GPUManager* gpu_manager = GPUManager::getInstance();             \
       EL_CHECK_CUBLAS(cublas ## TypeChar ## geam(                      \
            *gpu_manager,                                               \
            CharTocuBLASOp(transA), CharTocuBLASOp(transB),             \
            m, n, &alpha, A, ALDim, &beta, B, BLDim, C, CLDim));        \
    }

// BLAS 1
ADD_AXPY_IMPL(float, S)
ADD_AXPY_IMPL(double, D)

ADD_COPY_IMPL(float, S)
ADD_COPY_IMPL(double, D)

// BLAS 2
ADD_GEMV_IMPL(float, S)
ADD_GEMV_IMPL(double, D)

// BLAS 3
ADD_GEMM_IMPL(float, S)
ADD_GEMM_IMPL(double, D)

// BLAS-like extension
ADD_GEAM_IMPL(float, S)
ADD_GEAM_IMPL(double, D)

}// namespace cublas

// Global static pointer used to ensure a single instance of the GPUManager class.
std::unique_ptr<GPUManager> GPUManager::instance_ = nullptr;

void InitializeCUDA(int argc, char*argv[])
{
    int device_count;
    auto error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess)
        RuntimeError("CUDA initialize error: ", cudaGetErrorString(error));

    if (device_count < 1)
        RuntimeError("No CUDA devices found!");

    // If the device ID is properly set, store it in the GPUManager
    GPUManager* gpu_manager = GPUManager::getInstance();
    gpu_manager->set_local_device_count(device_count);

    int requested_device_id = -1;
    // if (argc != 0) {
    //   requested_device_id = atoi(argv[0]);
    //   if(requested_device_id >= device_count) {
    //     RuntimeError("Requested device id is out of range, device count = ", device_count);
    //   }
    // }

    char *env = nullptr;
    int local_rank = 0;
    if(env == nullptr) {
      env = getenv("SLURM_LOCALID");
    }
    if(env == nullptr) {
      env = getenv("MV2_COMM_WORLD_LOCAL_RANK");
    }
    if(env == nullptr) {
      env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    }
    if(env != nullptr) {
      local_rank = atoi(env);
    }

    int device_id = local_rank;
    if(requested_device_id >= 0) {
      device_id = requested_device_id;
    }

    if(device_id >= device_count) {
        RuntimeError("Selected local rank is out of range, device count = ", device_count);
    }

    // const char* visible_devices = getenv("CUDA_VISIBLE_DEVICES");
    // if(visible_devices != nullptr && strlen(visible_devices) > 0) {
    //   std::cout << "visible gpus " << visible_devices << std::endl;
    // }

    cudaDeviceProp deviceProp;
    error = cudaGetDeviceProperties(&deviceProp, device_id);

    if (error != cudaSuccess)
        RuntimeError("CUDA initialize error: ", cudaGetErrorString(error));

    if (deviceProp.computeMode == cudaComputeModeProhibited)
        RuntimeError(std::string {} + "Device " + std::to_string(device_id)
                     + " is in ComputeModeProhibited mode. Can't use.");

    EL_FORCE_CHECK_CUDA(cudaSetDevice(device_id));

    gpu_manager->set_local_device_id(device_id);
    gpu_manager->create_local_stream();
    gpu_manager->create_local_cublas_handle();
}

}// namespace El
