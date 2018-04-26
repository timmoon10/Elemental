/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_CORE_MEMORY_IMPL_HPP_
#define EL_CORE_MEMORY_IMPL_HPP_

#include <iostream>
#include <sstream>

#ifdef HYDROGEN_HAVE_CUDA
#include <cuda_runtime.h>
#endif // HYDROGEN_HAVE_CUDA
#ifdef HYDROGEN_HAVE_CUB
#include "cub/util_allocator.cuh"
#endif // HYDROGEN_HAVE_CUB

#include "El/hydrogen_config.h"
#include "decl.hpp"

namespace El
{

namespace
{

// Partially specializable object
template <typename G, Device D>
struct MemHelper;

// CPU impls are very simple
template <typename G>
struct MemHelper<G,Device::CPU>
{
    static G* New( size_t size, unsigned int mode )
    {
        G* ptr = nullptr;
        switch (mode) {
        case 0: ptr = new G[size]; break;
#ifdef HYDROGEN_HAVE_CUDA
        case 1:
            {
                // Pinned memory
                auto error = cudaMallocHost(&ptr, size);
                if (error != cudaSuccess)
                {
                    RuntimeError("Failed to allocate pinned memory with message: ",
                                 "\"", cudaGetErrorString(error), "\"");
                }
            }
            break;
#endif // HYDROGEN_HAVE_CUDA
        default: RuntimeError("Invalid CPU memory allocation mode");
        }
        return ptr;
    }
    static void Delete( G*& ptr, unsigned int mode )
    {
        switch (mode) {
        case 0: delete[] ptr; break;
#ifdef HYDROGEN_HAVE_CUDA
        case 1:
            {
                // Pinned memory
                auto error = cudaFreeHost(ptr);
                if (error != cudaSuccess)
                {
                    RuntimeError("Failed to free pinned memory with message: ",
                                 "\"", cudaGetErrorString(error), "\"");
                }
            }
            break;
#endif // HYDROGEN_HAVE_CUDA
        default: RuntimeError("Invalid CPU memory deallocation mode");
        }
        ptr = nullptr;
    }
    static void MemZero( G* buffer, size_t numEntries, unsigned int mode )
    {
        MemZero(buffer, numEntries);
    }
};

#ifdef HYDROGEN_HAVE_CUDA

#ifdef HYDROGEN_HAVE_CUB
// GPU memory pool
cub::CachingDeviceAllocator cubMemPool(2);
#endif // HYDROGEN_HAVE_CUB

// GPU impls are just a smidge longer
template <typename G>
struct MemHelper<G,Device::GPU>
{
    static G* New( size_t size, unsigned int mode )
    {
        G* dptr = nullptr;
#ifdef HYDROGEN_HAVE_CUB
        cudaStream_t stream = 0; // TODO: non-default stream
        auto error = cubMemPool.DeviceAllocate(&dptr,
                                               size * sizeof(G),
                                               stream);
        if (error != cudaSuccess)
        {
            RuntimeError("CUB memory pool failed to allocate GPU memory ",
                         "with message: \"", cudaGetErrorString(error), "\"");
        }
#else
        auto error = cudaMalloc(&dptr, size * sizeof(G));
        if (error != cudaSuccess)
        {
            RuntimeError("cudaMalloc failed with message: \"",
                         cudaGetErrorString(error), "\"");
        }
#endif // HYDROGEN_HAVE_CUB
        return dptr;
    }

    static void Delete( G*& ptr, unsigned int mode )
    { 
#ifdef HYDROGEN_HAVE_CUB
        auto error = cubMemPool.DeviceFree(dptr);
        if (error != cudaSuccess)
        {
            RuntimeError("CUB memory pool failed to deallocate GPU memory ",
                         "with message: \"", cudaGetErrorString(error), "\"");
        }
#else
        auto error = cudaFree(ptr);
        if (error != cudaSuccess)
        {
            RuntimeError("cudaFree failed with message: \"",
                         cudaGetErrorString(error), "\"");
        }
#endif // HYDROGEN_HAVE_CUB
        ptr = nullptr;
    }

    static void MemZero( G* buffer, size_t numEntries, unsigned int mode )
    {
        cudaStream_t stream = 0; // TODO: non-default stream
        auto error = cudaMemsetAsync(buffer,
                                     0,
                                     numEntries * sizeof(G),
                                     stream);
        if (error != cudaSuccess)
        {
            RuntimeError("cudaMemset failed with message: \"",
                         cudaGetErrorString(error), "\"");
        }
    }

};

#endif // HYDROGEN_HAVE_CUDA

} // namespace <anonymous>

template<typename G, Device D>
Memory<G,D>::Memory()
: size_(0), rawBuffer_(nullptr), buffer_(nullptr), mode_(0)
{ }

template<typename G, Device D>
Memory<G,D>::Memory(size_t size, unsigned int mode)
: size_(0), rawBuffer_(nullptr), buffer_(nullptr), mode_(mode)
{ Require(size); }

template<typename G, Device D>
Memory<G,D>::Memory(Memory<G,D>&& mem)
: size_(mem.size_), rawBuffer_(nullptr), buffer_(nullptr), mode_(0)
{ ShallowSwap(mem); }

template<typename G, Device D>
Memory<G,D>& Memory<G,D>::operator=(Memory<G,D>&& mem)
{ ShallowSwap(mem); return *this; }

template<typename G, Device D>
void Memory<G,D>::ShallowSwap(Memory<G,D>& mem)
{
    std::swap(size_,mem.size_);
    std::swap(rawBuffer_,mem.rawBuffer_);
    std::swap(buffer_,mem.buffer_);
    std::swap(mode_,mem.mode_);
}

template<typename G, Device D>
Memory<G,D>::~Memory()
{ Empty(); }

template<typename G, Device D>
G* Memory<G,D>::Buffer() const EL_NO_EXCEPT { return buffer_; }

template<typename G, Device D>
size_t  Memory<G,D>::Size() const EL_NO_EXCEPT { return size_; }

template<typename G, Device D>
G* Memory<G,D>::Require(size_t size)
{
    if(size > size_)
    {
        Empty();
#ifndef EL_RELEASE
        try
        {
#endif
            // TODO: Optionally overallocate to force alignment of buffer_
            rawBuffer_ = MemHelper<G,D>::New(size, mode_);
            buffer_ = rawBuffer_;
            size_ = size;
#ifndef EL_RELEASE
        }
        catch(std::bad_alloc& e)
        {
            size_ = 0;
            std::ostringstream os;
            os << "Failed to allocate " << size*sizeof(G)
               << " bytes on process " << mpi::Rank() << std::endl;
            std::cerr << os.str();
            throw e;
        }
#endif
#ifdef EL_ZERO_INIT
        MemHelper<G,D>::MemZero(buffer_, size_, mode_);
#elif defined(EL_HAVE_VALGRIND)
        if(EL_RUNNING_ON_VALGRIND)
            MemHelper<G,D>::MemZero(buffer_, size_, mode_);
#endif
    }
    return buffer_;
}

template<typename G, Device D>
void Memory<G,D>::Release()
{ this->Empty(); }

template<typename G, Device D>
void Memory<G,D>::Empty()
{
    if(rawBuffer_ != nullptr)
    {
        MemHelper<G,D>::Delete(rawBuffer_, mode_);
    }
    buffer_ = nullptr;
    size_ = 0;
}

template<typename G, Device D>
void Memory<G,D>::SetMode(unsigned int mode)
{
    if (size_ > 0 && mode_ != mode)
    {
        G* newRawBuffer = MemHelper<G,D>::New(size_, mode);
        G* newBuffer = newRawBuffer;
        MemCopy(newBuffer, buffer_, size_);
        MemHelper<G,D>::Delete(rawBuffer_, mode_);
        rawBuffer_ = newRawBuffer;
        buffer_ = newBuffer;
    }
    mode_ = mode;
}

template<typename G, Device D>
unsigned int Memory<G,D>::Mode() const
{ return mode_; }

#ifdef EL_INSTANTIATE_CORE
# define EL_EXTERN
#else
# define EL_EXTERN// extern
#endif

#if 0
#define PROTO(T) EL_EXTERN template class Memory<T,Device::CPU>;
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>
#endif // 0

EL_EXTERN template class Memory<double, Device::CPU>;

// GPU instantiations
#ifdef HYDROGEN_HAVE_CUDA
EL_EXTERN template class Memory<float, Device::GPU>;
EL_EXTERN template class Memory<double, Device::GPU>;
#endif

#undef EL_EXTERN

} // namespace El
#endif // EL_CORE_MEMORY_IMPL_HPP_
