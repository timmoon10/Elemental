/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_MEMORY_DECL_HPP
#define EL_MEMORY_DECL_HPP

namespace El
{

template <Device D>
constexpr unsigned DefaultMemoryMode();

template <>
constexpr unsigned DefaultMemoryMode<Device::CPU>()
{
    return 0;
}

#ifdef HYDROGEN_HAVE_CUDA
template <>
constexpr unsigned DefaultMemoryMode<Device::GPU>()
{
#ifdef HYDROGEN_HAVE_CUB
    return 1;
#else
    return 0;
#endif
}
#endif // HYDROGEN_HAVE_CUB

template<typename G, Device D=Device::CPU>
class Memory
{
public:
    Memory(SyncInfo<D> const& syncInfo = SyncInfo<D>{});
    Memory(size_t size, SyncInfo<D> const& syncInfo = SyncInfo<D>{});
    Memory(size_t size, unsigned int mode,
           SyncInfo<D> const& syncInfo = SyncInfo<D>{});
    ~Memory();

    Memory(Memory<G,D>&& mem);
    Memory<G,D>& operator=(Memory<G,D>&& mem);

    // Exchange metadata with 'mem'
    void ShallowSwap(Memory<G,D>& mem);

    G* Buffer() const EL_NO_EXCEPT;
    size_t Size() const EL_NO_EXCEPT;

    G* Require(size_t size);
    void Release();
    void Empty();

    void ResetSyncInfo(SyncInfo<D> const& syncInfo = SyncInfo<D>{});
    SyncInfo<D> const& GetSyncInfo() const;

    void SetMode(unsigned int mode);
    unsigned int Mode() const;
private:
    size_t size_;
    G* rawBuffer_;
    G* buffer_;
    unsigned int mode_ = DefaultMemoryMode<D>();
    SyncInfo<D> syncInfo_ = SyncInfo<D>{};

};// class Memory

} // namespace El

#endif // ifndef EL_MEMORY_DECL_HPP
