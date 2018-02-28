/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/

#ifdef __GNUG__
// GCC-4.9 fails to implement std::align
// See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57350
namespace std {
#pragma weak align
inline void*
align(size_t __align, size_t __size, void*& __ptr, size_t& __space) noexcept
{
  const auto __intptr = reinterpret_cast<uintptr_t>(__ptr);
  const auto __aligned = (__intptr - 1u + __align) & -__align;
  const auto __diff = __aligned - __intptr;
  if ((__size + __diff) > __space)
    return nullptr;
  else
    {
      __space -= __diff;
      return __ptr = reinterpret_cast<void*>(__aligned);
    }
}
}
#endif // __GNUG__

namespace El {

namespace memory {

template<typename T>
Int AlignmentOffset( const T* buf )
{
    std::size_t lineSize = CacheLineSize<T>() * sizeof(T);
    void* alignedBuf = reinterpret_cast<void*>(const_cast<T*>(buf));
    std::size_t alignedSize = 2 * lineSize;
    std::align( lineSize, sizeof(T), alignedBuf, alignedSize );
    if( alignedBuf != nullptr ) {
        return reinterpret_cast<const T*>(alignedBuf) - buf;
    } else {
        return 0;
    }
}

} // namespace memory

namespace {

template<typename G>
static G* New( size_t size )
{
    return new G[size];
}

template<typename G>
static void Delete( G*& ptr )
{
    delete[] ptr;
    ptr = nullptr;
}

} // anonymous namespace

template<typename G>
Memory<G>::Memory()
: size_(0), rawBuffer_(nullptr), buffer_(nullptr)
{ }

template<typename G>
Memory<G>::Memory( size_t size )
: size_(0), rawBuffer_(nullptr), buffer_(nullptr)
{ Require( size ); }

template<typename G>
Memory<G>::Memory( Memory<G>&& mem )
: size_(mem.size_), rawBuffer_(nullptr), buffer_(nullptr)
{ ShallowSwap(mem); }

template<typename G>
Memory<G>& Memory<G>::operator=( Memory<G>&& mem )
{ ShallowSwap( mem ); return *this; }

template<typename G>
void Memory<G>::ShallowSwap( Memory<G>& mem )
{
    std::swap(size_,mem.size_);
    std::swap(rawBuffer_,mem.rawBuffer_);
    std::swap(buffer_,mem.buffer_);
}

template<typename G>
Memory<G>::~Memory() 
{ 
    Delete( rawBuffer_ );
}

template<typename G>
G* Memory<G>::Buffer() const EL_NO_EXCEPT { return buffer_; }

template<typename G>
size_t  Memory<G>::Size() const EL_NO_EXCEPT { return size_; }

template<typename G>
G* Memory<G>::Require( size_t size )
{
    if( size > size_ )
    {
        Delete( rawBuffer_ );

#ifndef EL_RELEASE
        try {
#endif

            // TODO: Optionally overallocate to force alignment of buffer_
            rawBuffer_ = New<G>( size );
            buffer_ = rawBuffer_;

            size_ = size;
#ifndef EL_RELEASE
        } 
        catch( std::bad_alloc& e )
        {
            size_ = 0;
            ostringstream os;
            os << "Failed to allocate " << size*sizeof(G) 
               << " bytes on process " << mpi::Rank() << endl;
            cerr << os.str();
            throw e;
        }
#endif
#ifdef EL_ZERO_INIT
        MemZero( buffer_, size_ );
#elif defined(EL_HAVE_VALGRIND)
        if( EL_RUNNING_ON_VALGRIND )
            MemZero( buffer_, size_ );
#endif
    }
    return buffer_;
}

template<typename G>
void Memory<G>::Release()
{ this->Empty(); }

template<typename G>
void Memory<G>::Empty()
{
    Delete( rawBuffer_ );
    buffer_ = nullptr;
    size_ = 0;
}

#ifdef EL_INSTANTIATE_CORE
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T) EL_EXTERN template class Memory<T>;
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El
