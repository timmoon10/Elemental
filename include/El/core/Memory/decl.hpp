/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_MEMORY_DECL_HPP
#define EL_MEMORY_DECL_HPP

#ifndef EL_CACHE_LINE_SIZE
// Cache lines on x86 systems are 64B
#define EL_CACHE_LINE_SIZE 64
#endif

namespace El {

namespace memory {

template<typename T=char> inline
Int CacheLineSize()
{
    return Max( Int( EL_CACHE_LINE_SIZE / sizeof(T) ), Int(1) );
}

// Get the first position in buf aligned with a cache line
template<typename T>
Int AlignmentOffset( const T* buf );

} // namespace memory

template<typename G>
class Memory
{
    size_t size_;
    G* rawBuffer_;
    G* buffer_;
public:
    Memory();
    Memory( size_t size );
    ~Memory();

    Memory( Memory<G>&& mem );
    Memory<G>& operator=( Memory<G>&& mem );

    // Exchange metadata with 'mem'
    void ShallowSwap( Memory<G>& mem );

    G* Buffer() const EL_NO_EXCEPT;
    size_t Size() const EL_NO_EXCEPT;

    G* Require( size_t size );
    void Release();
    void Empty();
};

} // namespace El

#endif // ifndef EL_MEMORY_DECL_HPP
