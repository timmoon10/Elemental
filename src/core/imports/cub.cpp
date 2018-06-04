#include "El-lite.hpp"
#include "El/core/imports/cub.hpp"

namespace
{
/** Singleton instance of CUB memory pool. */
std::unique_ptr<::cub::CachingDeviceAllocator> memoryPool_;
} // namespace <anon>

namespace El
{
namespace cub
{

::cub::CachingDeviceAllocator& MemoryPool()
{
    if (!memoryPool_)
        memoryPool_.reset(new ::cub::CachingDeviceAllocator(2u));
    return *memoryPool_;
}

void DestroyMemoryPool()
{ memoryPool_.reset(); }

} // namespace CUBMemoryPool
} // namespace El
