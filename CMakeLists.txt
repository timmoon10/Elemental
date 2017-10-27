cmake_minimum_required(VERSION 3.5)

project(Hydrogen C CXX)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
endif ()

set(HYDROGEN_VERSION_MAJOR 0)
set(HYDROGEN_VERSION_MINOR 1)

# Prevent in-source builds
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source build attempted; please clean the CMake cache and then "
    "switch to an out-of-source build, e.g.,\n"
    "rm CMakeCache.txt && rm -Rf CMakeFiles/\nmkdir build/ && "
    "cd build/ && cmake ..\n")
endif()

# Set the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

# Add warning flags
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-pedantic" __has_pedantic)
check_cxx_compiler_flag("-Wpedantic" __has_wpedantic)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
if (__has_pedantic)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
elseif (__has_wpedantic)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
endif()

# Legacy options, with EL_ prefix intact
option(BINARY_SUBDIRECTORIES "Install binaries into tree based on type" ON)



#
# MEMORY-RELATED OPTIONS
#

option(EL_USE_64BIT_INTS "Use 64-bit integers for El indexing" OFF)
option(EL_USE_64BIT_BLAS_INTS "Use 64-bit integers for BLAS/LAPACK" OFF)
if(EL_USE_64BIT_BLAS_INTS)
  set(EL_USE_64BIT_INTS ON)
endif()

option(EL_ZERO_INIT "Initialize buffers to zero by default?" OFF)
mark_as_advanced(EL_ZERO_INIT)

option(EL_DISABLE_VALGRIND "Prevent Elemental from looking for valgrind?" OFF)
mark_as_advanced(EL_DISABLE_VALGRIND)

#
# OpenMP
#

# At one point, a bug was found in IBM's C++ compiler for Blue Gene/P,
# where OpenMP statements of the form a[i] += alpha b[i], with complex data,
# would segfault and/or return incorrect results
option(EL_AVOID_OMP_FMA "Avoid a bug in the IBM compilers." OFF)
mark_as_advanced(EL_AVOID_OMP_FMA)

#
# MPI
#

option(EL_USE_CUSTOM_ALLTOALLV "Avoid MPI_Alltoallv for performance reasons" ON)
mark_as_advanced(EL_USE_CUSTOM_ALLTOALLV)

# Since it is surprisingly common for MPI libraries to have bugs in their
# support for complex data, the following option forces Elemental to cast
# all possible MPI communications in terms of twice as many real units of data.
option(EL_AVOID_COMPLEX_MPI "Avoid potentially buggy complex MPI routines" ON)
mark_as_advanced(EL_AVOID_COMPLEX_MPI)

# Due to a subtle flaw in the Blue Gene/P extensions for MPICH2, treating
# floating-point data as a collection of byte-sized objects results in a
# better algorithm being chosen for MPI_Allgather. This should not effect
# performance on most machines.
option(EL_USE_BYTE_ALLGATHERS "Avoid BG/P allgather performance bug." ON)
mark_as_advanced(EL_USE_BYTE_ALLGATHERS)

# If MPI_Reduce_scatter_block doesn't exist, perform it by composing
# MPI_Allreduce and std::memcpy rather than MPI_Reduce and MPI_Scatter
option(EL_REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE
       "AllReduce based block MPI_Reduce_scatter" OFF)
mark_as_advanced(EL_REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE)

#
# Runtime warnings
#

# Print a warning any time a redistribution is performed which unpacks a
# large amount of data with a non-unit stride
option(EL_CACHE_WARNINGS "Warns when using cache-unfriendly routines" OFF)
mark_as_advanced(EL_CACHE_WARNINGS)

# Print a warning when an improperly aligned redistribution is performed,
# i.e., if an unnecessary permutation communication stage must take place
option(EL_UNALIGNED_WARNINGS
       "Warn when performing unaligned redistributions" OFF)
mark_as_advanced(EL_UNALIGNED_WARNINGS)

# Print a warning if an opportunity was missed to implement a redistribution
# approach specifically for vectors (instead of matrices)
option(EL_VECTOR_WARNINGS
       "Warn when vector redistribution chances are missed" OFF)
mark_as_advanced(EL_VECTOR_WARNINGS)

#
# Find third-party libraries
#

include(FindAndVerifyLAPACK)
include(FindAndVerifyMPI)

# FIXME: GET RID OF FORTRAN
include(detect/Fortran)

# External projects build internally
# TODO Investigate why
add_subdirectory(external/pmrrr)

# Add SuiteSparse (TODO: Why bundled here?)
add_subdirectory(external/suite_sparse)
include_directories(external/suite_sparse/include)

# Macro for setting up full paths
macro(set_full_path VAR)
  unset(__tmp_names)
  foreach(filename ${ARGN})
    unset(__name)
    get_filename_component(__name "${filename}" ABSOLUTE)
    list(APPEND __tmp_names "${__name}")
  endforeach()
  set(${VAR} "${__tmp_names}")
endmacro()

set(HYDROGEN_HEADERS)
set(HYDROGEN_SOURCES)
add_subdirectory(include)
add_subdirectory(src)

configure_file("${PROJECT_SOURCE_DIR}/cmake/configure_files/config.h.in"
  "${PROJECT_BINARY_DIR}/include/El/config.h")

add_library(hydrogen "${HYDROGEN_SOURCES}" "${HYDROGEN_HEADERS}")
target_include_directories(hydrogen PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/El>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include/El>)
target_link_libraries(hydrogen PUBLIC pmrrr)
target_link_libraries(hydrogen PUBLIC MPI::MPI_CXX)
target_link_libraries(hydrogen PUBLIC LAPACK::lapack)

if (BUILD_SHARED_LIBS)
  if (APPLE)
    target_link_libraries(hydrogen PUBLIC "-Wl,-undefined,error")
  else ()
    target_link_libraries(hydrogen PUBLIC "-Wl,-no-undefined")
  endif()
endif()
