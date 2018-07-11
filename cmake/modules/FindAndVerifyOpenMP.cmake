
# Attempt to use the built-in module
find_package(OpenMP COMPONENTS CXX)

if (NOT OpenMP_FOUND AND APPLE)

  find_library(_OpenMP_LIBRARY
    NAMES omp gomp iomp5md
    HINTS ${OpenMP_DIR} $ENV{OpenMP_DIR}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
    DOC "The libomp library.")
  mark_as_advanced(_OpenMP_LIBRARY)

  if (NOT _OpenMP_LIBRARY)
    message(WARNING "No OpenMP library found.")
  else ()
    get_filename_component(_OpenMP_LIB_DIR "${_OpenMP_LIBRARY}" DIRECTORY)

    if (${_OpenMP_LIBRARY} MATCHES "libomp*")
      set(OpenMP_libomp_LIBRARY ${_OpenMP_LIBRARY}
        CACHE PATH "The OpenMP omp library.")
      foreach (lang IN ITEMS C CXX)
        set(OpenMP_${lang}_LIB_NAMES libomp)
        set(OpenMP_${lang}_FLAGS "-fopenmp=libomp")
      endforeach ()
    elseif (${_OpenMP_LIBRARY} MATCHES "libgomp*")
      set(OpenMP_libgomp_LIBRARY ${_OpenMP_LIBRARY}
        CACHE PATH "The OpenMP gomp library.")
      foreach (lang IN ITEMS C CXX)
        set(OpenMP_${lang}_LIB_NAMES libgomp)
        set(OpenMP_${lang}_FLAGS "-fopenmp")
      endforeach ()
    elseif (${_OpenMP_LIBRARY} MATCHES "libiomp5md*")
      set(OpenMP_libiomp5md_LIBRARY ${_OpenMP_LIBRARY}
        CACHE PATH "The OpenMP iomp5md library.")
      foreach (lang IN ITEMS C CXX)
        set(OpenMP_${lang}_LIB_NAMES libiomp5md)
        set(OpenMP_${lang}_FLAGS "-fopenmp=libiomp5")
      endforeach ()
    endif ()

    # Let's try this again
    find_package(OpenMP REQUIRED)
    foreach (lang IN ITEMS C CXX)
      set_property(TARGET OpenMP::OpenMP_${lang} APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES "-L${_OpenMP_LIB_DIR}")
    endforeach ()

  endif (NOT _OpenMP_LIBRARY)
endif (NOT OpenMP_FOUND AND APPLE)

set(_OPENMP_TEST_SOURCE
  "
#include <omp.h>
int main() {
#pragma omp parallel
{
  auto x = omp_get_num_threads();
}
}")

include(CheckCXXSourceRuns)
set(CMAKE_REQUIRED_FLAGS "${OpenMP_CXX_FLAGS}")
set(CMAKE_REQUIRED_LIBRARIES OpenMP::OpenMP_CXX)
check_cxx_source_runs("${_OPENMP_TEST_SOURCE}" _OPENMP_TEST_RUNS)
unset(CMAKE_REQUIRED_FLAGS)
unset(CMAKE_REQUIRED_LIBRARIES)

get_target_property(_OMP_FLAGS OpenMP::OpenMP_CXX INTERFACE_COMPILE_OPTIONS)
set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
  INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:${_OMP_FLAGS}>)

set(OpenMP_FOUND ${_OPENMP_TEST_RUNS})

if (OpenMP_FOUND)
  set(EL_HAVE_OPENMP TRUE)
else ()
  set(EL_HAVE_OPENMP FALSE)
endif ()

if (EL_HAVE_OPENMP)
  set(CMAKE_REQUIRED_FLAGS ${OpenMP_CXX_FLAGS})
  set(OMP_COLLAPSE_CODE
    "#include <omp.h>
     int main( int argc, char* argv[] )
     {
         int k[100];
     #pragma omp collapse(2)
         for( int i=0; i<10; ++i )
             for( int j=0; j<10; ++j )
                 k[i+j*10] = i+j;
         return 0;
     }")
  check_cxx_source_compiles("${OMP_COLLAPSE_CODE}" EL_HAVE_OMP_COLLAPSE)
  set(CMAKE_REQUIRED_FLAGS)
else ()
  set(EL_HAVE_OMP_COLLAPSE FALSE)
endif ()

# See if we have 'simd' support, which was introduced in OpenMP 4.0
if (EL_HAVE_OPENMP)
  set(CMAKE_REQUIRED_FLAGS ${OpenMP_CXX_FLAGS})
  set(OMP_SIMD_CODE
      "#include <omp.h>
       int main( int argc, char* argv[] )
       {
           int k[10];
       #pragma omp simd
           for( int i=0; i<10; ++i )
               k[i] = i;
           return 0;
       }")
  check_cxx_source_compiles("${OMP_SIMD_CODE}" EL_HAVE_OMP_SIMD)
  set(CMAKE_REQUIRED_FLAGS)
else ()
  set(EL_HAVE_OMP_SIMD FALSE)
endif ()

# See if we have 'taskloop' support, which was introduced in OpenMP 4.0
if (EL_HAVE_OPENMP)
  set(CMAKE_REQUIRED_FLAGS ${OpenMP_CXX_FLAGS})
  set(OMP_TASKLOOP_CODE
      "#include <omp.h>
       int main( int argc, char* argv[] )
       {
           int k[10];
       #pragma omp taskloop
           for( int i=0; i<10; ++i )
               k[i] = i;
           return 0;
       }")
  check_cxx_source_compiles("${OMP_TASKLOOP_CODE}" HYDROGEN_HAVE_OMP_TASKLOOP)
  set(CMAKE_REQUIRED_FLAGS)
else ()
  set(HYDROGEN_HAVE_OMP_TASKLOOP FALSE)
endif ()
