# Sets the following variables:
#
#   Quadmath_FOUND
#   QUADMATH_INCLUDE_DIR -- Location of quadmath.h
#   QUADMATH_LIBRARIES -- libquadmath library

find_path(QUADMATH_INCLUDE_DIR quadmath.h
  HINTS ${QUADMATH_DIR} $ENV{QUADMATH_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with quadmath.h header.")
find_path(QUADMATH_INCLUDE_DIRS quadmath.h)

find_library(QUADMATH_LIBRARY quadmath
  HINTS ${QUADMATH_DIR} $ENV{QUADMATH_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The QuadMath library.")
find_library(QUADMATH_LIBRARY quadmath)

# Check that we can compile with quadmath library
if (QUADMATH_LIBRARY AND QUADMATH_INCLUDE_DIR AND NOT QUADMATH_TEST_RUN)
  set(CMAKE_REQUIRED_LIBRARIES "${QUADMATH_LIBRARY}")
  set(CMAKE_REQUIRED_INCLUDES "${QUADMATH_INCLUDE_DIR}")
  set(QUADMATH_CODE "
#include <complex>
#include <iostream>
#include <quadmath.h>
int main( int argc, char* argv[] )
{
  __float128 a = 2.0q;

  char aStr[128];
  quadmath_snprintf( aStr, sizeof(aStr), \"%Q\", a );
  std::cout << aStr << std::endl;

  __complex128 y;
  std::complex<__float128> z;

  return 0;
}")

  check_cxx_source_compiles("${QUADMATH_CODE}" QUADMATH_WORKS)
  set(QUADMATH_TEST_RUN 1 CACHE INTERNAL "Whether the QUADMATH test has run")
  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_LIBRARIES)

  if (NOT QUADMATH_WORKS)

    set(${PROJECT_NAME}_ENABLE_QUADMATH OFF)
    message(WARNING "Found libquadmath but could compile with it.")

  endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Quadmath DEFAULT_MSG
  QUADMATH_LIBRARY QUADMATH_INCLUDE_DIR QUADMATH_WORKS)

if (NOT TARGET EP::quadmath)
  add_library(EP::quadmath INTERFACE IMPORTED)

  set_property(TARGET EP::quadmath
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${QUADMATH_INCLUDE_DIR}")
  set_property(TARGET EP::quadmath
    PROPERTY INTERFACE_LINK_LIBRARIES "${QUADMATH_LIBRARY}")
endif ()

set(QUADMATH_LIBRARIES EP::quadmath)
mark_as_advanced(QUADMATH_INCLUDE_DIR QUADMATH_LIBRARY)
