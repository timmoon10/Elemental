# Sets the following variables:
#
#   MPFR_FOUND
#   MPFR_VERSION_OK
#   MPFR_INCLUDE_DIR -- Location of mpfr.h
#   MPFR_LIBRARIES -- libmpfr library

if (MPFR_FIND_VERSION_COUNT EQUAL 0)
  if (MPFR_REQUIRED_VERSION)
    set(MPFR_FIND_VERSION "${MPFR_REQUIRED_VERSION}")
  else ()
    set(MPFR_FIND_VERSION "1.0.0")
  endif ()
endif ()

if (MPFR_FIND_QUIETLY)
  set(__quiet_flag "QUIET")
else ()
  unset(__quiet_flag)
endif ()
find_package(GMP "${GMP_REQUIRED_VERSION}" ${__quiet_flag})

if (GMP_FOUND)
  find_path(MPFR_INCLUDE_DIR mpfr.h
    HINTS ${MPFR_DIR} $ENV{MPFR_DIR} ${GMP_DIR} $ENV{GMP_DIR}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
    DOC "Directory with mpfr.h header.")
  find_path(MPFR_INCLUDE_DIR mpfr.h)

  find_library(MPFR_LIBRARY mpfr
    HINTS ${MPFR_DIR} $ENV{MPFR_DIR} ${GMP_DIR} $ENV{GMP_DIR}
    PATH_SUFFIXES lib64 lib
    NO_DEFAULT_PATH
    DOC "The MPFR library.")
  find_library(MPFR_LIBRARY mpfr)

  if (MPFR_LIBRARY AND MPFR_INCLUDE_DIR)
    
    set(MPFR_VERSION_CODE "
#include <iostream>
#include <gmp.h>
#include <mpfr.h>
int main(void)
{
  mpfr_t a;
  mpfr_prec_t prec = 256;
  mpfr_init2( a, prec );

  gmp_randstate_t randState;
  gmp_randinit_default( randState );
  const long seed = 1024;
  gmp_randseed_ui( randState, seed );

  std::cout << mpfr_get_version();
}")
    file(WRITE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx"
      "${MPFR_VERSION_CODE}\n")
    
    if(NOT MPFR_FIND_QUIETLY)
      message(STATUS "Performing Test MPFR_VERSION_COMPILES")
    endif()
    
    try_run(MPFR_VERSION_RUNS MPFR_VERSION_COMPILES
      ${CMAKE_BINARY_DIR}
      ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx
      LINK_LIBRARIES "${MPFR_LIBRARY}" "${GMP_LIBRARY}"
      CMAKE_FLAGS
      -DCOMPILE_DEFINITIONS=-DMPFR_VERSION_COMPILES
      "-DINCLUDE_DIRECTORIES=${MPFR_INCLUDE_DIR};${GMP_INCLUDE_DIR}"
      -DCMAKE_SKIP_RPATH:BOOL=${CMAKE_SKIP_RPATH}
      COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
      RUN_OUTPUT_VARIABLE RUN_OUTPUT)

    if (NOT MPFR_VERSION_RUNS STREQUAL "FAILED_TO_RUN")
      if (RUN_OUTPUT VERSION_LESS MPFR_FIND_VERSION)
        set(MPFR_VERSION_OK FALSE)
      else ()
        set(MPFR_VERSION_FOUND "${RUN_OUTPUT}")
        set(MPFR_VERSION_OK TRUE)
      endif ()
    else ()
      
      message(WARNING "Found libmpfr but could compile with it.")

    endif (NOT MPFR_VERSION_RUNS STREQUAL "FAILED_TO_RUN")
  endif (MPFR_LIBRARY AND MPFR_INCLUDE_DIR)
endif (GMP_FOUND)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPFR DEFAULT_MSG
  MPFR_VERSION_FOUND MPFR_LIBRARY MPFR_INCLUDE_DIR MPFR_VERSION_OK)

if (MPFR_FOUND)
  if (NOT TARGET EP::mpfr)
    add_library(EP::mpfr INTERFACE IMPORTED)
    
    set_property(TARGET EP::mpfr
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPFR_INCLUDE_DIR}")
    set_property(TARGET EP::mpfr
      PROPERTY INTERFACE_LINK_LIBRARIES "${MPFR_LIBRARY}")
  endif ()
  
  set(MPFR_LIBRARIES EP::mpfr ${GMP_LIBRARIES})
  mark_as_advanced(MPFR_LIBRARY MPFR_INCLUDE_DIR)
endif ()
