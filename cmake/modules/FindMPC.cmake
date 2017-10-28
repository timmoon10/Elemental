# Sets the following variables:
#
#   MPC_FOUND
#   MPC_VERSION_OK
#   MPC_INCLUDE_DIR -- Location of mpfr.h
#   MPC_LIBRARIES -- libmpfr library

if (MPC_FIND_VERSION_COUNT EQUAL 0)
  if (MPC_REQUIRED_VERSION)
    set(MPC_FIND_VERSION "${MPC_REQUIRED_VERSION}")
  else ()
    set(MPC_FIND_VERSION "1.0.0")
  endif ()
endif ()

if (MPC_FIND_QUIETLY)
  set(__quiet_flag "QUIET")
else ()
  unset(__quiet_flag)
endif ()
find_package(MPFR "${MPFR_REQUIRED_VERSION}" ${__quiet_flag})

if (MPFR_FOUND)
  find_path(MPC_INCLUDE_DIR mpc.h
    HINTS ${MPC_DIR} $ENV{MPC_DIR} ${MPFR_DIR} $ENV{MPFR_DIR}
      ${GMP_DIR} $ENV{GMP_DIR}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
    DOC "Directory with mpc.h header.")
  find_path(MPC_INCLUDE_DIR mpc.h)

  find_library(MPC_LIBRARY mpc
    HINTS ${MPC_DIR} $ENV{MPC_DIR} ${MPFR_DIR} $ENV{MPFR_DIR}
      ${GMP_DIR} $ENV{GMP_DIR}
    PATH_SUFFIXES lib64 lib
    NO_DEFAULT_PATH
    DOC "The MPC library.")
  find_library(MPC_LIBRARY mpc)

  if (MPC_LIBRARY AND MPC_INCLUDE_DIR)
    
    set(MPC_VERSION_CODE "
#include <iostream>
#include <gmp.h>
#include <mpfr.h>
#include <mpc.h>
int main(void)
{
  mpfr_t a;
  mpfr_prec_t prec = 256;
  mpfr_init2( a, prec );

  mpc_t b;
  mpc_init2( b, prec );

  gmp_randstate_t randState;
  gmp_randinit_default( randState );
  const long seed = 1024;
  gmp_randseed_ui( randState, seed );

  std::cout << mpc_get_version();
}")
    file(WRITE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx"
      "${MPC_VERSION_CODE}\n")
    
    if(NOT MPC_FIND_QUIETLY)
      message(STATUS "Performing Test MPC_VERSION_COMPILES")
    endif()

    set(__include_dirs
      "${MPC_INCLUDE_DIR}" "${MPFR_INCLUDE_DIR}" "${GMP_INCLUDE_DIR}")
    try_run(MPC_VERSION_RUNS MPC_VERSION_COMPILES
      ${CMAKE_BINARY_DIR}
      ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx
      LINK_LIBRARIES "${MPC_LIBRARY}" "${MPFR_LIBRARY}" "${GMP_LIBRARY}"
      CMAKE_FLAGS
      -DCOMPILE_DEFINITIONS=-DMPC_VERSION_COMPILES
      "-DINCLUDE_DIRECTORIES=${__include_dirs}"
      -DCMAKE_SKIP_RPATH:BOOL=${CMAKE_SKIP_RPATH}
      COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
      RUN_OUTPUT_VARIABLE RUN_OUTPUT)

    if (NOT MPC_VERSION_RUNS STREQUAL "FAILED_TO_RUN")
      if (RUN_OUTPUT VERSION_LESS MPC_FIND_VERSION)
        set(MPC_VERSION_OK FALSE)
      else ()
        set(MPC_VERSION_FOUND "${RUN_OUTPUT}")
        set(MPC_VERSION_OK TRUE)
      endif ()
    else ()
      
      message(WARNING "Found libmpfr but could compile with it.")

    endif (NOT MPC_VERSION_RUNS STREQUAL "FAILED_TO_RUN")
  endif (MPC_LIBRARY AND MPC_INCLUDE_DIR)
endif (MPFR_FOUND)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPC DEFAULT_MSG
  MPC_VERSION_FOUND MPC_LIBRARY MPC_INCLUDE_DIR MPC_VERSION_OK)

if (MPC_FOUND)
  if (NOT TARGET EP::mpc)
    add_library(EP::mpc INTERFACE IMPORTED)
    
    set_property(TARGET EP::mpc
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPC_INCLUDE_DIR}")
    set_property(TARGET EP::mpc
      PROPERTY INTERFACE_LINK_LIBRARIES "${MPC_LIBRARY}")
  endif ()
  
  set(MPC_LIBRARIES EP::mpc ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
  mark_as_advanced(MPC_LIBRARY MPC_INCLUDE_DIR)
endif ()
