# Sets the following variables:
#
#   GMP_FOUND
#   GMP_VERSION_OK
#   GMP_INCLUDE_DIR -- Location of gmp.h
#   GMP_LIBRARIES -- libgmp library

if (GMP_FIND_VERSION_COUNT EQUAL 0)
  if (GMP_REQUIRED_VERSION)
    set(GMP_FIND_VERSION "${GMP_REQUIRED_VERSION}")
  else ()
    set(GMP_FIND_VERSION "5.1.1")
  endif ()
endif ()

find_path(GMP_INCLUDE_DIR gmp.h
  HINTS ${GMP_DIR} $ENV{GMP_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with gmp.h header.")
find_path(GMP_INCLUDE_DIR gmp.h)

find_library(GMP_LIBRARY gmp
  HINTS ${GMP_DIR} $ENV{GMP_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The GMP library.")
find_library(GMP_LIBRARY gmp)

if (GMP_LIBRARY AND GMP_INCLUDE_DIR)
  
  set(GMP_VERSION_CODE "
#include <iostream>
#include \"gmp.h\"
int main(void)
{
  gmp_randstate_t randState;
  gmp_randinit_default( randState );
  const long seed = 1024;
  gmp_randseed_ui( randState, seed );

  std::cout << gmp_version;
}")
  file(WRITE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx"
    "${GMP_VERSION_CODE}\n")
  
  if(NOT GMP_FIND_QUIETLY)
    message(STATUS "Performing Test GMP_VERSION_COMPILES")
  endif()
  
  try_run(GMP_VERSION_RUNS GMP_VERSION_COMPILES
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx
    LINK_LIBRARIES "${GMP_LIBRARY}"
    CMAKE_FLAGS
    -DCOMPILE_DEFINITIONS=-DGMP_VERSION_COMPILES
    "-DINCLUDE_DIRECTORIES=${GMP_INCLUDE_DIR}"
    -DCMAKE_SKIP_RPATH:BOOL=${CMAKE_SKIP_RPATH}
    COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
    RUN_OUTPUT_VARIABLE RUN_OUTPUT)

  if (NOT GMP_VERSION_RUNS STREQUAL "FAILED_TO_RUN")
    if (RUN_OUTPUT VERSION_LESS GMP_FIND_VERSION)
      set(GMP_VERSION_OK FALSE)
    else ()
      set(GMP_VERSION_FOUND "${RUN_OUTPUT}")
      set(GMP_VERSION_OK TRUE)
    endif ()
  else ()
    
    message(WARNING "Found libgmp but could compile with it.")

  endif (NOT GMP_VERSION_RUNS STREQUAL "FAILED_TO_RUN")
endif (GMP_LIBRARY AND GMP_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP DEFAULT_MSG
  GMP_VERSION_FOUND GMP_LIBRARY GMP_INCLUDE_DIR GMP_VERSION_OK)

if (GMP_FOUND)
  if (NOT TARGET EP::gmp)
    add_library(EP::gmp INTERFACE IMPORTED)
    
    set_property(TARGET EP::gmp
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}")
    set_property(TARGET EP::gmp
      PROPERTY INTERFACE_LINK_LIBRARIES "${GMP_LIBRARY}")
  endif ()
  
  set(GMP_LIBRARIES EP::gmp)
  mark_as_advanced(GMP_LIBRARY GMP_INCLUDE_DIR)
endif (GMP_FOUND)
