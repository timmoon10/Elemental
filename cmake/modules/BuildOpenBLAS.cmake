message(FATAL_ERROR "Uhhh please just build this yourself...\n"
  "That is, this is a work in progress.")

include (ExternalProject)

# If not using the Makefile generator for CMake, using
# CMAKE_MAKE_PROGRAM probably won't work here (in particular, ninja
# cannot process Makefiles). So we go looking for plain ol' "make"
# instead.
find_program(GNU_MAKE_PROGRAM make)

# Where to go looking
if (NOT DEFINED OpenBLAS_URL)
  set(OpenBLAS_URL "https://github.com/xianyi/OpenBLAS.git")
endif()

# The git tag
if (NOT DEFINED OpenBLAS_TAG)
  set(OpenBLAS_TAG "v0.2.15")
endif()

message(STATUS "Will pull OpenBLAS (tag ${OpenBLAS_TAG}) from ${OpenBLAS_URL}")

if(${PROJECT_NAME}_USE_64BIT_BLAS_INTS)
  set(OpenBLAS_SUFFIX 64)
else()
  set(OpenBLAS_SUFFIX)
endif()

set(OpenBLAS_SOURCE_DIR "${PROJECT_BINARY_DIR}/download/OpenBLAS/source")
set(OpenBLAS_BINARY_DIR "${PROJECT_BINARY_DIR}/download/OpenBLAS/build")

if(APPLE)
  if(NOT OpenBLAS_ARCH_COMMAND)
    # This is a hack but is a good default for modern Mac's
    set(OpenBLAS_ARCH_COMMAND TARGET=SANDYBRIDGE)
  endif()
else()
  if(NOT OpenBLAS_ARCH_COMMAND)
    set(OpenBLAS_ARCH_COMMAND)
  endif()
endif()

if(NOT OpenBLAS_THREAD_COMMAND)
  if(EL_HYBRID)
    set(OpenBLAS_THREAD_COMMAND USE_OPENMP=1)
  else()
    set(OpenBLAS_THREAD_COMMAND USE_THREAD=0)
  endif()
endif()

if(EL_USE_64BIT_BLAS_INTS)
  set(OpenBLAS_INTERFACE_COMMAND INTERFACE64=1 SYMBOLSUFFIX=64)
else()
  set(OpenBLAS_INTERFACE_COMMAND INTERFACE64=0)
endif()

if (NOT OpenBLAS_INSTALL_PREFIX)
  set(OpenBLAS_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

ExternalProject_Add(project_openblas
  PREFIX ${OpenBLAS_INSTALL_PREFIX}
  GIT_REPOSITORY ${OpenBLAS_URL}
  GIT_TAG ${OpenBLAS_TAG}
  STAMP_DIR ${OpenBLAS_BINARY_DIR}/stamp
  BUILD_IN_SOURCE 1
  SOURCE_DIR ${OpenBLAS_SOURCE_DIR}
  TMP_DIR    ${OpenBLAS_BINARY_DIR}/tmp
  INSTALL_DIR ${OpenBLAS_INSTALL_PREFIX}
  CONFIGURE_COMMAND ""
  UPDATE_COMMAND "" 
  BUILD_COMMAND ${GNU_MAKE_PROGRAM} -j${DEPENDENCY_MAKE_DASH_J}
    CC=${DEPENDENCY_C_COMPILER}
    FC=${DEPENDENCY_Fortran_COMPILER}
    ${OpenBLAS_THREAD_COMMAND}
    ${OpenBLAS_ARCH_COMMAND}
    ${OpenBLAS_INTERFACE_COMMAND}
    libs netlib shared
  INSTALL_COMMAND ${GNU_MAKE_PROGRAM} install PREFIX=<INSTALL_DIR>
  )
