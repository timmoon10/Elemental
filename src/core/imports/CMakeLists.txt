# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  blas.cpp
  flame.cpp
  lapack.cpp
  mkl.cpp
  mpfr.cpp
  mpi.cpp
  openblas.cpp
  pmrrr.cpp
  qd.cpp
  qt5.cpp
  scalapack.cpp
  )

if (HYDROGEN_HAVE_CUDA)
  set_full_path(CUDA_SOURCES cuda.cpp cublas.cpp)
  list(APPEND THIS_DIR_SOURCES ${CUDA_SOURCES})
endif()

# Add the subdirectories
add_subdirectory(blas)
add_subdirectory(lapack)
add_subdirectory(mpfr)
add_subdirectory(scalapack)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
