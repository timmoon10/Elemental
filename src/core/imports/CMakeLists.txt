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

add_subdirectory(blas)
add_subdirectory(lapack)
add_subdirectory(mpfr)
add_subdirectory(scalapack)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
