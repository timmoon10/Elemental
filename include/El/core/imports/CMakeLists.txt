set_full_path(THIS_DIR_HEADERS
  blas.hpp
  choice.hpp
  flame.hpp
  lapack.hpp
  mkl.hpp
  mpfr.hpp
  mpi_choice.hpp
  mpi.hpp
  omp.hpp
  openblas.hpp
  pmrrr.hpp
  qd.hpp
  qt5.hpp
  scalapack.hpp
  suite_sparse.hpp
  valgrind.hpp
  )

add_subdirectory(scalapack)

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
