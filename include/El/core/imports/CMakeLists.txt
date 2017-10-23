# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  blas.hpp
  choice.hpp
  flame.hpp
  lapack.hpp
  mkl.hpp
  mpfr.hpp
  mpi.hpp
  mpi_choice.hpp
  omp.hpp
  openblas.hpp
  pmrrr.hpp
  qd.hpp
  qt5.hpp
  scalapack.hpp
  suite_sparse.hpp
  valgrind.hpp
  )

# Add the subdirectories
add_subdirectory(scalapack)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
