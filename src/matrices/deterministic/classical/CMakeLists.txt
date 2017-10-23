# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Cauchy.cpp
  CauchyLike.cpp
  Diagonal.cpp
  Egorov.cpp
  Fiedler.cpp
  Fourier.cpp
  GCDMatrix.cpp
  Hankel.cpp
  Hilbert.cpp
  Identity.cpp
  Jordan.cpp
  Ones.cpp
  Toeplitz.cpp
  Walsh.cpp
  Zeros.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
