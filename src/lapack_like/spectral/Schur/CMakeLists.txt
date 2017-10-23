# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  CheckReal.hpp
  Condense.hpp
  InverseFreeSDC.hpp
  QuasiTriangEig.hpp
  RealToComplex.hpp
  SDC.hpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
