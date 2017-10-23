# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  ApplyQ.hpp
  Cholesky.hpp
  Explicit.hpp
  Householder.hpp
  PanelHouseholder.hpp
  SolveAfter.hpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
