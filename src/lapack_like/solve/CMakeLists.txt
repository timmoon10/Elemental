# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  HPD.cpp
  Hermitian.cpp
  Linear.cpp
  MultiShiftHess.cpp
  SQSD.cpp
  Symmetric.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
