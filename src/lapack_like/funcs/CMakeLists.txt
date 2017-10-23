# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  HermitianFunction.cpp
  Pseudoinverse.cpp
  Sign.cpp
  SquareRoot.cpp
  )

# Add the subdirectories
add_subdirectory(Inverse)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
