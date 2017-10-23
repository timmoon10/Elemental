# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Bernoulli.cpp
  Gaussian.cpp
  Rademacher.cpp
  ThreeValued.cpp
  Uniform.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
