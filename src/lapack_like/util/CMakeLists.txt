# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Bisect.cpp
  Median.cpp
  NaturalBisect.cpp
  Sort.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
