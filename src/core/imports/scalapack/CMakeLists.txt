# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  blacs.cpp
  pblas.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
