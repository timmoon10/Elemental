# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  blocksizes.cpp
  )

# Add the subdirectories
add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
