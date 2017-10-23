# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  DoubleShift.hpp
  SingleShift.hpp
  )

# Add the subdirectories
add_subdirectory(DoubleShift)
add_subdirectory(SingleShift)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
