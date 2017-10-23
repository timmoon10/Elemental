# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  AED.hpp
  MultiBulge.hpp
  Simple.hpp
  )

# Add the subdirectories
add_subdirectory(AED)
add_subdirectory(MultiBulge)
add_subdirectory(Simple)
add_subdirectory(Util)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
