# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  level1.hpp
  level2.hpp
  level3.hpp
  )

# Add the subdirectories
add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
