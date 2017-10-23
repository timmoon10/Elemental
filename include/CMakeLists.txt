# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  El.hpp
  El-lite.hpp
  )

# Add the subdirectories
add_subdirectory(El)

# Propagate the files up the tree
set(HYDROGEN_HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
