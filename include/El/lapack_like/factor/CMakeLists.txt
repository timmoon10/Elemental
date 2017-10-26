# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  CReflect.hpp
  )

# Add the subdirectories
#add_subdirectory(ldl)
add_subdirectory(qr)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
