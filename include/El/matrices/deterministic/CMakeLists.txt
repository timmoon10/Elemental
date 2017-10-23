# Add the subdirectories
add_subdirectory(classical)
add_subdirectory(lattice)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
