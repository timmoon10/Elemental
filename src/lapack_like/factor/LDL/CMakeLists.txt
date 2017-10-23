# Add the subdirectories
add_subdirectory(dense)
add_subdirectory(sparse)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
