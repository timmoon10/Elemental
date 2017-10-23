# Add the subdirectories
add_subdirectory(numeric)
add_subdirectory(symbolic)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
