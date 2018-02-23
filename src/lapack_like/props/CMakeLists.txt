# Add the source files for this directory
#set_full_path(THIS_DIR_SOURCES
#  Condition.cpp
#  Determinant.cpp
#  Inertia.cpp
#  Norm.cpp
#  Trace.cpp
#  )

# Add the subdirectories
#add_subdirectory(Condition)
#add_subdirectory(Determinant)
add_subdirectory(Norm)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
