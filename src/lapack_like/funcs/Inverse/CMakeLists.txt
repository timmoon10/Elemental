# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  General.cpp
  HPD.cpp
  Hermitian.cpp
  Symmetric.cpp
  Triangular.cpp
  )

# Add the subdirectories
add_subdirectory(General)
add_subdirectory(HPD)
add_subdirectory(Triangular)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
