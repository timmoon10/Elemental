# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Inertia.hpp
  MultiplyAfter.hpp
  Pivoted.hpp
  SolveAfter.hpp
  Var3.hpp
  )

# Add the subdirectories
add_subdirectory(Pivoted)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
