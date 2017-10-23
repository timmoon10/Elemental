# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Full.hpp
  Local.hpp
  Mod.hpp
  Panel.hpp
  SolveAfter.hpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
