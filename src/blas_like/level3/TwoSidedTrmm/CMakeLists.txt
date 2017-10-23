# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  LVar1.hpp
  LVar2.hpp
  LVar4.hpp
  LVar5.hpp
  UVar1.hpp
  UVar2.hpp
  UVar4.hpp
  UVar5.hpp
  Unblocked.hpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
