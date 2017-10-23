# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  LLHB.hpp
  LLHF.hpp
  LLVB.hpp
  LLVF.hpp
  LUHB.hpp
  LUHF.hpp
  LUVB.hpp
  LUVF.hpp
  RLHB.hpp
  RLHF.hpp
  RLVB.hpp
  RLVF.hpp
  RUHB.hpp
  RUHF.hpp
  RUVB.hpp
  RUVF.hpp
  Util.hpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
