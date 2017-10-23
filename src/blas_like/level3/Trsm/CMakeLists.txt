# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  LLN.hpp
  LLT.hpp
  LUN.hpp
  LUT.hpp
  RLN.hpp
  RLT.hpp
  RUN.hpp
  RUT.hpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
