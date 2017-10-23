# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Local.hpp
  NNNN.hpp
  NNNT.hpp
  NNTN.hpp
  NNTT.hpp
  NTNN.hpp
  NTNT.hpp
  NTTN.hpp
  NTTT.hpp
  TNNN.hpp
  TNNT.hpp
  TNTN.hpp
  TNTT.hpp
  TTNN.hpp
  TTNT.hpp
  TTTN.hpp
  TTTT.hpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
