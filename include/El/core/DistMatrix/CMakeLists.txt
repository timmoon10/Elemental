# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  Abstract.hpp
  Block.hpp
  Element.hpp
  )

# Add the subdirectories
add_subdirectory(Block)
add_subdirectory(Element)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
