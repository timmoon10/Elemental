# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Abstract.cpp
  Block.cpp
  Element.cpp
  )

# Add the subdirectories
add_subdirectory(Block)
add_subdirectory(Element)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
