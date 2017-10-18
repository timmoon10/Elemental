set_full_path(THIS_DIR_SOURCES
  Abstract.cpp
  Block.cpp
  Element.cpp
  )

add_subdirectory(Block)
add_subdirectory(Element)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
