set_full_path(THIS_DIR_HEADERS
  Abstract.hpp
  Block.hpp
  Element.hpp
  )

add_subdirectory(Block)
add_subdirectory(Element)

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
