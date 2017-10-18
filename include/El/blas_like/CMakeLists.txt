set_full_path(THIS_DIR_HEADERS
  level1.hpp
  level2.hpp
  level3.hpp
  )

add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
