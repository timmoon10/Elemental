set_full_path(THIS_DIR_HEADERS
  numeric.hpp
  symbolic.hpp
  )

add_subdirectory(symbolic)

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
