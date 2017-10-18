set_full_path(THIS_DIR_HEADERS
  El.hpp
  El-lite.hpp
  )

add_subdirectory(El)

set(HYDROGEN_HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
