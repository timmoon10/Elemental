set_full_path(THIS_DIR_HEADERS
  decl.hpp
  impl.hpp
  )

add_subdirectory(Complex)

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
