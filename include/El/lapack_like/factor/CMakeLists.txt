set_full_path(THIS_DIR_HEADERS
  CReflect.hpp
  )

add_subdirectory(ldl)
add_subdirectory(qr)

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
