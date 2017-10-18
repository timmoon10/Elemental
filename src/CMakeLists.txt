set_full_path(THIS_DIR_SOURCES
  io.cpp
  )

add_subdirectory(blas_like)
add_subdirectory(control)
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(lapack_like)
add_subdirectory(matrices)

set(HYDROGEN_SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
