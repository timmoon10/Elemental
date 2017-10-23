# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  blas_like.hpp
  control.hpp
  core.hpp
  io.hpp
  lapack_like.hpp
  matrices.hpp
  )

# Add the subdirectories
add_subdirectory(blas_like)
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(lapack_like)
add_subdirectory(macros)
add_subdirectory(matrices)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
