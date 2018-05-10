# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  decl.hpp
  impl.hpp
  impl_cpu.hpp
  impl_gpu.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
