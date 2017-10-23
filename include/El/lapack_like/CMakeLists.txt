# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  CReflect.hpp
  condense.hpp
  equilibrate.hpp
  euclidean_min.hpp
  factor.hpp
  funcs.hpp
  perm.hpp
  props.hpp
  reflect.hpp
  solve.hpp
  spectral.hpp
  util.hpp
  )

# Add the subdirectories
add_subdirectory(condense)
add_subdirectory(euclidean_min)
add_subdirectory(factor)
add_subdirectory(props)
add_subdirectory(solve)
add_subdirectory(spectral)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
