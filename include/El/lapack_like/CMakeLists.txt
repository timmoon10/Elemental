set_full_path(THIS_DIR_HEADERS
  condense.hpp
  CReflect.hpp
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

add_subdirectory(condense)
add_subdirectory(euclidean_min)
add_subdirectory(factor)
add_subdirectory(props)
add_subdirectory(solve)
add_subdirectory(spectral)

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
