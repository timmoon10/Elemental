# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  CReflect.hpp
  DistMap.hpp
  DistMatrix.hpp
  DistPermutation.hpp
  Element.hpp
  FlamePart.hpp
  Grid.hpp
  Matrix.hpp
  Memory.hpp
  Permutation.hpp
  Proxy.hpp
  Serialize.hpp
  Timer.hpp
  View.hpp
  limits.hpp
  types.hpp
  )

# Add the subdirectories
add_subdirectory(DistMap)
add_subdirectory(DistMatrix)
add_subdirectory(Element)
add_subdirectory(FlamePart)
add_subdirectory(Matrix)
add_subdirectory(Memory)
add_subdirectory(View)
add_subdirectory(environment)
add_subdirectory(imports)
add_subdirectory(indexing)
add_subdirectory(random)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
