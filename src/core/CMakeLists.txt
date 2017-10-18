set_full_path(THIS_DIR_SOURCES
  callStack.cpp
  DistGraph.cpp
  DistMap.cpp
  Element.cpp
  environment.cpp
  Graph.cpp
  Grid.cpp
  indent.cpp
  Instantiate.cpp
  logging.cpp
  mpi_register.cpp
  random.cpp
  Serialize.cpp
  Timer.cpp
  types.cpp
  )

add_subdirectory(DistMatrix)
add_subdirectory(FlamePart)
add_subdirectory(imports)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
