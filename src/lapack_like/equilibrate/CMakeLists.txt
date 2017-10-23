# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  GeomEquil.cpp
  RuizEquil.cpp
  SymmetricDiagonalEquil.cpp
  SymmetricGeomEquil.cpp
  SymmetricRuizEquil.cpp
  Util.hpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
