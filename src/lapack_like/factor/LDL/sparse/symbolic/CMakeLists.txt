# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Analysis.cpp
  NaturalNestedDissection.cpp
  NestedDissection.cpp
  NodeInfo.cpp
  Separator.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
