# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  BullsHead.cpp
  Forsythe.cpp
  Grcar.cpp
  OneTwoOne.cpp
  TrefethenEmbree.cpp
  TriW.cpp
  Triangle.cpp
  Whale.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
