set_full_path(THIS_DIR_SOURCES
  ApplyPacked.cpp
  ExpandPacked.cpp
  Householder.cpp
  Hyperbolic.cpp
  )

add_subdirectory(ApplyPacked)
add_subdirectory(ExpandPacked)
add_subdirectory(Householder)
add_subdirectory(Hyperbolic)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
