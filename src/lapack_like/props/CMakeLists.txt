set_full_path(THIS_DIR_SOURCES
  Condition.cpp
  Determinant.cpp
  Inertia.cpp
  Norm.cpp
  Trace.cpp
  )

add_subdirectory(Condition)
add_subdirectory(Determinant)
add_subdirectory(Norm)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
