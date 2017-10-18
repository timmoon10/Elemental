set_full_path(THIS_DIR_SOURCES
  General.cpp
  Hermitian.cpp
  HPD.cpp
  Symmetric.cpp
  Triangular.cpp
  )

add_subdirectory(General)
add_subdirectory(HPD)
add_subdirectory(Triangular)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
