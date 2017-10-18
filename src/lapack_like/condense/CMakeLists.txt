set_full_path(THIS_DIR_SOURCES
  Bidiag.cpp
  HermitianTridiag.cpp
  Hessenberg.cpp
  )

add_subdirectory(Bidiag)
add_subdirectory(HermitianTridiag)
add_subdirectory(Hessenberg)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
