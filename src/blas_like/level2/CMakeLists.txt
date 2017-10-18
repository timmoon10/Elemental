set_full_path(THIS_DIR_SOURCES
  ApplyGivensSequence.cpp
  Gemv.cpp
  Ger.cpp
  Geru.cpp
  Hemv.cpp
  Her2.cpp
  Her.cpp
  QuasiTrsv.cpp
  Symv.cpp
  Syr2.cpp
  Syr.cpp
  Trmv.cpp
  Trr2.cpp
  Trr.cpp
  Trsv.cpp
  )

add_subdirectory(Gemv)
add_subdirectory(QuasiTrsv)
add_subdirectory(Symv)
add_subdirectory(Trsv)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
