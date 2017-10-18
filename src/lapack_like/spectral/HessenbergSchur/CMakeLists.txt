set_full_path(THIS_DIR_SOURCES
  AED.hpp
  MultiBulge.hpp
  Simple.hpp
  )

add_subdirectory(AED)
add_subdirectory(MultiBulge)
add_subdirectory(Simple)
add_subdirectory(Util)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
