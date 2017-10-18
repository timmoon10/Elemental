set_full_path(THIS_DIR_SOURCES
  ApplyReflector.hpp
  ApplyReflectors.hpp
  ComputeReflectors.hpp
  Dist.hpp
  IntroduceBulge.hpp
  VigilantDeflation.hpp
  )

add_subdirectory(Dist)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
