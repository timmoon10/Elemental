# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  ApplyReflector.hpp
  ApplyReflectors.hpp
  ComputeReflectors.hpp
  Dist.hpp
  IntroduceBulge.hpp
  VigilantDeflation.hpp
  )

# Add the subdirectories
add_subdirectory(Dist)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
