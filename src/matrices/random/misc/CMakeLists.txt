set_full_path(THIS_DIR_SOURCES
  Haar.cpp
  HatanoNelson.cpp
  HermitianUniformSpectrum.cpp
  NormalUniformSpectrum.cpp
  UniformHelmholtzGreens.cpp
  Wigner.cpp
  )

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
