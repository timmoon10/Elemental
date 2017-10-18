set_full_path(THIS_DIR_SOURCES
  GLM.cpp
  LeastSquares.cpp
  LSE.cpp
  Ridge.cpp
  Tikhonov.cpp
  )

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
