set_full_path(THIS_DIR_SOURCES
  HermitianFunction.cpp
  Pseudoinverse.cpp
  Sign.cpp
  SquareRoot.cpp
  )

add_subdirectory(Inverse)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
