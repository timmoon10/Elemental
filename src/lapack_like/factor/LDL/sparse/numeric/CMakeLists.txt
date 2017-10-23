# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  ChangeFrontType.cpp
  DiagonalScale.cpp
  DiagonalSolve.cpp
  DistFront.cpp
  DistMatrixNode.cpp
  DistMultiVecNode.cpp
  DistSparseLDLFactorization.cpp
  Front.cpp
  FrontType.cpp
  MatrixNode.cpp
  SparseLDLFactorization.cpp
  Process.hpp
  ProcessFront.hpp
  )

# Add the subdirectories
add_subdirectory(LowerMultiply)
add_subdirectory(LowerSolve)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
