set_full_path(THIS_DIR_SOURCES
  ColumnMinAbs.cpp
  ColumnNorms.cpp
  Copy.cpp
  GetSubgraph.cpp
  HilbertSchmidt.cpp
  Instantiate.cpp
  MaxAbs.cpp
  MaxAbsLoc.cpp
  Max.cpp
  MaxLoc.cpp
  MinAbsLoc.cpp
  Min.cpp
  MinLoc.cpp
  NormsFromScaledSquares.hpp
  RowMinAbs.cpp
  RowNorms.cpp
  Swap.cpp
  Symmetric2x2Inv.cpp
  Transform2x2.cpp
  )

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
