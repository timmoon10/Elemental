set_full_path(THIS_DIR_SOURCES
  ComputeShifts.hpp
  PairShifts.hpp
  RedundantlyHandleWindow.hpp
  Sweep.hpp
  Transform.hpp
  TwoByTwo.hpp
  )

add_subdirectory(Sweep)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
