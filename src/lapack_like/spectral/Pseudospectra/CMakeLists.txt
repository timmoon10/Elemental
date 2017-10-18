set_full_path(THIS_DIR_SOURCES
  Analytic.hpp
  HagerHigham.hpp
  IRA.hpp
  IRL.hpp
  Lanczos.hpp
  Power.hpp
  Util.hpp
  )

add_subdirectory(Util)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
