# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Analytic.hpp
  HagerHigham.hpp
  IRA.hpp
  IRL.hpp
  Lanczos.hpp
  Power.hpp
  Util.hpp
  )

# Add the subdirectories
add_subdirectory(Util)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
