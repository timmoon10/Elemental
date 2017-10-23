# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  ApplyQ.hpp
  FormQ.hpp
  LowerBlocked.hpp
  LowerPanel.hpp
  LowerUnblocked.hpp
  UpperBlocked.hpp
  UpperPanel.hpp
  UpperUnblocked.hpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
