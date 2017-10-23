# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  CReflect.hpp
  ComplexDisplayWindow-premoc.hpp
  DisplayWidget.hpp
  DisplayWindow-premoc.hpp
  SpyWidget.hpp
  SpyWindow.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
