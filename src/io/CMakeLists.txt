set_full_path(THIS_DIR_SOURCES
  ColorMap.cpp
  ComplexDisplayWindow.cpp
  Display.cpp
  DisplayWidget.cpp
  DisplayWindow.cpp
  File.cpp
  Print.cpp
  Read.cpp
  Spy.cpp
  SpyWidget.cpp
  SpyWindow.cpp
  Write.cpp
  )

add_subdirectory(Read)
add_subdirectory(Write)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
