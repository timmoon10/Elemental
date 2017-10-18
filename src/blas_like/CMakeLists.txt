set_full_path(THIS_DIR_SOURCES
  blocksizes.cpp)

add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
