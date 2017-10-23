# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Ascii.hpp
  AsciiMatlab.hpp
  Binary.hpp
  BinaryFlat.hpp
  Image.hpp
  MatrixMarket.hpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
