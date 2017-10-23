# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  CReflect.hpp
  HermitianEig.hpp
  Lanczos.hpp
  ProductLanczos.hpp
  SVD.hpp
  Schur.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
