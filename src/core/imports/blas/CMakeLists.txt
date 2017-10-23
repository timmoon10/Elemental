# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Axpy.hpp
  Copy.hpp
  Dot.hpp
  Gemm.hpp
  Gemv.hpp
  Ger.hpp
  MaxInd.hpp
  Nrm.hpp
  Rot.hpp
  Scal.hpp
  Swap.hpp
  Symm.hpp
  Symv.hpp
  Syr.hpp
  Syr2.hpp
  Syr2k.hpp
  Syrk.hpp
  Trmm.hpp
  Trmv.hpp
  Trsm.hpp
  Trsv.hpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
