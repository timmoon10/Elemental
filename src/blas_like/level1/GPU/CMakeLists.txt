set_full_path(THIS_DIR_SOURCES
  Axpy.cu
  Copy.cu
  Fill.cu
  Hadamard.cu
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
