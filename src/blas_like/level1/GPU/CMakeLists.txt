set_full_path(THIS_DIR_SOURCES
  Fill.cu
  Hadamard.cu
  Scale.cu
  Transpose.cu
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
