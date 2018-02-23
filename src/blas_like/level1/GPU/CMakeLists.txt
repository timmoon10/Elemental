set_full_path(THIS_DIR_SOURCES
  AllReduce.cu
  BasicInterleaveMatrix.cu
  Fill.cu
  Hadamard.cu
  Scale.cu
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
