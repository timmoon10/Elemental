# Exports the following variables
#
#   NVML_FOUND
#   NVML_INCLUDE_PATH
#   NVML_LIBRARIES
#
# Also adds the following imported target:
#
#   cuda::nvml
#

find_path(NVML_INCLUDE_PATH nvml.h
  HINTS ${NVML_DIR} $ENV{NVML_DIR} ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "The NVML header directory."
  )
find_path(NVML_INCLUDE_PATH nvml.h)

find_library(NVML_LIBRARY nvidia-ml
  HINTS ${NVML_DIR} $ENV{NVML_DIR} ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES lib64 lib lib64/stubs lib/stubs
  NO_DEFAULT_PATH
  DOC "The NVML library.")
find_library(NVML_LIBRARY nvidia-ml)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVML
  DEFAULT_MSG
  NVML_LIBRARY NVML_INCLUDE_PATH)

# Setup the imported target
if (NOT TARGET cuda::nvml)
  add_library(cuda::nvml INTERFACE IMPORTED)
endif (NOT TARGET cuda::nvml)

# Set the include directories for the target
set_property(TARGET cuda::nvml APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${NVML_INCLUDE_PATH})

set_property(TARGET cuda::nvml APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES ${NVML_LIBRARY})

#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE NVML_INCLUDE_DIRS)

# Set the libraries
set(NVML_LIBRARIES cuda::nvml)
