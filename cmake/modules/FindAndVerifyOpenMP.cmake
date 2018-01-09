find_package(OpenMP)

if (NOT OpenMP_FOUND AND APPLE)

  find_library(_OpenMP_LIBRARY
    NAMES omp gomp iomp5md
    HINTS ${OpenMP_DIR} $ENV{OpenMP_DIR}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
    DOC "The libomp library.")

  if (${_OpenMP_LIBRARY} MATCHES "libomp*")
    set(OpenMP_omp_LIBRARY ${_OpenMP_LIBRARY}
      CACHE PATH "The OpenMP omp library.")
    set(OpenMP_C_LIB_NAMES libomp)
    set(OpenMP_CXX_LIB_NAMES libomp)
  elseif (${_OpenMP_LIBRARY} MATCHES "libgomp*")
    set(OpenMP_gomp_LIBRARY ${_OpenMP_LIBRARY}
      CACHE PATH "The OpenMP gomp library.")
    set(OpenMP_C_LIB_NAMES libgomp)
    set(OpenMP_CXX_LIB_NAMES libgomp)
  elseif (${_OpenMP_LIBRARY} MATCHES "libiomp5md*")
    set(OpenMP_libiomp5md_LIBRARY ${_OpenMP_LIBRARY}
      CACHE PATH "The OpenMP iomp5md library.")
    set(OpenMP_C_LIB_NAMES libiomp5md)
    set(OpenMP_CXX_LIB_NAMES libiomp5md)
  endif ()

  # Let's try this again
  find_package(OpenMP REQUIRED)

endif (NOT OpenMP_FOUND AND APPLE)
