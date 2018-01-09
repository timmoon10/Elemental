find_package(OpenMP)

if (NOT OpenMP_FOUND AND APPLE)

  find_library(_OpenMP_LIBRARY
    NAMES omp gomp iomp5md
    HINTS ${OpenMP_DIR} $ENV{OpenMP_DIR}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
    DOC "The libomp library.")

  if (${_OpenMP_LIBRARY} MATCHES "libomp*")
    set(OpenMP_libomp_LIBRARY ${_OpenMP_LIBRARY}
      CACHE PATH "The OpenMP omp library.")
    foreach (lang IN ITEMS C CXX)
      set(OpenMP_${lang}_LIB_NAMES libomp)
      set(OpenMP_${lang}_FLAGS "-fopenmp=libomp")
    endforeach ()
  elseif (${_OpenMP_LIBRARY} MATCHES "libgomp*")
    set(OpenMP_libgomp_LIBRARY ${_OpenMP_LIBRARY}
      CACHE PATH "The OpenMP gomp library.")
    foreach (lang IN ITEMS C CXX)
      set(OpenMP_${lang}_LIB_NAMES libgomp)
      set(OpenMP_${lang}_FLAGS "-fopenmp")
    endforeach ()
  elseif (${_OpenMP_LIBRARY} MATCHES "libiomp5md*")
    set(OpenMP_libiomp5md_LIBRARY ${_OpenMP_LIBRARY}
      CACHE PATH "The OpenMP iomp5md library.")
    foreach (lang IN ITEMS C CXX)
      set(OpenMP_${lang}_LIB_NAMES libiomp5md)
      set(OpenMP_${lang}_FLAGS "-fopenmp=libiomp5")
    endforeach ()
  endif ()

  # Let's try this again
  find_package(OpenMP REQUIRED)

endif (NOT OpenMP_FOUND AND APPLE)
