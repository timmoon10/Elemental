include(CheckCXXSourceCompiles)

if (NOT UPPER_PROJECT_NAME)
  string(TOUPPER "${PROJECT_NAME}" UPPER_PROJECT_NAME)
endif ()

# Check for quad-precision support
# ================================
if(${PROJECT_NAME}_ENABLE_QUADMATH)
  find_package(Quadmath)

  if (Quadmath_FOUND)
    set(${UPPER_PROJECT_NAME}_HAVE_QUADMATH TRUE)
    list(APPEND EXTENDED_PRECISION_LIBRARIES "${QUADMATH_LIBRARIES}")
  else()
    message(WARNING "libquadmath requested but not found. Disabling.")
    set(${UPPER_PROJECT_NAME}_HAVE_QUADMATH FALSE)
  endif()
endif()

# Check for QD
# ============
if (${PROJECT_NAME}_ENABLE_QD)
  find_package(QD)

  if (QD_FOUND)
    set(${UPPER_PROJECT_NAME}_HAVE_QD TRUE)
    list(APPEND EXTENDED_PRECISION_LIBRARIES "${QD_LIBRARIES}")
  else ()
    message(WARNING "QD requested but not found. Disabling.")
    set(${UPPER_PROJECT_NAME}_HAVE_QD FALSE)
  endif()
endif()

# Check for GMP, MPFR, *and* MPC support
# ======================================
if (${PROJECT_NAME}_ENABLE_MPC)
  if (NOT EL_HAVE_MPI_LONG_LONG)
    message("Disabling MPFR since MPI_LONG_LONG was not detected")
    set(${PROJECT_NAME}_ENABLE_MPC OFF)
  else ()
    set(GMP_REQUIRED_VERSION "6.0.0")
    set(MPFR_REQUIRED_VERSION "3.1.0")
    set(MPC_REQUIRED_VERSION "1.0.0")

    find_package(MPC "${MPC_REQUIRED_VERSION}")
    if (MPC_FOUND)
      set(${UPPER_PROJECT_NAME}_HAVE_GMP TRUE)
      set(${UPPER_PROJECT_NAME}_HAVE_MPFR TRUE)
      set(${UPPER_PROJECT_NAME}_HAVE_MPC TRUE)
      list(APPEND EXTENDED_PRECISION_LIBRARIES "${MPC_LIBRARIES}")
    else ()
      message(WARNING "MPC requested but not found. Disabling.")
      set(${UPPER_PROJECT_NAME}_HAVE_GMP FALSE)
      set(${UPPER_PROJECT_NAME}_HAVE_MPFR FALSE)
      set(${UPPER_PROJECT_NAME}_HAVE_MPC FALSE)
    endif (MPC_FOUND)
  endif (NOT EL_HAVE_MPI_LONG_LONG)
endif (${PROJECT_NAME}_ENABLE_MPC)

if (NOT TARGET EP::extended_precision)
  add_library(EP::extended_precision INTERFACE IMPORTED)

  set_property(TARGET EP::extended_precision
    PROPERTY INTERFACE_LINK_LIBRARIES ${EXTENDED_PRECISION_LIBRARIES})
endif (NOT TARGET EP::extended_precision)
