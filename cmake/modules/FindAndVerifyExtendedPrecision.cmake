include(CheckCXXSourceCompiles)

# Check for quad-precision support
# ================================
if(${PROJECT_NAME}_ENABLE_QUAD)
  find_package(Quadmath QUIET)

  if (Quadmath_FOUND)
    set(EL_HAVE_QUAD TRUE)
    list(APPEND EXTENDED_PRECISION_LIBRARIES "${QUADMATH_LIBRARIES}")
  else()
    message(WARNING "libquadmath requested but not found. Disabling.")
    set(EL_HAVE_QUAD FALSE)
  endif()
endif()

# Check for QD
# ============
if (EL_ENABLE_QD)
  find_package(QD QUIET)

  if (QD_FOUND)
    set(EL_HAVE_QD TRUE)
    list(APPEND EXTENDED_PRECISION_LIBRARIES "${QD_LIBRARIES}")
  else ()
    message(WARNING "QD requested but not found. Disabling.")
    set(EL_HAVE_QD FALSE)
  endif()
endif()

# Check for GMP, MPFR, *and* MPC support
# ======================================
if (EL_ENABLE_MPC)
  if (NOT EL_HAVE_MPI_LONG_LONG)
    message("Disabling MPFR since MPI_LONG_LONG was not detected")
    set(EL_ENABLE_MPC OFF)
  else ()
    set(GMP_REQUIRED_VERSION "6.0.0")
    set(MPFR_REQUIRED_VERSION "3.1.0")
    set(MPC_REQUIRED_VERSION "1.0.0")

    find_package(MPC "${MPC_REQUIRED_VERSION}")
    if (MPC_FOUND)
      set(EL_HAVE_GMP TRUE)
      set(EL_HAVE_MPFR TRUE)
      set(EL_HAVE_MPC TRUE)
      list(APPEND EXTENDED_PRECISION_LIBRARIES "${MPC_LIBRARIES}")
    else ()
      message(WARNING "MPC requested but not found. Disabling.")
      set(EL_HAVE_GMP FALSE)
      set(EL_HAVE_MPFR FALSE)
      set(EL_HAVE_MPC FALSE)
    endif (MPC_FOUND)
  endif (NOT EL_HAVE_MPI_LONG_LONG)
endif (EL_ENABLE_MPC)
