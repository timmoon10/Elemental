include(CheckCXXSourceCompiles)

# Check for quad-precision support
# ================================
if(${PROJECT_NAME}_ENABLE_QUAD)
  
  find_path(QUADMATH_INCLUDE_DIR quadmath.h
    HINTS ${QUADMATH_DIR} $ENV{QUADMATH_DIR}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
    DOC "Directory with quadmath.h header.")
  find_path(QUADMATH_INCLUDE_DIRS quadmath.h)

  find_library(QUADMATH_LIBRARY quadmath
    HINTS ${QUADMATH_DIR} $ENV{QUADMATH_DIR}
    PATH_SUFFIXES lib64 lib
    NO_DEFAULT_PATH
    DOC "The QuadMath library.")
  find_library(QUADMATH_LIBRARY quadmath)

  if(QUADMATH_LIBRARY AND QUADMATH_INCLUDE_DIR)

    # Check that we can compile with quadmath library
    set(CMAKE_REQUIRED_INCLUDES "${QUADMATH_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES "${QUADMATH_LIBRARY}")
    set(QUADMATH_CODE
      "#include <complex>
       #include <iostream>
       #include <quadmath.h>
       int main( int argc, char* argv[] )
       {
           __float128 a = 2.0q;

           char aStr[128];
           quadmath_snprintf( aStr, sizeof(aStr), \"%Q\", a );
           std::cout << aStr << std::endl;

           __complex128 y;
           std::complex<__float128> z;

           return 0;    
       }")
    check_cxx_source_compiles("${QUADMATH_CODE}" EL_HAVE_QUADMATH)
    if(EL_HAVE_QUADMATH)
      if (NOT TARGET EP::quadmath)
        add_library(EP::quadmath INTERFACE IMPORTED)

        set_property(TARGET EP::quadmath
          PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${QUADMATH_INCLUDE_DIR}")
        set_property(TARGET EP::quadmath
          PROPERTY INTERFACE_LINK_LIBRARIES "${QUADLIB_LIBRARY}")
      endif ()
      list(APPEND EXTENDED_PRECISION_LIBRARIES EP::quadmath)

      set(EL_HAVE_QUAD TRUE)
    else()
      message(WARNING "Found libquadmath but could not use it in C++.")
    endif()
    unset(CMAKE_REQUIRED_INCLUDES)
    unset(CMAKE_REQUIRED_LIBRARIES)

  else()
    message(WARNING "libquadmath requested but not found. Disabling.")
    set(EL_HAVE_QUAD FALSE)
    unset(QUADMATH_INCLUDE_DIR)
    unset(QUADMATH_LIBRARY)
  endif()
endif()

# Check for QD
# ============
if (EL_ENABLE_QD)
  
  find_path(QD_INCLUDE_DIR NAMES qd/dd_real.h qd/qd_real.h
    HINTS ${QD_DIR} $ENV{QD_DIR}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
    DOC "Directory with qd.h header.")
  find_path(QD_INCLUDE_DIRS NAMES qd/dd_real.h qd/qd_real.h)

  find_library(QD_LIBRARY qd
    HINTS ${QD_DIR} $ENV{QD_DIR}
    PATH_SUFFIXES lib64 lib
    NO_DEFAULT_PATH
    DOC "The Qd library.")
  find_library(QD_LIBRARY qd)


  if (QD_LIBRARY AND QD_INCLUDE_DIR)
    set(CMAKE_REQUIRED_LIBRARIES ${QD_LIBRARY})
    set(CMAKE_REQUIRED_INCLUDES ${QD_INCLUDE_DIR})
    set(QD_CODE
      "#include <iostream>
       #include <qd/qd_real.h>
       int main( int argc, char* argv[] )
       {
           double a1=1., a2=2., b1=3., b2=4.;
           dd_real a(a1,a2), b(b1,b2);
           dd_real c = a*b;
           std::cout << \"c=\" << c << std::endl;
           qd_real d(a1,a2,b1,b2);
           std::cout << \"d=\" << d << std::endl;
       }")
    check_cxx_source_compiles("${QD_CODE}" EL_HAVE_QD)
    if (EL_HAVE_QD)
      if (NOT TARGET EP::qd)
        add_library(EP::qd INTERFACE IMPORTED)
        
        set_property(TARGET EP::qd
          PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${QD_INCLUDE_DIR}")
        set_property(TARGET EP::qd
          PROPERTY INTERFACE_LINK_LIBRARIES "${QUADLIB_LIBRARY}")
      endif ()
      list(APPEND EXTENDED_PRECISION_LIBRARIES EP::qd)

    else ()
      message(WARNING "Found QD but could not successfully compile with it")
    endif ()
    unset(CMAKE_REQUIRED_LIBRARIES)
    unset(CMAKE_REQUIRED_INCLUDES)
  else ()
    message(WARNING "QD requested but not found. Disabling.")
    unset(QD_LIBRARY)
    unset(QD_INCLUDE_DIR)
  endif()
endif()

# Check for GMP, MPFR, *and* MPC support
# ======================================
if (EL_ENABLE_MPFR)
  if (NOT EL_HAVE_MPI_LONG_LONG)
    message("Disabling MPFR since MPI_LONG_LONG was not detected")
    set(EL_ENABLE_MPFR OFF)
  else ()
    
    find_package(GMP 6.0.0)
    if(GMP_VERSION_OK)
      set(CMAKE_REQUIRED_LIBRARIES ${GMP_LIBRARIES})
      set(CMAKE_REQUIRED_INCLUDES ${GMP_INCLUDES})
      set(GMP_CODE
        "#include <gmp.h>
         int main( int argc, char* argv[] )
         {
           gmp_randstate_t randState;
           gmp_randinit_default( randState );
           const long seed = 1024;
           gmp_randseed_ui( randState, seed );
           return 0;
         }")
      check_cxx_source_compiles("${GMP_CODE}" EL_HAVE_GMP)
      if(NOT EL_HAVE_GMP)
        message(WARNING "Found GMP but could not successfully compile with it")
      endif()
      unset(CMAKE_REQUIRED_LIBRARIES)
      unset(CMAKE_REQUIRED_INCLUDES)
    endif()
    
    if(EL_HAVE_GMP)
      # TODO: See if this requirement could be lowered
      find_package(MPFR 3.1.0)
      if(MPFR_FOUND)
        set(CMAKE_REQUIRED_LIBRARIES ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
        set(CMAKE_REQUIRED_INCLUDES ${MPFR_INCLUDES} ${GMP_INCLUDES})
        set(MPFR_CODE
          "#include <mpfr.h>
           int main( int argc, char* argv[] )
           {
             mpfr_t a;
             mpfr_prec_t prec = 256;
             mpfr_init2( a, prec );
  
             /* Also test that GMP links */
             gmp_randstate_t randState;
             gmp_randinit_default( randState );
             const long seed = 1024;
             gmp_randseed_ui( randState, seed );
             
             return 0;
           }")
        check_cxx_source_compiles("${MPFR_CODE}" EL_HAVE_MPFR)
        if(NOT EL_HAVE_MPFR)
          message(WARNING "Found MPFR but could not successfully compile with it")
        endif()
        unset(CMAKE_REQUIRED_LIBRARIES)
        unset(CMAKE_REQUIRED_INCLUDES)
      endif()
    endif()

    if(EL_HAVE_GMP AND EL_HAVE_MPFR)
      find_package(MPC 1.0.0)
      if(MPC_FOUND) 
        set(CMAKE_REQUIRED_LIBRARIES
          ${MPC_LIBRARIES} ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
        set(CMAKE_REQUIRED_INCLUDES
          ${MPC_INCLUDES} ${MPFR_INCLUDES} ${GMP_INCLUDES})
        set(MPC_CODE
          "#include <mpc.h>
           int main( int argc, char* argv[] )
           {
             mpc_t a;
             mpfr_prec_t prec = 256;
             mpc_init2( a, prec );
              
             /* Also test that GMP links */
             gmp_randstate_t randState;
             gmp_randinit_default( randState );
             const long seed = 1024;
             gmp_randseed_ui( randState, seed );
              
             return 0;
           }")
        check_cxx_source_compiles("${MPC_CODE}" EL_HAVE_MPC)
        if(EL_HAVE_MPC)
          set(EL_HAVE_MPC TRUE) # Switch from '1' to 'TRUE' for Make
          list(APPEND MATH_LIBS
            ${MPC_LIBRARIES} ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
          list(APPEND MATH_LIBS_AT_CONFIG
            ${MPC_LIBRARIES} ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
          message(STATUS "Including ${GMP_INCLUDES}, ${MPFR_INCLUDES}, and ${MPC_INCLUDES} to add support for GMP, MPFR, and MPC")
          include_directories(${GMP_INCLUDES} ${MPFR_INCLUDES} ${MPC_INCLUDES})
        else()
          message(WARNING "Found MPC but could not successfully compile with it")
        endif()
        unset(CMAKE_REQUIRED_LIBRARIES)
        unset(CMAKE_REQUIRED_INCLUDES)
      endif()
    else()
      unset(GMP_LIBRARIES)
      unset(MPFR_LIBRARIES)
      unset(MPC_LIBRARIES)
      unset(GMP_INCLUDES)
      unset(MPFR_INCLUDES)
      unset(MPC_INCLUDES)
    endif()
  endif()
