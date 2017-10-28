# Sets the following variables:
#
#   QD_FOUND
#   QD_INCLUDE_DIR -- Location of qd/dd_real.h and qd/qd_real.h
#   QD_LIBRARIES -- QD library

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
  
  check_cxx_source_compiles("${QD_CODE}" QD_WORKS)
  unset(CMAKE_REQUIRED_LIBRARIES)
  unset(CMAKE_REQUIRED_INCLUDES)

  if (NOT QD_WORKS)
    
    message(WARNING "Found QD but could not compile with it")

  endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(QD DEFAULT_MSG
  QD_LIBRARY QD_INCLUDE_DIR QD_WORKS)

if (QD_FOUND)
  if (NOT TARGET EP::qd)
    add_library(EP::qd INTERFACE IMPORTED)
    
    set_property(TARGET EP::qd
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${QD_INCLUDE_DIR}")
    set_property(TARGET EP::qd
      PROPERTY INTERFACE_LINK_LIBRARIES "${QD_LIBRARY}")
  endif ()
  
  set(QD_LIBRARIES EP::qd)
  mark_as_advanced(QD_INCLUDE_DIR QD_LIBRARY)
endif (QD_FOUND)
