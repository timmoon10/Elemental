# Add the subdirectories
add_subdirectory(blas_like)
add_subdirectory(core)
add_subdirectory(lapack_like)

foreach (src_file ${SOURCES})

  get_filename_component(__test_name "${src_file}" NAME_WE)

  # Create the executable
  add_executable("${__test_name}" ${src_file})
  target_link_libraries("${__test_name}" PRIVATE Hydrogen)
  
  # Create test
  add_test(NAME "${__test_name}.test"
    COMMAND "${__test_name}")

endforeach ()
