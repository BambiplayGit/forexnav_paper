#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "highs::highs" for configuration "Release"
set_property(TARGET highs::highs APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(highs::highs PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhighs.so.1.13.0"
  IMPORTED_SONAME_RELEASE "libhighs.so.1"
  )

list(APPEND _IMPORT_CHECK_TARGETS highs::highs )
list(APPEND _IMPORT_CHECK_FILES_FOR_highs::highs "${_IMPORT_PREFIX}/lib/libhighs.so.1.13.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
