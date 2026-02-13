#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "altrocpp::common" for configuration "Release"
set_property(TARGET altrocpp::common APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(altrocpp::common PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcommon.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS altrocpp::common )
list(APPEND _IMPORT_CHECK_FILES_FOR_altrocpp::common "${_IMPORT_PREFIX}/lib/libcommon.a" )

# Import target "altrocpp::threadpool" for configuration "Release"
set_property(TARGET altrocpp::threadpool APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(altrocpp::threadpool PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libthreadpool.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS altrocpp::threadpool )
list(APPEND _IMPORT_CHECK_FILES_FOR_altrocpp::threadpool "${_IMPORT_PREFIX}/lib/libthreadpool.a" )

# Import target "altrocpp::ilqr" for configuration "Release"
set_property(TARGET altrocpp::ilqr APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(altrocpp::ilqr PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libilqr.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS altrocpp::ilqr )
list(APPEND _IMPORT_CHECK_FILES_FOR_altrocpp::ilqr "${_IMPORT_PREFIX}/lib/libilqr.a" )

# Import target "altrocpp::problem" for configuration "Release"
set_property(TARGET altrocpp::problem APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(altrocpp::problem PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libproblem.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS altrocpp::problem )
list(APPEND _IMPORT_CHECK_FILES_FOR_altrocpp::problem "${_IMPORT_PREFIX}/lib/libproblem.a" )

# Import target "altrocpp::utils" for configuration "Release"
set_property(TARGET altrocpp::utils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(altrocpp::utils PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libutils.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS altrocpp::utils )
list(APPEND _IMPORT_CHECK_FILES_FOR_altrocpp::utils "${_IMPORT_PREFIX}/lib/libutils.a" )

# Import target "altrocpp::constraints" for configuration "Release"
set_property(TARGET altrocpp::constraints APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(altrocpp::constraints PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libconstraints.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS altrocpp::constraints )
list(APPEND _IMPORT_CHECK_FILES_FOR_altrocpp::constraints "${_IMPORT_PREFIX}/lib/libconstraints.a" )

# Import target "altrocpp::augmented_lagrangian" for configuration "Release"
set_property(TARGET altrocpp::augmented_lagrangian APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(altrocpp::augmented_lagrangian PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaugmented_lagrangian.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS altrocpp::augmented_lagrangian )
list(APPEND _IMPORT_CHECK_FILES_FOR_altrocpp::augmented_lagrangian "${_IMPORT_PREFIX}/lib/libaugmented_lagrangian.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
