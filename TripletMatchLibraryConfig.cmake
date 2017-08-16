###############################################################################
# Find TripletMatch
#
# This sets the following variables:
# TRIPLET_MATCH_FOUND - True if TRIPLET_MATCH was found.
# TRIPLET_MATCH_INCLUDE_DIRS - Directories containing the TRIPLET_MATCH include files.
# TRIPLET_MATCH_LIBRARY_DIRS - Directories containing the TRIPLET_MATCH library.
# TRIPLET_MATCH_LIBRARIES - TRIPLET_MATCH library files.

if(WIN32)
    find_path(TRIPLET_MATCH_INCLUDE_DIR triplet_match PATHS "/usr/include" "/usr/local/include" "/usr/x86_64-w64-mingw32/include" "$ENV{PROGRAMFILES}" NO_DEFAULT_PATHS)

    find_library(TRIPLET_MATCH_LIBRARY_PATH triplet_match PATHS "/usr/lib" "/usr/local/lib" "/usr/x86_64-w64-mingw32/lib" NO_DEFAULT_PATHS)

    if(EXISTS ${TRIPLET_MATCH_LIBRARY_PATH})
        get_filename_component(TRIPLET_MATCH_LIBRARY ${TRIPLET_MATCH_LIBRARY_PATH} NAME)
        find_path(TRIPLET_MATCH_LIBRARY_DIR ${TRIPLET_MATCH_LIBRARY} PATHS "/usr/lib" "/usr/local/lib" "/usr/x86_64-w64-mingw32/lib" NO_DEFAULT_PATHS)
    endif()
else(WIN32)
    find_path(TRIPLET_MATCH_INCLUDE_DIR triplet_match PATHS "/usr/include" "/usr/local/include" "$ENV{PROGRAMFILES}" NO_DEFAULT_PATHS)
    find_library(TRIPLET_MATCH_LIBRARY_PATH triplet_match PATHS "/usr/lib" "/usr/local/lib" NO_DEFAULT_PATHS)

    if(EXISTS ${TRIPLET_MATCH_LIBRARY_PATH})
        get_filename_component(TRIPLET_MATCH_LIBRARY ${TRIPLET_MATCH_LIBRARY_PATH} NAME)
        find_path(TRIPLET_MATCH_LIBRARY_DIR ${TRIPLET_MATCH_LIBRARY} PATHS "/usr/lib" "/usr/local/lib" NO_DEFAULT_PATHS)
    endif()
endif(WIN32)

set(TRIPLET_MATCH_INCLUDE_DIRS ${TRIPLET_MATCH_INCLUDE_DIR})
set(TRIPLET_MATCH_LIBRARY_DIRS ${TRIPLET_MATCH_LIBRARY_DIR})
set(TRIPLET_MATCH_LIBRARIES ${TRIPLET_MATCH_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TRIPLET_MATCH DEFAULT_MSG TRIPLET_MATCH_INCLUDE_DIR TRIPLET_MATCH_LIBRARY TRIPLET_MATCH_LIBRARY_DIR)

mark_as_advanced(TRIPLET_MATCH_INCLUDE_DIR)
mark_as_advanced(TRIPLET_MATCH_LIBRARY_DIR)
mark_as_advanced(TRIPLET_MATCH_LIBRARY)
mark_as_advanced(TRIPLET_MATCH_LIBRARY_PATH)
