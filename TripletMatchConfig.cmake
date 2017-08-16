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
else(WIN32)
    find_path(TRIPLET_MATCH_INCLUDE_DIR triplet_match PATHS "/usr/include" "/usr/local/include" "$ENV{PROGRAMFILES}" NO_DEFAULT_PATHS)
endif(WIN32)

set(TRIPLET_MATCH_INCLUDE_DIRS ${TRIPLET_MATCH_INCLUDE_DIR})
set(TRIPLET_MATCH_LIBRARY_DIRS "")
set(TRIPLET_MATCH_LIBRARIES "")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TRIPLET_MATCH DEFAULT_MSG TRIPLET_MATCH_INCLUDE_DIR TRIPLET_MATCH_LIBRARY TRIPLET_MATCH_LIBRARY_DIR)

mark_as_advanced(TRIPLET_MATCH_INCLUDE_DIR)
mark_as_advanced(TRIPLET_MATCH_LIBRARY_DIR)
mark_as_advanced(TRIPLET_MATCH_LIBRARY)
mark_as_advanced(TRIPLET_MATCH_LIBRARY_PATH)
