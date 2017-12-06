cmake_minimum_required(VERSION 3.8.2)
project(triplet_match)

set (triplet_match_VERSION_MAJOR 0)
set (triplet_match_VERSION_MINOR 1)

option(TRIPLET_MATCH_PRECOMPILE_HEADERS "Precompile implementation for common point types" ON)
configure_file (
  "${PROJECT_SOURCE_DIR}/config.hpp.in"
  "${PROJECT_SOURCE_DIR}/include/config"
)

include_directories("${PROJECT_SOURCE_DIR}")
#include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")

install (DIRECTORY include/ DESTINATION include/triplet_match)

if (TRIPLET_MATCH_PRECOMPILE_HEADERS)
    find_package(PCL COMPONENTS common io search kdtree)
    find_package(VoxelScore)
    find_package(Eigen)
    find_package(OpenCL)
    find_package(fmt)
    file (GLOB_RECURSE obj RELATIVE "${PROJECT_SOURCE_DIR}" "src/*.cpp")
    if (EIGEN_FOUND AND PCL_FOUND AND VOXEL_SCORE_FOUND AND OpenCL_FOUND)
        if(CMAKE_COMPILER_IS_GNUCXX)
            add_definitions(-fPIC)
            add_definitions(-O3)
            add_definitions(-g)
            add_definitions(-std=c++17)
            add_definitions(-Wall)
            add_definitions(-Wno-int-in-bool-context)
            add_definitions(-Wno-ignored-attributes)
        endif()
        include_directories(${EIGEN_INCLUDE_DIRS})
        include_directories(${VOXEL_SCORE_INCLUDE_DIRS})
        include_directories(${PCL_INCLUDE_DIRS})
        include_directories(${OpenCL_INCLUDE_DIRS})

        add_library(triplet_match SHARED ${obj})
        set_property(TARGET triplet_match PROPERTY CXX_STANDARD 17)
        target_link_libraries(triplet_match ${PCL_LIBRARIES} ${VOXEL_SCORE_LIBRARIES} ${OpenCL_LIBRARIES} fmt::fmt "dl")

        #add_executable(demo "apps/demo.cpp")
        #set_property(TARGET demo PROPERTY CXX_STANDARD 17)
        #target_link_libraries(demo ${PCL_LIBRARIES} ${VOXEL_SCORE_LIBRARIES} ${OpenCL_LIBRARIES} fmt::fmt triplet_match "dl")

        install (TARGETS triplet_match DESTINATION lib)
    endif()
endif(TRIPLET_MATCH_PRECOMPILE_HEADERS)

if (NOT WIN32)
    if (TRIPLET_MATCH_PRECOMPILE_HEADERS)
        file (GLOB find_modules RELATIVE "${PROJECT_SOURCE_DIR}" "TripletMatchLibraryConfig.cmake")
        install (FILES ${find_modules} DESTINATION share/TripletMatch RENAME "TripletMatchConfig.cmake")
    else(TRIPLET_MATCH_PRECOMPILE_HEADERS)
        file (GLOB find_modules RELATIVE "${PROJECT_SOURCE_DIR}" "TripletMatchConfig.cmake")
        install (FILES ${find_modules} DESTINATION share/TripletMatch)
    endif(TRIPLET_MATCH_PRECOMPILE_HEADERS)
endif(NOT WIN32)
