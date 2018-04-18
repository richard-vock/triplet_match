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
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

install (DIRECTORY include/ DESTINATION include/triplet_match)

if (TRIPLET_MATCH_PRECOMPILE_HEADERS)
    find_package(PCL COMPONENTS common io search kdtree)
    find_package(Eigen3)
    find_package(fmt)
    #find_package(CUDA REQUIRED)
    #find_package(OPTIX)
    file (GLOB_RECURSE obj RELATIVE "${PROJECT_SOURCE_DIR}" "src/*.cpp")
    if (EIGEN_FOUND AND PCL_FOUND)
        if(CMAKE_COMPILER_IS_GNUCXX)
            add_definitions(-fPIC)
            add_definitions(-O3)
            add_definitions(-g)
            add_definitions(-fopenmp)
            add_definitions(-std=c++17)
            add_definitions(-Wall)
            add_definitions(-Wno-int-in-bool-context)
            add_definitions(-Wno-ignored-attributes)
            add_definitions(-Wno-deprecated-declarations)
            add_definitions(-Wno-maybe-uninitialized)
            add_definitions(-Wno-sign-compare)
        endif()
        include_directories(${PCL_INCLUDE_DIRS})
        #include_directories(${CUDA_INCLUDE_DIRS})
        #include_directories(${OPTIX_INCLUDE_DIRS})
        add_library(triplet_match SHARED ${obj})

        #cuda_compile_ptx(
            #cuda_ptx_files
            #"${PROJECT_SOURCE_DIR}/optix/RayCaster.cu"
        #)

        #add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/RayCaster.ptx COMMAND ${CMAKE_COMMAND} -E rename ${cuda_ptx_files} ${CMAKE_BINARY_DIR}/RayCaster.ptx DEPENDS ${cuda_ptx_files}) 
        #add_custom_target(ptx ALL DEPENDS ${CMAKE_BINARY_DIR}/RayCaster.ptx "${PROJECT_SOURCE_DIR}/optix/RayCaster.cu" SOURCES "${PROJECT_SOURCE_DIR}/optix/RayCaster.cu")

        set_property(TARGET triplet_match PROPERTY CXX_STANDARD 17)
        target_link_libraries(triplet_match ${PCL_LIBRARIES} Eigen3::Eigen fmt::fmt "dl")
        #target_link_libraries(triplet_match ${PCL_LIBRARIES} ${OPTIX_LIBRARIES} Eigen3::Eigen fmt::fmt ${OPTIX_LIBRARIES} "dl")
        install (TARGETS triplet_match DESTINATION lib)
        #install (FILES "${CMAKE_BINARY_DIR}/RayCaster.ptx" DESTINATION share/triplet_match/optix)
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
