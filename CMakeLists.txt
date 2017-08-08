cmake_minimum_required(VERSION 3.8.2)
project(triplet_match)

set (triplet_match_VERSION_MAJOR 0)
set (triplet_match_VERSION_MINOR 1)

include_directories("${PROJECT_SOURCE_DIR}")
#include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")

find_package(Eigen)
find_package(PCL COMPONENTS common io search kdtree)
find_package(VoxelScore)
find_package(OpenCL)

if (EIGEN_FOUND AND PCL_FOUND AND VOXEL_SCORE_FOUND AND OpenCL_FOUND)
    include_directories(${EIGEN_INCLUDE_DIRS})
    include_directories(${PCL_INCLUDE_DIRS})
    include_directories(${VOXEL_SCORE_INCLUDE_DIRS})

    add_executable(triplet_match "apps/triplet_match.cpp")
    set_property(TARGET triplet_match PROPERTY CXX_STANDARD 17)
    target_link_libraries(triplet_match ${PCL_LIBRARIES} ${OpenCL_LIBRARIES})
endif ()

install (DIRECTORY include/ DESTINATION include/triplet_match)
