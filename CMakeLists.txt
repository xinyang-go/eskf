cmake_minimum_required(VERSION 3.15)
project(ins)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Ceres 2.1 REQUIRED)

add_library(eskf INTERFACE)
target_include_directories(eskf INTERFACE include)
target_link_libraries(eskf INTERFACE Ceres::ceres)


option(BUILD_EXAMPLE "build example." ON)
if(BUILD_EXAMPLE)
    add_subdirectory(example/ahrs)
endif()


