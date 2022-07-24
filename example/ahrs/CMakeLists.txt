find_package(pybind11 2.9 REQUIRED)

pybind11_add_module(ahrs ahrs.cpp)
target_link_libraries(ahrs PRIVATE eskf)
set_target_properties(ahrs 
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)