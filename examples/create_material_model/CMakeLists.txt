cmake_minimum_required(VERSION 3.2)
project(create_material_model)

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                               ${PTX_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC OptixRender)
