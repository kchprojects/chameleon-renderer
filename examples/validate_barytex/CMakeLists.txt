cmake_minimum_required(VERSION 3.2)
project(validate_barytex)

message("{CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                               ${PTX_LEARN_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC OptixRender)
