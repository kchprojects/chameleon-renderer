
add_executable(triangle_mat ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp )
target_link_libraries(triangle_mat PUBLIC OptixRender)
