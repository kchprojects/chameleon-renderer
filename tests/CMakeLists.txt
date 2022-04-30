cmake_minimum_required(VERSION 3.2)
project(tests)

add_executable(${PROJECT_NAME}_triangle_mesh ${CMAKE_CURRENT_SOURCE_DIR}/test_triangle_mesh.cpp
                               ${PTX_SOURCES})
target_link_libraries(${PROJECT_NAME}_triangle_mesh PUBLIC OptixRender)


add_executable(${PROJECT_NAME}_lookup_tree ${CMAKE_CURRENT_SOURCE_DIR}/test_lookup_tree.cpp
                               ${PTX_SOURCES})
target_link_libraries(${PROJECT_NAME}_lookup_tree PUBLIC OptixRender)


add_executable(${PROJECT_NAME}_lookup_forest ${CMAKE_CURRENT_SOURCE_DIR}/test_lookup_forest.cpp
                               ${PTX_SOURCES})
target_link_libraries(${PROJECT_NAME}_lookup_forest PUBLIC OptixRender)


add_executable(${PROJECT_NAME}_solver ${CMAKE_CURRENT_SOURCE_DIR}/test_solver.cpp
                               ${PTX_SOURCES})
target_link_libraries(${PROJECT_NAME}_solver PUBLIC OptixRender)

add_executable(${PROJECT_NAME}_isotropic_brdf ${CMAKE_CURRENT_SOURCE_DIR}/test_isotropic_brdf.cpp
                               ${PTX_SOURCES})
target_link_libraries(${PROJECT_NAME}_isotropic_brdf PUBLIC OptixRender)