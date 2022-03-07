cmake_minimum_required(VERSION 3.2)
project(render_normals)

set(SHADERS
    # Core shaders.
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/any_hit.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/closest_hit.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/miss.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/raygen.cu)

# set(SHADERS_DEPENDENCIES
# ${CMAKE_CURRENT_SOURCE_DIR}/include/shader_utils/common.cuh
# ${CMAKE_CURRENT_SOURCE_DIR}/include/shader_utils/PerRayData.h )

# message( "found optix include : ${OptiX_INCLUDE}") message( "PTX output:
# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}") message( "found shaders: ${SHADERS}")
# message( "found shaders: ${SHADERS_HEADERS}")
message("DIRS: ${SHADERS_DEPENDENCY_DIRS}")

nvcuda_compile_ptx(
  SOURCES
  ${SHADERS}
  DEPENDENCIES
  ${SHADERS_DEPENDENCY_HEADERS}
  TARGET_PATH
  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  GENERATED_FILES
  PTX_SOURCES
  NVCC_OPTIONS
  "--gpu-architecture=compute_50"
  "--use_fast_math"
  "--relocatable-device-code=true"
  "--generate-line-info"
  "-Wno-deprecated-gpu-targets"
  "-I${OptiX_INCLUDE}"
  "-I${SHADERS_DEPENDENCY_DIRS}"
  "-I${glm_dir}")

message("SSSS${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${PTX_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC OptixRender)
