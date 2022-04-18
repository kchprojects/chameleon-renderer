cmake_minimum_required(VERSION 3.2)

option(DEFAULT_COMPILER "Use gcc-9 g++-9" ON)

if(DEFAULT_COMPILER)
  set(CMAKE_CXX_COMPILER /bin/g++-9)
  set(CMAKE_C_COMPILER /bin/gcc-9)
endif(DEFAULT_COMPILER)

project(OptixRender)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}
                    "-Wextra -Wno-missing-field-initializers")
set(CUDA_PROPAGATE_HOST_FLAGS ON)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# ------------------------------------------------------------------
# Get Optix
# ------------------------------------------------------------------
find_package(OptiX REQUIRED VERSION 7.2)

# ------------------------------------------------------------------
# include used cmake functions
# ------------------------------------------------------------------
include(${CMAKE_MODULE_PATH}/configure_build_type.cmake)
include(${CMAKE_MODULE_PATH}/configure_cuda.cmake)
include(${CMAKE_MODULE_PATH}/nvcuda_compile_ptx.cmake)
include(${CMAKE_MODULE_PATH}/shaders.cmake)
mark_as_advanced(CUDA_SDK_ROOT_DIR)

# ------------------------------------------------------------------
# add glm
# ------------------------------------------------------------------
set(glm_dir ${PROJECT_SOURCE_DIR}/thirdparty/glm)
# ------------------------------------------------------------------
# build glfw
# ------------------------------------------------------------------
# set(OpenGL_GL_PREFERENCE LEGACY) find_package(OpenGL REQUIRED)
# find_package(glfw3) add_subdirectory(thirdparty/glfWindow EXCLUDE_FROM_ALL)

# for cuda programs need to add directories without target
include_directories(${glm_dir} ${OptiX_INCLUDE})

set(SHADERS_DEPENDENCY_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/")
file(GLOB SHADERS_DEPENDENCY_HEADERS
     "${CMAKE_CURRENT_SOURCE_DIR}/include/shader_utils/*.h"
     "${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/*.h")

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nlohmann_json REQUIRED)
add_subdirectory(thirdparty/glm)
add_subdirectory(thirdparty/spdlog)

# file(GLOB_RECURSE SCENE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/scene/*.cpp")
# file(GLOB_RECURSE RENDERER_SOURCES
# "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/*.cpp")
file(GLOB_RECURSE ALL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cpp")
# barytex_shaders
#
message("compiling barytex shaders")
set(BARYTEX_SHADER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/barytex/shaders/learn)
set(BARYTEX_SHADERS ${BARYTEX_SHADER_PATH}/any_hit.cu ${BARYTEX_SHADER_PATH}/closest_hit.cu
            ${BARYTEX_SHADER_PATH}/miss.cu ${BARYTEX_SHADER_PATH}/raygen.cu)

nvcuda_compile_ptx(
  SOURCES
  ${BARYTEX_SHADERS}
  DEPENDENCIES
  ${BARYTEX_SHADERS_DEPENDENCY_HEADERS}
  TARGET_PATH
  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/barytex_shaders"
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

message("ptx: ${PTX_SOURCES}")
set(OPTRENDER_SOURCES ${ALL_SOURCES})
add_library(${PROJECT_NAME} ${OPTRENDER_SOURCES} ${PTX_SOURCES})

target_include_directories(
  ${PROJECT_NAME} PUBLIC ${OptiX_INCLUDE} ${OpenCV_INCLUDE_DIRS} include
                         thirdparty)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC ${optix_LIBRARY}
         ${CUDA_LIBRARIES}
         ${CUDA_CUDA_LIBRARY}
         ${OpenCV_LIBS}
         Eigen3::Eigen
         nlohmann_json::nlohmann_json
         ${OPENGL_gl_LIBRARY}
         glm::glm
         spdlog::spdlog
         -lstdc++fs)

target_compile_options(
  ${PROJECT_NAME}
  PUBLIC
  INTERFACE -Wno-missing-field-initializers)

add_subdirectory(examples)
