find_package(CUDA REQUIRED)

#include_directories(${CUDA_TOOLKIT_INCLUDE})
if (CUDA_TOOLKIT_ROOT_DIR)
	include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
endif()
include_directories(${OptiX_INCLUDE})

if (WIN32)
  add_definitions(-DNOMINMAX)
endif()

find_program(BIN2C bin2c
  DOC "Path to the cuda-sdk bin2c executable.")

add_definitions(-D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__=1)


