function(compile_shaders SHADER_PATH TARGET_PATH)
    set(SHADERS ${SHADER_PATH}/any_hit.cu ${SHADER_PATH}/closest_hit.cu
              ${SHADER_PATH}/miss.cu ${SHADER_PATH}/raygen.cu)
    message("compiling shaders")
    nvcuda_compile_ptx(
    SOURCES
    ${SHADERS}
    TARGET_PATH
    ${TARGET_PATH}  
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
endfunction()

