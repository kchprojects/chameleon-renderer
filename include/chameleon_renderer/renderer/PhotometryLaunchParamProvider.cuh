#include <chameleon_renderer/cuda/LaunchParams.h>
/*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" __constant__ chameleon::photometry_render::LaunchParams optixLaunchParams;