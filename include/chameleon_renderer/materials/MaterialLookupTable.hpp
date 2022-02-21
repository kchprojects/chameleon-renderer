#pragma once
#include <vector>
#include <chameleon_renderer/stdutils.hpp>

namespace chameleon_renderer::materials{
    enum class LookupChannel{
        Red,
        Green,
        Blue,
        Grayscale
    };

    struct LookupKey{
        float pan_in; //phi of input ray
        float tilt_in; //theta of input ray

        float pan_out; //phi of output ray
        float tilt_out; //theta of output ray

        LookupChannel interested_channel;
    };

    struct MaterialLookupTable{
        virtual float lookup(LookupKey key) const = 0;
        virtual void load(const fs::path& filepath) = 0;
        virtual ~MaterialLookupTable()=default;
    };
}

