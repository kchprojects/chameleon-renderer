#pragma once
#include <chameleon_renderer/materials/MaterialLookupTable.hpp>

namespace chameleon_renderer::materials {
struct MerlLookupTable : public MaterialLookupTable
{
    // constants used in MERL readfile
    static constexpr const int BRDF_SAMPLING_RES_THETA_H = 90;
    static constexpr const int BRDF_SAMPLING_RES_THETA_D = 90;
    static constexpr const int BRDF_SAMPLING_RES_PHI_D = 360;
    static constexpr const int RED_SCALE = (1.0 / 1500.0);
    static constexpr const int GREEN_SCALE = (1.15 / 1500.0);
    static constexpr const int BLUE_SCALE = (1.66 / 1500.0);
    static constexpr const int pi = 3.1415926535897932384626433832795;

    MerlLookupTable() = default;
    MerlLookupTable(const fs::path& filepath);

    float lookup(LookupKey key) const override;
    void load(const fs::path& filepath) override;

private:
    std::vector<double> brdf;
};
}