#define DEBUG_MESSAGES
#include <chameleon_renderer/materials/barytex/BRDFMeasurement.hpp>
#include <iostream>
void print_vec(glm::vec3 v) {
    std::cout << v.x << ", " << v.y << ", " << v.z << std::endl;
}

using namespace chameleon;
int main(int argc, char const *argv[]) {
    MeasurementHit hit = {0,           {0, 0, 0}, {1, 1, 1},       {0, 1, 1},
                          {1, 0, 0.5}, glm::normalize(glm::vec3(0, -1, -1)), {0.5, 0.5, 0.5}, true};
    IsotropicBRDFMeasurement imes(hit);
    std::cout << "e_el: " << imes.eye_elevation << ", l_el"
              << imes.light_elevation << ", l_az" << imes.light_azimuth
              << std::endl;
    print_vec(imes.eye());
    print_vec(imes.light());
    return 0;
}
