// #include <GL/gl.h>

#include <chameleon_renderer/materials/barytex/MeasurementTree.hpp>
#include <chrono>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>

// #define SHOW_SHADOWS
namespace chameleon {

extern "C" int main(int argc, char** argv) {
    std::vector<MeasurementHit> hits = {
        {0, {0.7, 0.2, 0.1}, {1, 0, 0}, {1, 0, 0}, {1, 1, 1}, true},
        {0, {0.2, 0.7, 0.1}, {1, 0, 0}, {1, 0, 0}, {1, 1, 1}, true}};

    MeasurementTree mt(hits,4);
    

    return 0;
}  // namespace chameleon

}  // namespace chameleon
