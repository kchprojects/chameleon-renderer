#include <chameleon_renderer/renderer/barytex/BarytexObservation.hpp>

namespace chameleon
{

    CUDABarytexObservation BarytexObservation::get_cuda(const eigen_utils::Mat4<float>& obj_mat) const {
        CUDABarytexObservation out;
        input_image = {image};
        out.image.data = reinterpret_cast<glm::vec3*>(input_image.buffer_ptr());
        out.image.cols = input_image.res_x;
        out.image.rows = input_image.res_y;

        out.light = light->convert_to_cuda(obj_mat);   

        return out;
    }

} // namespace chameleon

