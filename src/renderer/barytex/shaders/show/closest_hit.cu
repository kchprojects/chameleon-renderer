#include <chameleon_renderer/cuda/RayType.h>

#include <chameleon_renderer/renderer/barytex/BarytexShowLaunchParamProvider.cuh>
#include <chameleon_renderer/shader_utils/common.cuh>
namespace chameleon {
using namespace barytex_show_render;
extern "C" __global__ void __closesthit__shadow() {}

inline __device__ bool cast_shadow_ray(const SurfaceInfo& si,
                                       const glm::vec3& ray_dir) {
    bool is_visible = false;
    uint32_t u0, u1;
    packPointer(&is_visible, u0, u1);
    glm::vec3 position = si.surfPos + ray_dir * 0.001f;
    optixTrace(
        optixLaunchParams.traversable, {position.x, position.y, position.z},
        {ray_dir.x, ray_dir.y, ray_dir.z},
        0.f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,                 // OPTIX_RAY_FLAG_NONE,
        unsigned(photometry_renderer::ray_t::SHADOW),  // SBT offset
        unsigned(photometry_renderer::ray_t::RAY_TYPE_COUNT),  // SBT stride
        unsigned(photometry_renderer::ray_t::SHADOW),          // missSBTIndex
        u0, u1);
    return is_visible;
}

inline __device__ bool is_in_divisor(const CUDAMaterialLookupTree::Divisor& d,
                                     const CUDAMaterialLookupTree& tree,
                                     const glm::vec3& position) {
    const glm::vec3& a_pos = tree.points[d.A_id].position;
    const glm::vec3& b_pos = tree.points[d.B_id].position;
    const glm::vec3& c_pos = tree.points[d.C_id].position;

    glm::vec3 base = b_pos - a_pos;
    glm::vec3 c_dir = c_pos - a_pos;
    glm::vec3 p_dir = position - a_pos;
    return glm::dot(c_dir, base) * glm::dot(p_dir, base) > 0;
}

inline __device__ int find_divisor_id(const CUDAMaterialLookupTree& tree,
                                      const glm::vec3& position) {
    int div_node_id = 0;
    while (div_node_id != -1) {
        const auto& div_node = tree.divisors[div_node_id];
        for (int i = 0; i < 3; ++i) {
            if (is_in_divisor(div_node.divisors[i], tree, position)) {
                if (div_node.divisors[i].child_id != -1) {
                    div_node_id = div_node.divisors[i].child_id;
                    continue;
                } else {
                    return div_node_id;
                }
            }
        }
        if (div_node.divisors[3].child_id != -1) {
            div_node_id = div_node.divisors[3].child_id;
            continue;
        } else {
            return div_node_id;
        }
    }
}

inline __device__ glm::vec3 compute_reflectance(const SurfaceInfo& si,
                                                const glm::vec3& N,
                                                const glm::vec3& L,
                                                const glm::vec3& E) {
    if (optixLaunchParams.material_forest.tree_count <= si.primID) {
        return {0, 0, 255};
    }
    const auto& curr_tree = optixLaunchParams.material_forest.trees[si.primID];

    // TODO: find divisor
    const auto& curr_divisor =
        curr_tree.divisors[find_divisor_id(curr_tree, si.surfPos)];
    // const auto& curr_divisor = curr_tree.divisors[0];


    glm::vec3 sum_val = {0.f, 0.f, 0.f};
    float weight_sum = 0;
    int point_iter = 0;
    for (int point_id : {curr_divisor.A, curr_divisor.B, curr_divisor.C}) {
        const auto& curr_point = curr_tree.points[point_id];
        if (curr_point.material.data != nullptr) {
            glm::vec3 refl;
            glm::vec3 alb;
            for (int channel = 0; channel < curr_point.material.channel_count;
                 ++channel) {
                if (curr_point.material.model == MaterialModel::Lambert) {
                    ModelData<MaterialModel::Lambert, double> lamb_model_data;
                    get_lambert_material_data(curr_point.material, channel,
                                              lamb_model_data);
                    refl[channel] = lamb_model_data.albedo;
                    // refl[channel] =
                    //     ModelReflectance<MaterialModel::Lambert,
                    //                      double>::compute(N, L, E,
                    //                                       lamb_model_data);
                }

                else if (curr_point.material.model ==
                         MaterialModel::BlinPhong) {
                    ModelData<MaterialModel::BlinPhong, double> bp_model_data;
                    get_blinphong_material_data(curr_point.material, channel,
                                                bp_model_data);
                    refl[channel] =
                        ModelReflectance<MaterialModel::BlinPhong,
                                         double>::compute(N, L, E,
                                                          bp_model_data);
                } else if (curr_point.material.model ==
                           MaterialModel::CookTorrance) {
                    ModelData<MaterialModel::CookTorrance, double>
                        ct_model_data;
                    get_cook_torrance_material_data(curr_point.material,
                                                    channel, ct_model_data);
                    refl[channel] =
                        ModelReflectance<MaterialModel::CookTorrance,
                                         double>::compute(N, L, E,
                                                          ct_model_data);
                }
            }
            if (curr_point.material.channel_count == 1) {
                refl[1] = refl[0];
                refl[2] = refl[0];
            }
            float w = 1.f / max(glm::distance(si.surfPos, curr_point.position),
                                0.0001f);
            // printf("W: %f\n", w);
            // printf("%f %f %f\n",alb[0],alb[1],alb[2]);
            sum_val += refl * w;
            weight_sum += w;
        }
        ++point_iter;
    }
    if (weight_sum != 0) {
        auto out = sum_val / weight_sum;

        // printf("%f %f %f\n", out[0], out[1], out[2]);
        return out;
    }
    return {0.f, 0.f, 0.f};
}  // namespace chameleon

extern "C" __global__ void __closesthit__radiance() {
    SurfaceInfo si = get_surface_info();
    const glm::vec3 ray_dir = {optixGetWorldRayDirection().x,
                               optixGetWorldRayDirection().y,
                               optixGetWorldRayDirection().z};
    // printf(__FUNCTION__);
    RadiationRayData* prd = getPRD<RadiationRayData>();
    glm::vec3 N = glm::normalize(si.Ns);
    glm::vec3 E = glm::normalize(-ray_dir);
    if (glm::dot(N, E) < 0) {
        N *= -1.f;
    }
    prd->mask = 255;
    for (int light_id = 0; light_id < optixLaunchParams.light_data.count;
         ++light_id) {
        const auto& curr_light = optixLaunchParams.light_data.data[light_id];
        glm::vec3 to_light = curr_light.position - si.surfPos;
        // glm::vec3 to_light = -ray_dir;
        glm::vec3 L = glm::normalize(to_light);
        if(glm::dot(N,L) < 0){
            prd->value = {255,0,0};
        }
        else if(!cast_shadow_ray(si, L)){
            prd->value = {0,255,0};
        }
        else{
            float dist_att = distance_attenuation(
                curr_light.position, optixLaunchParams.camera.pos, si.surfPos);
            float rad_att = radial_attenuation(curr_light, -L);
            if (rad_att != 0) {
                printf("rad: %f\n", rad_att);
            }

            prd->value = prd->value + 255.f * compute_reflectance(si, N, L, E); //*rad_att;  // * inv sq l*/

            prd->value = max_vec(min_vec(prd->value, 255), 0);
        }
    }
}

}  // namespace chameleon