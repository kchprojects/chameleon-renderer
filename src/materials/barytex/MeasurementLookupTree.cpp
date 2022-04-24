#include <algorithm>
#include <chameleon_renderer/materials/barytex/MeasurementLookupTree.hpp>
#include <chameleon_renderer/utils/math_io.hpp>

namespace chameleon {
MeasurementLookupTree::MeasurementLookupTree(const glm::vec3& A,
                                             const glm::vec3& B,
                                             const glm::vec3& C,
                                             float min_distance,
                                             int max_depth) {
    float max_dist = std::max(glm::length(A - B),
                              std::max(glm::length(B - C), glm::length(A - C)));
    int new_max_depth = 0;
    float distance_factor = max_dist;
    while (distance_factor >= min_distance && (max_depth == -1 || new_max_depth < max_depth)) {
        distance_factor /= 2;
        ++new_max_depth;
    }
    tree_depth = std::max(std::max(new_max_depth, max_depth),1);

    int sub_division_count = std::pow(2, tree_depth - 1);
    int layer_count = sub_division_count + 1;
    int point_count =
        ((layer_count + 1) * (layer_count)) / 2;  // see graphical proof

    // TODO: build points column_wise
    measurement_points.resize(point_count);
    int layer_begin = 0;
    int prev_layer_begin = 0;
    std::vector<int> coordinate_ticks(layer_count);
    std::generate(coordinate_ticks.begin() + 1, coordinate_ticks.end(),
                  [i = 0, last_val = 0]() mutable {
                      ++i;
                      last_val += i;
                      return last_val;
                  });
    auto coords_from_index = [&coordinate_ticks](int index) {
        auto lower = std::upper_bound(coordinate_ticks.begin(),
                                      coordinate_ticks.end(), index);
        --lower;
        return std::make_pair<int, int>(
            std::distance(coordinate_ticks.begin(), lower), index - *lower);
    };
    auto index_from_coords = [&coordinate_ticks](int layer, int movement) {
        return coordinate_ticks[layer] + movement;
    };

    for (int division_layer = 0; division_layer < tree_depth;
         ++division_layer) {
        int division_node_count = std::pow(4, division_layer);
        for (int divisor_id = 0; divisor_id < division_node_count;
             ++divisor_id) {
            divisors.emplace_back();
            auto& curr_divisor = divisors.back();
            if (division_layer != 0) {
                curr_divisor.parent_id = prev_layer_begin + divisor_id / 4;
                auto& parent_divisor = divisors[curr_divisor.parent_id];
                auto& move_line = parent_divisor.divisors[divisor_id % 4];
                curr_divisor.A = move_line.A_id;
                curr_divisor.B = move_line.B_id;
                curr_divisor.C = move_line.C_id;
                move_line.child_id = divisors.size()-1;
            } else {
                curr_divisor.C = 0;
                curr_divisor.A = measurement_points.size() - layer_count;
                curr_divisor.B = measurement_points.size() - 1;

                measurement_points[curr_divisor.C].position = C;
                measurement_points[curr_divisor.C].filled = true;
                measurement_points[curr_divisor.A].position = A;
                measurement_points[curr_divisor.A].filled = true;
                measurement_points[curr_divisor.B].position = B;
                measurement_points[curr_divisor.B].filled = true;
            }

            auto [A_layer, A_movement] = coords_from_index(curr_divisor.A);
            auto [B_layer, B_movement] = coords_from_index(curr_divisor.B);
            auto [C_layer, C_movement] = coords_from_index(curr_divisor.C);
            int A_B_half = index_from_coords((A_layer + B_layer) / 2,
                                             (A_movement + B_movement) / 2);
            if (!measurement_points[A_B_half].filled) {
                measurement_points[A_B_half].position =
                    (measurement_points[curr_divisor.A].position +
                     measurement_points[curr_divisor.B].position) /
                    2.f;
                measurement_points[A_B_half].filled = true;
            }
            int B_C_half = index_from_coords((B_layer + C_layer) / 2,
                                             (B_movement + C_movement) / 2);
            if (!measurement_points[B_C_half].filled) {
                measurement_points[B_C_half].position =
                    (measurement_points[curr_divisor.B].position +
                     measurement_points[curr_divisor.C].position) /
                    2.f;
                measurement_points[B_C_half].filled = true;
            }
            int C_A_half = index_from_coords((C_layer + A_layer) / 2,
                                             (C_movement + A_movement) / 2);
            if (!measurement_points[C_A_half].filled) {
                measurement_points[C_A_half].position =
                    (measurement_points[curr_divisor.C].position +
                     measurement_points[curr_divisor.A].position) /
                    2.f;
                measurement_points[C_A_half].filled = true;
            }
            curr_divisor.divisors[0] = {A_B_half, C_A_half, divisors.back().A};
            curr_divisor.divisors[1] = {A_B_half, B_C_half, divisors.back().B};
            curr_divisor.divisors[2] = {C_A_half, B_C_half, divisors.back().C};
            curr_divisor.divisors[3] = {A_B_half, B_C_half, C_A_half};
        }
        prev_layer_begin = layer_begin;
        layer_begin += division_node_count;
    }
}

MeasurementLookupTree::MeasurementLookupTree(const nlohmann::json& j) {
    tree_depth = j.at("tree_depth");
    for (const auto& point_j : j.at("points")) {
        measurement_points.emplace_back();
        if (point_j.at("material") != nullptr) {
            // TODO:
        }
        measurement_points.back().position = vec_from_json(point_j.at("position"));
    }
    for (const auto& div_j : j.at("divisor_nodes")) {
        divisors.emplace_back();
        divisors.back().parent_id = div_j.at("parent_id");
        divisors.back().A = div_j.at("A");
        divisors.back().B = div_j.at("B");
        divisors.back().C = div_j.at("C");

        for (int i = 0; i < 4; ++i) {
            divisors.back().divisors[i].A_id = div_j.at("divisors")[i].at("A_id");
            divisors.back().divisors[i].B_id = div_j.at("divisors")[i].at("B_id");
            divisors.back().divisors[i].C_id = div_j.at("divisors")[i].at("C_id");
            divisors.back().divisors[i].child_id = div_j.at("divisors")[i].at("child_id");
        }
    }
}

nlohmann::json MeasurementLookupTree::serialize() const {
    nlohmann::json out;
    out["tree_depth"] = tree_depth;
    out["points"] = nlohmann::json::array();
    for (const auto& point : measurement_points) {
        nlohmann::json p_j;
        if (!point.material) {
            p_j["material"] = nullptr;
        } else {
            // TODO:
        }
        p_j["position"] = to_json(point.position);
        out["points"].push_back(p_j);
    }
    out["divisor_nodes"] = nlohmann::json::array();
    for (const auto& div : divisors) {
        nlohmann::json d_j;
        d_j["parent_id"] = div.parent_id;
        d_j["A"] = div.A;
        d_j["B"] = div.B;
        d_j["C"] = div.C;
        d_j["divisors"] = nlohmann::json::array();
        for (const auto& d_small : div.divisors) {
            nlohmann::json d_small_j;
            d_small_j["A_id"] = d_small.A_id;
            d_small_j["B_id"] = d_small.B_id;
            d_small_j["C_id"] = d_small.C_id;
            d_small_j["child_id"] = d_small.child_id;
            d_j["divisors"].push_back(d_small_j);
        }
        out["divisor_nodes"].push_back(d_j);
    }
    return out;
}
}  // namespace chameleon
