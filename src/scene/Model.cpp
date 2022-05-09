#include <chameleon_renderer/scene/Model.hpp>
#include <chameleon_renderer/utils/io.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <opencv2/opencv.hpp>

// std
#include <set>

namespace std {

inline bool operator<(const tinyobj::index_t& a, const tinyobj::index_t& b) {
    if (a.vertex_index < b.vertex_index) return true;

    if (a.vertex_index > b.vertex_index) return false;

    if (a.normal_index < b.normal_index) return true;

    if (a.normal_index > b.normal_index) return false;

    if (a.texcoord_index < b.texcoord_index) return true;

    if (a.texcoord_index > b.texcoord_index) return false;

    return false;
}

}  // namespace std

namespace chameleon {

TriangleMesh::TraversableTriangleMesh::TraversableTriangleMesh(
    const TriangleMesh& base_mesh)
    : base_mesh(base_mesh) {
    vertices.resize(base_mesh.vertex.size());
    faces.resize(base_mesh.index.size());
    int curr_face= 0;
    std::cout<<"TMesh"<<std::endl;
    for (const auto& curr_vert_indices : base_mesh.index) {
        std::cout<< curr_face*100.0/base_mesh.index.size()<<"%               \r" << std::flush;
        for(int i = 0; i < 3; ++i){
            faces[curr_face].vert_indices[i] = curr_vert_indices[i];
            vertices[curr_vert_indices[i]].face_indices.push_back(curr_face);
            auto edge_pair = std::make_pair(curr_vert_indices[i],curr_vert_indices[(i+1)%3]);
            if(edges.count(edge_pair)){
                edges.at(edge_pair).face_indices[1] = curr_face; 
            }else{
                edges[edge_pair].face_indices[0] = curr_face;
            }
        }

        ++curr_face;
    }
    std::cout<<"TMesh done" << std::endl;
}

void TriangleMesh::TraversableTriangleMesh::export_ttm(const fs::path& path)const{
    
}
/*! find vertex with given position, normal, texcoord, and return
    its vertex ID, or, if it doesn't exit, add it to the mesh, and
    its just-created index */
static int addVertex(TriangleMesh& mesh, tinyobj::attrib_t& attributes,
                     const tinyobj::index_t& idx,
                     std::map<tinyobj::index_t, int>& knownVertices) {
    if (knownVertices.find(idx) != knownVertices.end())
        return knownVertices[idx];

    const glm::vec3* vertex_array =
        (const glm::vec3*)attributes.vertices.data();
    const glm::vec3* normal_array = (const glm::vec3*)attributes.normals.data();
    const glm::vec2* texcoord_array =
        (const glm::vec2*)attributes.texcoords.data();

    int newID = (int)mesh.vertex.size();
    knownVertices[idx] = newID;

    mesh.vertex.push_back(vertex_array[idx.vertex_index]);
    if (idx.normal_index >= 0) {
        while (mesh.normal.size() < mesh.vertex.size())
            mesh.normal.push_back(normal_array[idx.normal_index]);
    }
    if (idx.texcoord_index >= 0) {
        while (mesh.texcoord.size() < mesh.vertex.size())
            mesh.texcoord.push_back(texcoord_array[idx.texcoord_index]);
    }

    // just for sanity's sake:
    if (mesh.texcoord.size() > 0) mesh.texcoord.resize(mesh.vertex.size());
    // just for sanity's sake:
    if (mesh.normal.size() > 0) mesh.normal.resize(mesh.vertex.size());

    return newID;
}

/*! load a texture (if not already loaded), and return its ID in the
    model's textures[] vector. Textures that could not get loaded
    return -1 */
// int loadTexture(Model& model,
//                 std::map<std::string, int>& knownTextures,
//                 const std::string& inFileName,
//                 const std::string& modelPath) {
//     if (inFileName == "")
//         return -1;

//     if (knownTextures.find(inFileName) != knownTextures.end())
//         return knownTextures[inFileName];

//     std::string fileName = inFileName;
//     // first, fix backspaces:
//     for (auto& c : fileName)
//         if (c == '\\')
//             c = '/';
//     if (fileName[0] != '/') {
//         fileName = modelPath + "/" + fileName;
//     }

//     vec2i res;
//     int comp;
//     unsigned char* image =
//         stbi_load(fileName.c_str(), &res.x, &res.y, &comp, STBI_rgb_alpha);
//     int textureID = -1;
//     if (image) {
//         textureID = (int)model.textures.size();
//         Texture texture;
//         texture.resolution = res;
//         texture.pixel = (uint32_t*)image;

//         /* iw - actually, it seems that stbi loads the pictures
//            mirrored along the y axis - mirror them here */
//         for (int y = 0; y < res.y / 2; y++) {
//             uint32_t* line_y = texture.pixel + y * res.x;
//             uint32_t* mirrored_y = texture.pixel + (res.y - 1 - y) * res.x;
//             int mirror_y = res.y - 1 - y;
//             for (int x = 0; x < res.x; x++) {
//                 std::swap(line_y[x], mirrored_y[x]);
//             }
//         }

//         model.textures.push_back(std::move(texture));
//     } else {
//         std::cout << GDT_TERMINAL_RED << "Could not load texture from "
//                   << fileName << "!" << GDT_TERMINAL_DEFAULT << std::endl;
//     }

//     knownTextures[inFileName] = textureID;
//     return textureID;
// }

int loadTexture(Model& model, std::map<std::string, int>& knownTextures,
                const std::string& inFileName, const std::string& modelPath) {
    if (inFileName == "") return -1;

    if (knownTextures.find(inFileName) != knownTextures.end())
        return knownTextures[inFileName];

    std::string fileName = inFileName;
    // first, fix backspaces:
    for (auto& c : fileName)
        if (c == '\\') c = '/';
    if (fileName[0] != '/') {
        fileName = modelPath + "/" + fileName;
    }
    int comp;
    cv::Mat image = cv::imread(fileName, cv::IMREAD_UNCHANGED);
    int textureID = -1;
    if (!image.empty()) {
        if (image.channels() == 1) {
            cv::cvtColor(image, image, cv::COLOR_GRAY2BGRA);
        } else if (image.channels() == 3) {
            cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
        } else {
            std::cout << "type : " << image.type() << std::endl;
            std::cout << "channels : " << image.channels() << std::endl;
        }
        textureID = (int)model.textures.size();
        Texture texture;
        texture.id = textureID;
        texture.resolution = {image.cols, image.rows};
        texture.image = image;

        model.textures.push_back(std::move(texture));
    } else {
        spdlog::error("Could not load texture from {} !", fileName);
    }

    knownTextures[inFileName] = textureID;
    return textureID;
}

Model loadOBJ(const std::string& objFile) {
    Model model;

    const std::string modelDir = objFile.substr(0, objFile.rfind('/') + 1);

    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = "";

    bool readOK = tinyobj::LoadObj(&attributes, &shapes, &materials, &err, &err,
                                   objFile.c_str(), modelDir.c_str(),
                                   /* triangulate */ true);
    if (!readOK) {
        throw std::runtime_error("Could not read OBJ model from " + objFile +
                                 " : " + err);
    }

    if (materials.empty())
        throw std::runtime_error("could not parse materials ...");

    std::cout << "Done loading obj file - found " << shapes.size()
              << " shapes with " << materials.size() << " materials"
              << std::endl;
    for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
        tinyobj::shape_t& shape = shapes[shapeID];

        std::set<int> materialIDs;
        for (auto faceMatID : shape.mesh.material_ids)
            materialIDs.insert(faceMatID);

        std::map<tinyobj::index_t, int> knownVertices;
        std::map<std::string, int> knownTextures;

        for (int materialID : materialIDs) {
            TriangleMesh mesh;

            for (auto faceID = 0u; faceID < shape.mesh.material_ids.size();
                 faceID++) {
                if (shape.mesh.material_ids[faceID] != materialID) continue;
                tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                vec3i idx(addVertex(mesh, attributes, idx0, knownVertices),
                          addVertex(mesh, attributes, idx1, knownVertices),
                          addVertex(mesh, attributes, idx2, knownVertices));
                mesh.index.push_back(idx);
                // TOOD: ?????
                mesh.diffuse = (const glm::vec3&)materials[materialID].diffuse;
                // TODO:Textures
                mesh.diffuseTextureID = loadTexture(
                    model, knownTextures, materials[materialID].diffuse_texname,
                    modelDir);
            }

            if (!mesh.vertex.empty()) model.meshes.push_back(std::move(mesh));
        }
    }

    for (const auto& mesh : model.meshes)
        for (const auto& vtx : mesh.vertex) model.bounds.extend(vtx);

    std::cout << "created a total of " << model.meshes.size() << " meshes"
              << std::endl;
    return model;
}
}  // namespace chameleon
