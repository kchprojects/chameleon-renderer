#include <chameleon_renderer/scene/Model.hpp>
using namespace chameleon;

int main(){
    TriangleMesh tm;
    tm.vertex = {{0,0,0},{0,1,0},{1,0,0},{1,1,0},{2,1,0}};
    tm.index = {{0,1,2},{1,2,3},{2,3,4}};
    const auto& ttm = tm.traversable_mesh();
    std::cout<<"num of faces: " << ttm.faces.size()<<std::endl;
    std::cout<<"num of vertices: " << ttm.vertices.size()<<std::endl;
    std::cout<<"num of edges: " << ttm.edges.size()<<std::endl;
    for(const auto& f : ttm.faces){
        std::cout<<"face " << f.vert_indices[0] << "," << f.vert_indices[1] << "," << f.vert_indices[2]<<std::endl;  
    }
    for(int i = 0; i < 3; ++i){
        for(int fi : ttm.vertices[ttm.faces[0].vert_indices[i]].face_indices){
            std::cout<<fi<<",";
        }
        std::cout<<"\n------------"<<std::endl;
    }
    std::cout<<ttm.edges.at({1,2}).face_indices[0]<<"," << ttm.edges.at({1,2}).face_indices[1]<<std::endl;
    std::cout<<ttm.edges.at({2,3}).face_indices[0]<<"," << ttm.edges.at({2,3}).face_indices[1]<<std::endl;
    std::cout<<ttm.edges.at({0,1}).face_indices[0]<<"," << ttm.edges.at({0,1}).face_indices[1]<<std::endl;
    return 0;
}


// |\
// --
// |\|
