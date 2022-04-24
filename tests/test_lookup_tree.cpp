#include <chameleon_renderer/materials/barytex/MeasurementLookupTree.hpp>
#include <chameleon_renderer/materials/barytex/viz.hpp>
#include <fstream>
using namespace chameleon;

void draw_mlt(const MeasurementLookupTree& mlt, cv::Mat& canvas){
    for(const auto& div : mlt.divisors){
        cv::Mat empty_canvas = cv::Mat::zeros(1000,1000,CV_32FC3);
        glm::vec3 A = mlt.measurement_points[div.A].position;
        glm::vec3 B = mlt.measurement_points[div.B].position;
        glm::vec3 C = mlt.measurement_points[div.C].position;
        draw_triangle(canvas,{A.x,A.y},{B.x,B.y},{C.x,C.y},0,2,800,100);
        draw_triangle(empty_canvas,{A.x,A.y},{B.x,B.y},{C.x,C.y},0,2,800,100);
        cv::imshow("img",canvas);
        cv::imshow("empty_canvas",empty_canvas);
        cv::waitKey(10);
    }

    for(const auto& point : mlt.measurement_points){
        draw_point(canvas,{point.position.x,point.position.y},10,1);
        cv::imshow("img",canvas);
        cv::waitKey();
    }
}
int main(){
    const MeasurementLookupTree mlt({0,0,0},{1,0,0},{0.5,std::sqrt(1-0.25),0},0.1,-1);
    int count = 0;
    cv::Mat canvas = cv::Mat::zeros(1000,1000,CV_32FC3);
    std::printf("[%d,%d,%d]\n",mlt.divisors[0].A,mlt.divisors[0].B,mlt.divisors[0].C);
    for(int layer = 1;layer < 3; ++layer){
        int last_layer_end = count;
        for(int i = 0; i < pow(4,layer); ++i){
            const auto& div = mlt.divisors[last_layer_end + i+1];
            std::printf("[%d,%d,%d]",div.A,div.B,div.C);
            ++count;
        }
        std::printf("\n");
    }
    {
        std::ofstream ofs("test_mlt.json");
        ofs << mlt.serialize().dump(4); 
    }
    nlohmann::json loaded;
    {
        std::ifstream ifs("test_mlt.json");
        ifs >> loaded;
    }

    const MeasurementLookupTree mlt_loaded(loaded);
    draw_mlt(mlt_loaded,canvas);


    return 0;
}


// |\
// --
// |\|
