#include <ctime>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "mtcnn.hpp"

using namespace cv;
using namespace std;
using namespace mtcnn;

/*! \brief Timer */
class Timer {
  using Clock = std::chrono::high_resolution_clock;
public:
  /*! \brief start or restart timer */
  inline void Tic() {
    start_ = Clock::now();
  }
  /*! \brief stop timer */
  inline void Toc() {
    end_ = Clock::now();
  }
  /*! \brief return time in ms */
  inline double Elasped() {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
    return duration.count();
  }

private:
  Clock::time_point start_, end_;
};

int main(int argc, char* argv[]) {
  string model_dir = "../model/";
  MTCNNDetector detector(proto_dir + "det1.prototxt", model_dir + "det1.caffemodel",
                        proto_dir + "det2.prototxt", model_dir + "det2.caffemodel",
                        proto_dir + "det3.prototxt", model_dir + "det3.caffemodel",
                        proto_dir + "det4.prototxt", model_dir + "det4.caffemodel", 0);

  Mat img = cv::imread("../img/test1.jpg", CV_LOAD_IMAGE_COLOR);
  Timer timer;
  timer.Tic();
  detector.SetMinSize(40);
  vector<FaceInfo> faces = detector.Detect(img);
  timer.Toc();

  cout << "detect costs " << timer.Elasped() << "ms" << endl;

  for (int i = 0; i < faces.size(); i++) {
    FaceInfo& face = faces[i];
    cv::rectangle(img, face.bbox, Scalar(0, 0, 255), 2);
    for (int j = 0; j < 5; j++) {
      cv::circle(img, face.landmark5[j], 2, Scalar(0, 255, 0), -1);
    }
  }
  cv::imwrite("result.jpg", img);

  return 0;
}
