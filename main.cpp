
#include "detection.cpp"


void test_image(const cv::Mat &image) {

    std::pair<int2*,int*> results;
    results = detect_line_pixels(image);
    int2* points_device;
    int* len_device;
    int2* points_host;
    int* len_host = new int;

    std::tie(points_device, len_device) = results;
    
    HANDLE_ERROR( cudaMemcpy(len_device, len_host, sizeof(int), cudaDeviceToHost  ));
    points_host = new int2[*len_host];
    HANDLE_ERROR( cudaMemcpy(points_device, points_host, sizeof(int2) * (*len_host), cudaDeviceToHost ));

    for (int i = 0; i < *len_host; ++i) {
        std::cout << "Point " << i << ": (" << points_host[i].x << ", " << points_host[i].y << ")" << std::endl;
    }   

    delete len_host;
    delete[] points_host;
    cudaFree(points_device);
    cudaFree(len_device);


}


int main() {
  // Test with real images
  std::vector<cv::Mat> real_images;
  real_images.push_back(cv::imread("line1.jpg", cv::IMREAD_GRAYSCALE));
  real_images.push_back(cv::imread("line2.jpg", cv::IMREAD_GRAYSCALE));
  
  // Generate synthetic test patterns with known ground truth
  std::vector<cv::Mat> synthetic_images;
  std::vector<std::vector<cv::Point>> ground_truth;
  
  for (int width = 10; width <= 30; width += 5) {
    cv::Mat synth(480, 640, CV_8UC1, cv::Scalar(0));
    std::vector<cv::Point> points;
    
    // Draw a thick white line at 45 degrees
    cv::line(synth, cv::Point(100, 100), cv::Point(400, 400), cv::Scalar(255), width);
    
    // Store ground truth points
    for (int y = 0; y < synth.rows; y++) {
      for (int x = 0; x < synth.cols; x++) {
        if (synth.at<uchar>(y, x) > 200) {
          points.push_back(cv::Point(x, y));
        }
      }
    }
    
    synthetic_images.push_back(synth);
    ground_truth.push_back(points);
  }

  cv::Mat sacrifice = synthetic_images[2];

  test_image(sacrifice);
 
  return 0;
}

