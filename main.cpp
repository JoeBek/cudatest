
#include "detection.hpp"
#include <vector>
#include <string>


void test_image(const cv::Mat &image) {

    std::pair<int2*,int*> results;
    results = detect_line_pixels(image);
    int2* points_device;
    int* len_device;
    int2* points_host;
    int* len_host = new int;

    std::tie(points_device, len_device) = results;
    
    HANDLE_ERROR( cudaMemcpy(len_host, len_device, sizeof(int), cudaMemcpyDeviceToHost  ));
    points_host = new int2[*len_host];
    HANDLE_ERROR( cudaMemcpy(points_host, points_device, sizeof(int2) * (*len_host), cudaMemcpyDeviceToHost ));

    for (int i = 0; i < *len_host; ++i) {
        std::cout << "Point " << i << ": (" << points_host[i].x << ", " << points_host[i].y << ")" << std::endl;
    }   

    delete len_host;
    delete[] points_host;
    cudaFree(points_device);
    cudaFree(len_device);


}

void test_integral_image(const cv::Mat &image) {

    int width = image.cols + 1;
    int length = image.rows + 1;
    int total = width * length;


    std::pair<Npp32f*,Npp64f*> results;
    Npp32f* integral_device;
    Npp64f* integral_sq_device;
    Npp32f* integral_host;
    Npp64f* integral_sq_host = new Npp64f[total];
    integral_host = new Npp32f[total];

    results = __get_integral_image(image);
    
    std::tie(integral_device, integral_sq_device) = results;

    cudaDeviceSynchronize();
    
    HANDLE_ERROR( cudaMemcpy(integral_sq_host, integral_sq_device, sizeof(Npp64f) * total, cudaMemcpyDeviceToHost  ));
    HANDLE_ERROR( cudaMemcpy(integral_host, integral_device, sizeof(Npp32f) * total, cudaMemcpyDeviceToHost ));

    for (int i = 5; i < total; i += width) {
        std::cout << "row " << i << " integral result: " << integral_host[i]  << std::endl;
    }   

    delete integral_sq_host;
    delete[] integral_host;
    cudaFree(integral_device);
    cudaFree(integral_sq_device);


}

int main() {
  // Test with real images
  std::vector<cv::Mat> real_images;
  //real_images.push_back(cv::imread("line1.jpg", cv::IMREAD_GRAYSCALE));
  //real_images.push_back(cv::imread("line2.jpg", cv::IMREAD_GRAYSCALE));

  cv::Mat real = cv::imread("../../data/IMG_6942.JPG", cv::IMREAD_GRAYSCALE);
  
  // Generate synthetic test patterns with known ground truth
  std::vector<cv::Mat> synthetic_images;
  std::vector<std::vector<cv::Point>> ground_truth;
  
  for (int width = 10; width <= 30; width += 5) {
    cv::Mat synth(480, 640, CV_8UC1, cv::Scalar(1));
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

  cv::imshow("image numba " , sacrifice);
  //cv::imshow("real", real);

  std::cout << "Input image type: " << sacrifice.type() << std::endl;

  if (!sacrifice.isContinuous()) {
    std::cout << "not continous in memory" << std::endl;
    exit(EXIT_FAILURE);
  }

  uchar pixel = sacrifice.at<uchar>(0,0);
  std::cout << "pixel 0: " << static_cast<int>(pixel) << std::endl;

  test_image(sacrifice);

  cv::waitKey(0);
 
  return 0;
}

