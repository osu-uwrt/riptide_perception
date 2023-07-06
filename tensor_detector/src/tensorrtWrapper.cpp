#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/vision_msgs/msg/detection3_d.hpp"
#include "vision_msgs/vision_msgs/msg/detection3_d_array.hpp"

using namespace std::chrono_literals;

#include "tensor_detector/detector.hpp"

class TensorrtWrapper : public rclcpp::Node {
    public:
        TensorrtWrapper()
        : Node("tensorrt_wrapper"), count_(0) {
            rclcpp::QoS video_qos(10);

            detection_pub = this->create_publisher<vision_msgs::msg::Detection3DArray>("tensorrt/detections", 10);
            image_sub = this->create_subscription<sensor_msgs::msg::Image>("/zed2i/left/image_rect_color", video_qos, image_sub_callback);
        }

    private:
        void image_sub_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
            cv::Mat frame;

            frame = cv_bridge::toCvCopy(msg, msg->encoding);
            
            std::string engine_file = "yolo.engine";

            // okay, NO dont do this. I should explain the API better but this is really bad
            // this will re-load the tensorrt model every time as well as throw out the CUDA piplining in opencv::cuda
            // im not 100% sure this would be worst case, but I would expect processing times to be over 300ms per frame
            // move this out to the constructor for the class and load it once
            tensor_detector::YoloInfer infer = tensor_detector::YoloInfer(engine_file);

            cv::cuda::GpuMat gpu_frame;
            gpu_frame.upload(frame);

            infer.inferLoadedImg();

            std::vector<tensor_detector::YoloDetect> detections;
            infer.postProcessResults(detections)

            vision_msgs::msg::Detection3DArray detections3d;

            for (int i = 0; i < detections.size(); i++) {
                vision_msgs::msg::Detection3D detection3d;

                detection3d.header.stamp = msg->header.stamp;
            }
        }
}

int main(int argc, char** argv) {
    
}