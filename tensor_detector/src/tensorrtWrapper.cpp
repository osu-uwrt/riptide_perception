#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
// #include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/geometry_msgs/msg/pose.hpp"
#include "vision_msgs/vision_msgs/msg/object_hypothesis_with_pose.hpp"
#include "vision_msgs/vision_msgs/msg/detection3_d.hpp"
#include "vision_msgs/vision_msgs/msg/detection3_d_array.hpp"

using namespace std::chrono_literals;

#include "tensor_detector/detector.hpp"

class TensorrtWrapper : public rclcpp::Node
{
public:
    TensorrtWrapper()
        : Node("tensorrt_wrapper")
    {
        rclcpp::QoS video_qos(10);

        std::string engine_file = "yolo.engine";

        infer = std::make_shared<tensor_detector::YoloInfer>(engine_file);
        detection_pub = this->create_publisher<vision_msgs::msg::Detection3DArray>("tensorrt/detections", 1);
        left_image_sub = this->create_subscription<sensor_msgs::msg::Image>("/zed2i/zed_node/left/image_rect_color", video_qos, std::bind(&TensorrtWrapper::left_image_sub_callback, this, std::placeholders::_1));
        depth_image_sub = this->create_subscription<sensor_msgs::msg::Image>("/zed2i/zed_node/depth/depth_registered", video_qos, std::bind(&TensorrtWrapper::depth_image_sub_callback, this, std::placeholders::_1));
    }

private:
    void left_image_sub_callback(const sensor_msgs::msg::Image &msg)
    {
        // cv_bridge::CvImagePtr frame;

        // frame = cv_bridge::toCvCopy(msg, msg.encoding);

        cv::Mat frame;

        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(frame);

        infer->loadNextImage(gpu_frame);

        infer->inferLoadedImg();

        std::vector<tensor_detector::YoloDetect> detections;
        infer->postProcessResults(detections);

        vision_msgs::msg::Detection3DArray detections3d;

        for (tensor_detector::YoloDetect detection : detections)
        {
            int totalPoints = 0;
            float totalDepth = 0;
            for (int i = -1; i < 2; i++)
            {
                int u = (detection.bounds.x + (i * 3)) + detection.bounds.width / 2;
                for (int j = -1; j < 2; j++)
                {
                    int v = (detection.bounds.y + (j * 3)) + detection.bounds.height / 2;
                    int centerIdx = u + detection.bounds.width * v;

                    float depth = this->depths[centerIdx];

                    if (!isnanf(depth))
                    {
                        totalPoints++;
                        totalDepth += depth;
                    }
                }
            }

            if (totalPoints > 0)
            {
                float fx = 1078.96;
                float fy = 1079.0;
                float cx = 967.35;
                float cy = 544.594;
                float depth = totalDepth / totalPoints;

                float xPixLoc = detection.bounds.x + detections.bounds.width / 2;
                float yPixLoc = detection.bounds.y + detections.bounds.height / 2;

                float xPos = (xPixLoc - cx) * depth / fx;
                float yPos = (yPixLoc - cy) * depth / fy;

                vision_msgs::msg::Detection3D detection3d;

                detection3d.header.stamp = msg.header.stamp;
                detection3d.header.frame_id = "zed2i_left_optical_frame";

                geometry_msgs::msg::Pose objPose;
                vision_msgs::msg::ObjectHypothesisWithPose objHypo;

                objHypo.hypothesis.class_id = detection.class_id;
                objHypo.pose.pose.position.x = xPos;
                objHypo.pose.pose.position.y = yPos;
                objHypo.pose.pose.position.z = depth;

                detection3d.results.push_back(objPose);
                detections3d.detections.push_back(detection3d);
            }
        }
        detection_pub->publish(detections3d);
    }

    void depth_image_sub_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        this->depths = (float *)(&msg->data[0]);
    }

    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detection_pub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_image_sub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_image_sub;
    std::shared_ptr<tensor_detector::YoloInfer> infer;
    float *depths;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TensorrtWrapper>());
    rclcpp::shutdown();
}