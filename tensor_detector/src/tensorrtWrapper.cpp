#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/geometry_msgs/msg/pose.hpp"
#include "vision_msgs/vision_msgs/msg/object_hypothesis_with_pose.hpp"
#include "vision_msgs/vision_msgs/msg/detection3_d.hpp"
#include "vision_msgs/vision_msgs/msg/detection3_d_array.hpp"

#include "tensorrt_detector/yolov5_detector.hpp"

using namespace std::chrono_literals;

class TensorrtWrapper : public rclcpp::Node
{
public:
    TensorrtWrapper()
        : Node("tensorrt_wrapper")
    {
        std::string engine_file = "yolo.engine";

        // init and load the detector
        infer = std::make_shared<yolov5::Detector>();
        infer->init();
        infer->loadEngine(engine_file);
        infer->setScoreThreshold(0.3);

        // create the pub for detections
        detection_pub = this->create_publisher<vision_msgs::msg::Detection3DArray>("detections", rclcpp::SystemDefaultsQoS());

        // create the two image subs we need to consume
        left_image_sub = this->create_subscription<sensor_msgs::msg::Image>("/zed2i/zed_node/left/image_rect_color",
                                                                            rclcpp::SystemDefaultsQoS(), std::bind(&TensorrtWrapper::left_image_sub_callback, this, std::placeholders::_1));

        depth_image_sub = this->create_subscription<sensor_msgs::msg::Image>("/zed2i/zed_node/depth/depth_registered",
                                                                             rclcpp::SystemDefaultsQoS(), std::bind(&TensorrtWrapper::depth_image_sub_callback, this, std::placeholders::_1));
    }

private:
    void left_image_sub_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;

        if (frame.empty())
        {
            RCLCPP_WARN(get_logger(), "Empty image on topic %s", left_image_sub->get_topic_name());
            return;
        }

        // RCLCPP_INFO(get_logger(), "Image dim: (%d, %d)", frame.size().width, frame.size().height);

        cv::cuda::GpuMat gpu_frame;
        std::vector<yolov5::Detection> detections;

        gpu_frame.upload(frame);
        auto res = infer->detect(gpu_frame, &detections);

        // make the detection array and copy the header from the camera as this is all still camera relative
        vision_msgs::msg::Detection3DArray detections3d;
        detections3d.header = msg->header;
        if (depths.size().width > 0)
        {
            // process each of the detections
            for (yolov5::Detection detection : detections)
            {

                cv::Rect bbox = detection.boundingBox();
                RCLCPP_INFO(get_logger(), "Detected %i with conf %f", detection.classId(), detection.score());
                int totalPoints = 0;
                float totalDepth = 0;
                for (int i = -1; i < 2; i++)
                {
                    int u = (bbox.x + (i * 3));// + bbox.width / 2;
                    for (int j = -1; j < 2; j++)
                    {
                        int v = (bbox.y + (j * 3));// + bbox.height / 2;
                        float depth = depths.at<float>(u, v);

                        if (!isnanf(depth) && depth > 0 && depth < 100)
                        {
                            RCLCPP_INFO(get_logger(), "ADDING DEPTH: %f", depth);
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

                    float xPixLoc = bbox.x + bbox.width / 2;
                    float yPixLoc = bbox.y + bbox.height / 2;

                    float xPos = (xPixLoc - cx) * depth / fx;
                    float yPos = (yPixLoc - cy) * depth / fy;

                    vision_msgs::msg::Detection3D detection3d;

                    detection3d.header.stamp = msg->header.stamp;
                    detection3d.header.frame_id = "zed2i_left_optical_frame";

                    geometry_msgs::msg::Pose objPose;
                    vision_msgs::msg::ObjectHypothesisWithPose objHypo;

                    objHypo.hypothesis.class_id = detection.classId();
                    objHypo.pose.pose.position.x = xPos;
                    objHypo.pose.pose.position.y = yPos;
                    objHypo.pose.pose.position.z = depth;

                    detection3d.results.push_back(objHypo);
                    detections3d.detections.push_back(detection3d);
                }
            }
        }
        detection_pub->publish(detections3d);
    }

    void depth_image_sub_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        depths = cv_bridge::toCvShare(msg, "32FC1")->image;
        // RCLCPP_INFO(get_logger(), "Depth im size (%d, %d)", depths.size().width, depths.size().height);

        if (depths.empty())
        {
            RCLCPP_WARN(get_logger(), "Empty image on topic %s", depth_image_sub->get_topic_name());
            return;
        }
    }

    cv::Mat depths;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detection_pub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_image_sub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_image_sub;
    std::shared_ptr<yolov5::Detector> infer;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TensorrtWrapper>());
    rclcpp::shutdown();
}