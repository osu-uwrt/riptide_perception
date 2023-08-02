#include <chrono>
// #include <iostream>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"
#include "vision_msgs/msg/detection3_d.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"

#include "eigen3/Eigen/SVD"

#include "tensorrt_detector/yolov5_detector.hpp"

using namespace std::chrono_literals;

class TensorrtWrapper : public rclcpp::Node
{
public:
    TensorrtWrapper()
        : Node("tensorrt_wrapper")
    {
        std::string engine_file = "/home/coalman321/yolo.engine";

        // init and load the detector
        infer = std::make_shared<yolov5::Detector>();
        infer->init();
        infer->loadEngine(engine_file);
        infer->setScoreThreshold(0.3);

        // create the pub for detections
        detection_pub = this->create_publisher<vision_msgs::msg::Detection3DArray>("detections", rclcpp::SystemDefaultsQoS());

        // create the two image subs we need to consume
        left_image_sub = this->create_subscription<sensor_msgs::msg::Image>("/zedm/zed_node/left/image_rect_color",
                                                                            rclcpp::SystemDefaultsQoS(), std::bind(&TensorrtWrapper::left_image_sub_callback, this, std::placeholders::_1));

        depth_image_sub = this->create_subscription<sensor_msgs::msg::Image>("/zedm/zed_node/depth/depth_registered",
                                                                             rclcpp::SystemDefaultsQoS(), std::bind(&TensorrtWrapper::depth_image_sub_callback, this, std::placeholders::_1));

        camera_info_sub = this->create_subscription<sensor_msgs::msg::CameraInfo>("/zedm/zed_node/left/camera_info",
                                                                                  rclcpp::SystemDefaultsQoS(), std::bind(&TensorrtWrapper::camera_info_sub_callback, this, std::placeholders::_1));
    }

private:
    void left_image_sub_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        cv::Mat frame;
        cv_bridge::toCvShare(msg, "bgr8")->image.copyTo(frame);

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
        if (gotCalibrationInfo && depths.size().width > 0)
        {
            // process each of the detections
            for (yolov5::Detection detection : detections)
            {
                cv::Rect bbox = detection.boundingBox();
                int totalPoints = 0;
                float totalDepth = 0;
                std::vector<cv::Point2f> imgPoints;
                std::vector<float> depths;
                RCLCPP_INFO(get_logger(), "Detected %i with conf %f", detection.classId(), detection.score());
                for (int i = -1; i < 2; i++)
                {
                    int u = (bbox.x + (i * 3)); // + bbox.width / 2;
                    if (u < depths.size().width)
                    {
                        for (int j = -1; j < 2; j++)
                        {
                            int v = (bbox.y + (j * 3)); // + bbox.height / 2;
                            if (v < depths.size().height)
                            {
                                float depth = depths.at<float>(u, v);
                                // RCLCPP_INFO(get_logger(), "DEPTH AT POINT (%d, %d): %f", u, v, depth);

                                if (!isnanf(depth) && depth > 0 && depth < 10)
                                {
                                    // RCLCPP_INFO(get_logger(), "ADDING DEPTH: %f", depth);
                                    depths.push_back(depth);
                                    imgPoints.push_back(cv::Point2f(u, v));
                                    totalPoints++;
                                    totalDepth += depth;
                                }
                            }
                        }
                    }
                }

                if (totalPoints > 0)
                {
                    float depth = totalDepth / totalPoints;
                    // cv::remap()
                    std::vector<cv::Point2f> origPoint;
                    origPoint.emplace_back(cv::Point2f(bbox.x, bbox.y));
                    std::vector<cv::Point2f> rays;

                    cv::undistortPoints(origPoint, ray, camera_matrix, dist_coeffs);

                    float norm = std::sqrt(std::pow(ray[0].x, 2) + std::pow(ray[0].y, 2) + 1.0);

                    cv::Point3f fixedRay = cv::Point3f(ray[0].x / norm * depth, ray[0].y / norm * depth, 1 / norm * depth);

                    RCLCPP_INFO(get_logger(), "DEPTH: %f", depth);

                    vision_msgs::msg::Detection3D detection3d;

                    detection3d.header.stamp = msg->header.stamp;
                    detection3d.header.frame_id = "zedm_left_optical_frame";

                    geometry_msgs::msg::Pose objPose;
                    vision_msgs::msg::ObjectHypothesisWithPose objHypo;

                    objHypo.hypothesis.class_id = detection.classId();
                    objHypo.pose.pose.position.x = fixedRay.x;
                    objHypo.pose.pose.position.y = fixedRay.y;
                    objHypo.pose.pose.position.z = fixedRay.z;

                    detection3d.results.push_back(objHypo);
                    detections3d.detections.push_back(detection3d);
                }
            }
        }
        detection_pub->publish(detections3d);
    }

    void
    depth_image_sub_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        cv_bridge::toCvShare(msg, "32FC1")->image.copyTo(depths);
        // RCLCPP_INFO(get_logger(), "MSG TYPE: %s", msg->encoding.c_str());

        if (depths.empty())
        {
            RCLCPP_WARN(get_logger(), "Empty image on topic %s", depth_image_sub->get_topic_name());
            return;
        }
    }

    void camera_info_sub_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &msg)
    {
        if (!gotCalibrationInfo)
        {
            camera_matrix = cv::Mat(cv::Size(3, 3), CV_64FC1, (void *)msg->k.data());
            gotCalibrationInfo = true;
        }
    }

    bool gotCalibrationInfo = false;
    cv::Mat camera_matrix = cv::Mat(cv::Size(3, 3), CV_64FC1);
    cv::Mat dist_coeffs = cv::Mat().zeros(cv::Size(1, 5), CV_32FC1);
    cv::Mat depths;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detection_pub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_image_sub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_image_sub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub;
    std::shared_ptr<yolov5::Detector> infer;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TensorrtWrapper>());
    rclcpp::shutdown();
}