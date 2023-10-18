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
#include <opencv2/opencv.hpp>

#include "eigen3/Eigen/SVD"
#include "eigen3/Eigen/Geometry"

class TensorrtWrapper : public rclcpp::Node
{
public:
    TensorrtWrapper()
        : Node("tensorrt_wrapper")
    {

        // create the pub for detections
        detection_pub = this->create_publisher<vision_msgs::msg::Detection3DArray>("detected_objects", rclcpp::SystemDefaultsQoS());
        std::string tempFile = "/home/yiyao/template_matching/template/buoy_pic_left.png";
        templateImg = cv::imread(tempFile, cv::IMREAD_GRAYSCALE);
        bbox_pub = this->create_publisher<sensor_msgs::msg::Image>("det_image_bboxed", rclcpp::SystemDefaultsQoS());

        // create the two image subs we need to consume
        left_image_sub = this->create_subscription<sensor_msgs::msg::Image>("zed2i/zed_node/left/image_rect_color",
                                                                            rclcpp::SystemDefaultsQoS(), std::bind(&TensorrtWrapper::left_image_sub_callback, this, std::placeholders::_1));

        depth_image_sub = this->create_subscription<sensor_msgs::msg::Image>("zed2i/zed_node/depth/depth_registered",
                                                                             rclcpp::SystemDefaultsQoS(), std::bind(&TensorrtWrapper::depth_image_sub_callback, this, std::placeholders::_1));
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

        cv::Mat copyFrame;
        frame.copyTo(copyFrame);

        std::vector<double> scales = {0.7, 0.9, 1.0, 1.2, 1.3};
        cv::Mat grayFrame;

        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        double best_min_val = DBL_MAX;
        cv::Point best_top_left;
        int best_w = 0;
        int best_h = 0;

        for (double scale : scales)
        {
            cv::Mat resizedTemplate;
            cv::resize(templateImg, resizedTemplate, cv::Size(), scale, scale);

            if (resizedTemplate.rows > grayFrame.rows || resizedTemplate.cols > grayFrame.cols)
                continue;

            cv::Mat res;
            cv::matchTemplate(grayFrame, resizedTemplate, res, cv::TM_SQDIFF);

            double min_val;
            cv::Point min_loc;
            cv::minMaxLoc(res, &min_val, nullptr, &min_loc, nullptr);

            if (min_val < best_min_val)
            {
                best_min_val = min_val;
                best_top_left = min_loc;
                best_w = resizedTemplate.cols;
                best_h = resizedTemplate.rows;
            }
        }

        cv::rectangle(copyFrame, best_top_left, cv::Point(best_top_left.x + best_w, best_top_left.y + best_h), 255, 2);
        cv::putText(copyFrame, std::to_string(best_min_val), cv::Point(best_top_left.x + best_w, best_top_left.y + best_h), cv::FONT_HERSHEY_COMPLEX, 1.0, 255, 2, cv::LINE_AA);

        sensor_msgs::msg::Image bboxedImage;
        std_msgs::msg::Header hdr;
        cv_bridge::CvImage(hdr, "bgr8", copyFrame).toImageMsg(bboxedImage);
        bbox_pub->publish(bboxedImage);
    }
    void depth_image_sub_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
    }

    bool gotCalibrationInfo = false;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs = cv::Mat().zeros(cv::Size(1, 5), CV_32FC1);
    cv::Mat depths;
    cv::Mat templateImg;
    std::vector<std::string> objIds;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr bbox_pub;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detection_pub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_image_sub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_image_sub;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TensorrtWrapper>());
    rclcpp::shutdown();
}