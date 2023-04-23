#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/imgproc.hpp>

namespace video_compression {

    class Compressor {
        public:
        Compressor() : desiredWidth(0) { };
        Compressor(rclcpp::Node::SharedPtr node, image_transport::ImageTransport& it, const std::string& inputTopic, const std::string& outputTopic, const int desiredWidth);

        private:
        void callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

        const int desiredWidth;
        rclcpp::Node::SharedPtr node;
        image_transport::Subscriber sub;
        image_transport::Publisher pub;
    };

} //namespace video_compression
