#include "video_compression/compression.hpp"

using namespace std::placeholders;

const std_msgs::msg::Header hdr;
namespace video_compression {

    Compressor::Compressor(const rclcpp::Node::SharedPtr node, image_transport::ImageTransport& it, const std::string& inputTopic, const std::string& outputTopic, const int desiredWidth) 
    : desiredWidth(desiredWidth),
      node(node)
    {
        sub = it.subscribe(inputTopic, 1, &Compressor::callback, this);
        pub = it.advertise(outputTopic, 1);
    }

    void Compressor::callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        try {
            cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
            if(img.empty()) {
                RCLCPP_WARN(node->get_logger(), "Empty image on topic %s", sub.getTopic().c_str());
                return;
            }
            
            //figure out size to resize to based on aspect ratio and desired width
            cv::Size sz;
            sz.width = desiredWidth;
            sz.height = (int) (img.rows * (desiredWidth / (double) img.cols));

            //resize image
            cv::resize(img, img, sz);

            //convert back to correct type and publish
            std::shared_ptr<sensor_msgs::msg::Image> out = cv_bridge::CvImage(hdr, "bgr8", img).toImageMsg();
            pub.publish(out);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(node->get_logger(), "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
        }
    }

} // namespace video_compression
