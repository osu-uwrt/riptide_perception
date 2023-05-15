#include "video_compression/compression.hpp"
#include <vector>

std::string fullTopicName(rclcpp::Node::SharedPtr node, const std::string& topic) {
    if(topic.at(0) == '/') {
        return topic;
    }

    return node->get_namespace() + topic;
}

namespace video_compression {

    bool hasParametersForVideo(rclcpp::Node::SharedPtr node, int idx) {
        std::string prefix = "video" + std::to_string(idx);
        return
            node->get_parameter(prefix + ".input_topic").as_string().length() > 0
            && node->get_parameter(prefix + ".output_topic").as_string().length() > 0
            && node->get_parameter(prefix + ".desired_width").as_int() > 0;
    }

    int declareStreams(rclcpp::Node::SharedPtr node) {
        int videoIdx = 0;

        node->declare_parameter("video0.input_topic", "");
        node->declare_parameter("video0.output_topic", "");
        node->declare_parameter("video0.desired_width", 0);
        node->declare_parameter("video0.max_fps", 1);

        while(hasParametersForVideo(node, videoIdx)) {
            videoIdx++;

            //declare parameters for next video
            std::string prefix = "video" + std::to_string(videoIdx);
            node->declare_parameter(prefix + ".input_topic", "");
            node->declare_parameter(prefix + ".output_topic", "");
            node->declare_parameter(prefix + ".desired_width", 0);
            node->declare_parameter(prefix + ".max_fps", 1);
        }

        //the value of videoIdx is the number of video streams defined in the config
        return videoIdx;
    }

    void createCompressors(rclcpp::Node::SharedPtr node, image_transport::ImageTransport& it, std::shared_ptr<Compressor> *compressors, const int numStreams) {
        for(int i = 0; i < numStreams; i++) {
            std::string prefix = "video" + std::to_string(i);

            //get parameters for this stream
            std::string
                inputTopic  = node->get_parameter(prefix + ".input_topic").as_string(),
                outputTopic = node->get_parameter(prefix + ".output_topic").as_string();
            int
                desiredWidth = node->get_parameter(prefix + ".desired_width").as_int(),
                maxFps       = node->get_parameter(prefix + ".max_fps").as_int();

            RCLCPP_INFO(node->get_logger(), 
                "Streaming %s to %s at fixed width %i", 
                fullTopicName(node, inputTopic).c_str(), 
                fullTopicName(node, outputTopic).c_str(), 
                desiredWidth
            );

            compressors[i] = std::make_shared<Compressor>(node, it, inputTopic, outputTopic, desiredWidth, maxFps);
        }
    }

} //namespace video_compression


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("video_compressor");
    image_transport::ImageTransport it(node);
    const int numStreams = video_compression::declareStreams(node);
    if(numStreams > 0) {
        std::shared_ptr<video_compression::Compressor> *compressors = new std::shared_ptr<video_compression::Compressor>[numStreams];
        video_compression::createCompressors(node, it, compressors, numStreams);
        rclcpp::spin(node);
        delete[] compressors;
    } else {
        RCLCPP_WARN(node->get_logger(), "No streams found in parameters.");
    }

    rclcpp::shutdown();
}
