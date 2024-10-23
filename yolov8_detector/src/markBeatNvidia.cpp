#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <riptide_msgs2/msg/image_array.hpp>
#include <cv_bridge/cv_bridge.h>

#include <random>

#include "yolov8seg.h"

class zedYolo : public rclcpp::Node
{
public:
    zedYolo() : Node("zed_yolo")
    {
        auto qos = rclcpp::SensorDataQoS();

        // maskPublisher = this->create_publisher<riptide_msgs2::msg::ImageArray>("yolov8Detections", 10);
        imDetPublisher = this->create_publisher<sensor_msgs::msg::Image>("yolov8Detections", 10);
        imageSubscriber = this->create_subscription<sensor_msgs::msg::Image>("zed/zed_node/left/image_rect_color", qos, std::bind(&zedYolo::imageCallback, this, std::placeholders::_1));

        this->declare_parameter("model_path", rclcpp::ParameterValue("weights/yolov8n-seg.trt"));
        this->declare_parameter("conf_threshold", rclcpp::ParameterValue(0.8));
        this->declare_parameter("iou_threshold", rclcpp::ParameterValue(0.8));
        this->declare_parameter("class_names", rclcpp::ParameterValue(classNames));

        filename = this->get_parameter("model_path").as_string();
        confThreshold = this->get_parameter("conf_threshold").as_double();
        iouThreshold = this->get_parameter("iou_threshold").as_double();

        RCLCPP_WARN(get_logger(), "Filename: %s", filename.c_str());
        RCLCPP_WARN(get_logger(), "Confidence threshold: %f", confThreshold);
        RCLCPP_WARN(get_logger(), "IOU threshold: %f", iouThreshold);

        classNames = this->get_parameter("class_names").as_string_array();

        detector = std::make_shared<YoloV8Detector>(filename, confThreshold, iouThreshold);

        if (classNames.size() != detector->numClasses())
        {
            throw std::runtime_error("Class name and model class number mismatch");
        }

        std::default_random_engine blueGenerator;
        std::default_random_engine greenGenerator;
        std::default_random_engine redGenerator;

        std::uniform_int_distribution<int> blue(0, 255);
        std::uniform_int_distribution<int> green(0, 255);
        std::uniform_int_distribution<int> red(0, 255);

        for (int i = 0; i < classNames.size(); i++)
        {
            colors.push_back(cv::Scalar(blue(blueGenerator), green(greenGenerator), red(redGenerator)));
        }

        RCLCPP_INFO(get_logger(), "Initialized yolov8 detector");
        RCLCPP_INFO(get_logger(), "Num classes: %d", detector->numClasses());
    }

private:
    rclcpp::Publisher<riptide_msgs2::msg::ImageArray>::SharedPtr maskPublisher;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr imDetPublisher;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSubscriber;

    std::shared_ptr<YoloV8Detector> detector;

    std::string filename;
    std::vector<std::string> classNames;
    std::vector<cv::Scalar> colors;
    double confThreshold;
    double iouThreshold;

    cv_bridge::CvImage imgBridge;
    sensor_msgs::msg::Image imgMsg;

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv::Mat inputIm = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
        cv::Mat outputIm;
        inputIm.copyTo(outputIm);
        std::vector<Detection> detections = detector->runDetection(inputIm);

        for (Detection detection : detections)
        {
            cv::rectangle(outputIm, detection.bbox(), cv::Scalar(255, 0, 0));
            // cv::Mat segMask = detection.mask();
            // cv::Mat mask = cv::Mat::zeros(segMask.size(), CV_8UC1);
            // mask(detection.bbox()).setTo(255);

            // segMask.setTo(0, ~mask);

            // cv::cvtColor(segMask, segMask, cv::COLOR_GRAY2BGR);
            // segMask.setTo(colors[detection.classId()], segMask);
        }

        // RCLCPP_WARN(get_logger(), "Num Detections: %d", detections.size());
        
        imgBridge = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, outputIm);
        imgBridge.toImageMsg(imgMsg);
        imDetPublisher->publish(imgMsg);
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<zedYolo>());
    rclcpp::shutdown();
    return 0;
}