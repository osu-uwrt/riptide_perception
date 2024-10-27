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
        imDetPublisher = this->create_publisher<sensor_msgs::msg::Image>("yolov8Detections", rclcpp::SystemDefaultsQoS());
        imageSubscriber = this->create_subscription<sensor_msgs::msg::Image>("zed/zed_node/left/image_rect_color", qos, std::bind(&zedYolo::imageCallback, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "sub topic: %s", imageSubscriber->get_topic_name());

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

        // for (std::string name : classNames) {
        //     RCLCPP_INFO(get_logger(), "Name: %s", name.c_str());
        // }
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

        std::vector<int> blueVec, greenVec, redVec;

        for (int i = 0; i < classNames.size(); i++)
        {
            blueVec.push_back(blue(blueGenerator));
            greenVec.push_back(green(greenGenerator));
            redVec.push_back(red(redGenerator));
        }

        std::default_random_engine rng;
        std::shuffle(std::begin(blueVec), std::end(blueVec), rng);
        std::shuffle(std::begin(greenVec), std::end(greenVec), rng);
        std::shuffle(std::begin(redVec), std::end(redVec), rng);

        for (int i = 0; i < classNames.size(); i++)
        {
            colors.push_back(cv::Scalar(blueVec[i], greenVec[i], redVec[i]));
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

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv::Mat inputIm = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
        cv::Mat outputIm(inputIm.size(), inputIm.type());
        RCLCPP_INFO(get_logger(), "Created output image");
        inputIm.copyTo(outputIm);
        RCLCPP_INFO(get_logger(), "copied to output image");
        std::vector<Detection> detections = detector->runDetection(inputIm);
        RCLCPP_INFO(get_logger(), "Detected image");

        for (Detection detection : detections)
        {
            std::string label = classNames[detection.classId()];
            label += ": " + std::to_string(detection.confidence()).substr(0, 4);
            cv::rectangle(outputIm, detection.bbox(), colors[detection.classId()], 5);
            cv::rectangle(outputIm, cv::Rect(detection.bbox().x, detection.bbox().y, label.size() * 11, 30), colors[detection.classId()], -1);
            cv::putText(outputIm, label, cv::Point(detection.bbox().x + 10, detection.bbox().y + 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
            cv::Mat segMask = detection.mask();
            segMask.setTo(colors[detection.classId()], segMask == cv::Scalar(255, 255, 255));
            cv::addWeighted(outputIm, 1., segMask, 0.5, 0, outputIm);
            // cv::Mat mask = cv::Mat::zeros(segMask.size(), CV_8UC1);
            // mask(detection.bbox()).setTo(255);

            // segMask.setTo(0, ~mask);

            // cv::cvtColor(segMask, segMask, cv::COLOR_GRAY2BGR);
            // segMask.setTo(colors[detection.classId()], segMask);
        }

        // RCLCPP_WARN(get_logger(), "Num Detections: %d", detections.size());

        imgBridge = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, outputIm);

        sensor_msgs::msg::Image::SharedPtr imgMsg(imgBridge.toImageMsg());
        imDetPublisher->publish(*imgMsg.get());
        RCLCPP_INFO(get_logger(), "finished callback");
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<zedYolo>());
    rclcpp::shutdown();
    return 0;
}