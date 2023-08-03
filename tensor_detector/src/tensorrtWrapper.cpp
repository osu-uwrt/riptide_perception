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
#include "eigen3/Eigen/Geometry"

#include "tensorrt_detector/yolov5_detector.hpp"

class TensorrtWrapper : public rclcpp::Node
{
public:
    TensorrtWrapper()
        : Node("tensorrt_wrapper")
    {

        std::string engine_file = "/home/coalman321/yolo.engine";
        declare_parameter("engine_path", rclcpp::ParameterValue(engine_file));
        engine_file = get_parameter("engine_path").as_string();

        RCLCPP_WARN(get_logger(), "ENGINE FILE PATH: %s", engine_file.c_str());

        // init and load the detector
        infer = std::make_shared<yolov5::Detector>();
        infer->init();
        infer->loadEngine(engine_file);
        infer->setScoreThreshold(0.3);

        // rcl_interfaces::msg::ParameterDescriptor descriptor;
        declare_parameter("obj_ids", rclcpp::ParameterValue(objIds));
        objIds = get_parameter("obj_ids").as_string_array();

        // create the pub for detections
        detection_pub = this->create_publisher<vision_msgs::msg::Detection3DArray>("detected_objects", rclcpp::SystemDefaultsQoS());

        bbox_pub = this->create_publisher<sensor_msgs::msg::Image>("image_with_bboxes", rclcpp::SystemDefaultsQoS());

        // create the two image subs we need to consume
        left_image_sub = this->create_subscription<sensor_msgs::msg::Image>("zed2i/zed_node/left/image_rect_color",
                                                                            rclcpp::SystemDefaultsQoS(), std::bind(&TensorrtWrapper::left_image_sub_callback, this, std::placeholders::_1));

        depth_image_sub = this->create_subscription<sensor_msgs::msg::Image>("zed2i/zed_node/depth/depth_registered",
                                                                             rclcpp::SystemDefaultsQoS(), std::bind(&TensorrtWrapper::depth_image_sub_callback, this, std::placeholders::_1));

        camera_info_sub = this->create_subscription<sensor_msgs::msg::CameraInfo>("zed2i/zed_node/left/camera_info",
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

        cv::Mat copyFrame;
        frame.copyTo(copyFrame);

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
            // RCLCPP_INFO(get_logger(), "HERE");
            // process each of the detections
            for (yolov5::Detection detection : detections)
            {
                if (detection.score() > .7)
                {
                    int totalPoints = 0;
                    float totalDepths = 0;
                    std::vector<float> sampledDepths;
                    std::vector<cv::Point2f> imgPoints;
                    cv::Rect bbox = detection.boundingBox();

                    cv::rectangle(copyFrame, bbox, cv::Scalar(0, 0, 255));
                    // cv::putText(copyFrame, std::to_string(detection.classId()).c_str(), cv::Point(bbox.x, bbox.y), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255), 2);
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
                                    float depth = depths.at<float>(v, u);
                                    // RCLCPP_INFO(get_logger(), "DEPTH AT POINT (%d, %d): %f", u, v, depth);

                                    if (!isnanf(depth) && depth > 0 && depth < 10)
                                    {
                                        // RCLCPP_INFO(get_logger(), "ADDING DEPTH: %f", depth);
                                        sampledDepths.push_back(depth);
                                        imgPoints.push_back(cv::Point2f(v, u));
                                        totalPoints++;
                                        totalDepths += depth;
                                    }
                                }
                            }
                        }
                    }

                    if (totalPoints > 0)
                    {
                        float sampledDepth = totalDepths / totalPoints;

                        std::vector<cv::Point2f> centerPoint;
                        std::vector<cv::Point2f> ray;
                        centerPoint.push_back(cv::Point2f(bbox.x, bbox.y));
                        cv::undistortPoints(centerPoint, ray, camera_matrix, dist_coeffs);

                        // RCLCPP_INFO(get_logger(), "Depth: %f", sampledDepth);
                        // RCLCPP_INFO(get_logger(), "CenterPoint x, y: %d, %d", bbox.x, bbox.y);
                        // RCLCPP_INFO(get_logger(), "Ray x, y: %f, %f", ray[0].x, ray[0].y);
                        // RCLCPP_INFO(get_logger(), "Num depths: %d", totalPoints);

                        float singleNorm = std::sqrt(std::pow(ray[0].x, 2) + std::pow(ray[0].y, 2) + 1.0);

                        cv::Point3f fixedRay(ray[0].x / singleNorm * sampledDepth, ray[0].y / singleNorm * sampledDepth, 1.0 / singleNorm * sampledDepth);

                        vision_msgs::msg::Detection3D detection3d;

                        detection3d.header.stamp = msg->header.stamp;
                        detection3d.header.frame_id = "zed2i_left_optical_frame";

                        vision_msgs::msg::ObjectHypothesisWithPose objHypo;

                        objHypo.hypothesis.class_id = objIds.at(detection.classId());
                        objHypo.pose.pose.position.x = fixedRay.x;
                        objHypo.pose.pose.position.y = fixedRay.y;
                        objHypo.pose.pose.position.z = fixedRay.z;

                        if (totalPoints > 3)
                        {
                            std::vector<cv::Point2f> rays;

                            cv::undistortPoints(imgPoints, rays, camera_matrix, dist_coeffs);

                            std::vector<cv::Point3f> cvPoints3d;

                            Eigen::MatrixXf points3d(3, totalPoints);

                            cv::Point3f centroid;

                            for (int i = 0; i < rays.size(); i++)
                            {
                                float norm = std::sqrt(std::pow(rays[i].x, 2) + std::pow(rays[i].y, 2) + 1.0);
                                centroid.x += rays[i].x / norm * sampledDepths[i];
                                centroid.y += rays[i].y / norm * sampledDepths[i];
                                centroid.z += 1.0 / norm * sampledDepths[i];
                                RCLCPP_INFO(get_logger(), "Depth at point: %f", sampledDepths[i]);
                                // RCLCPP_INFO(get_logger(), "ADDED POINT: %f, %f, %f", points3d(0, i), points3d(1, i), points3d(2, i));
                            }
                            centroid.x /= totalPoints;
                            centroid.y /= totalPoints;
                            centroid.z /= totalPoints;

                            for (int i = 0; i < totalPoints; i++)
                            {
                                float norm = std::sqrt(std::pow(rays[i].x, 2) + std::pow(rays[i].y, 2) + 1.0);
                                points3d(0, i) = (rays[i].x / norm * sampledDepths[i]) - centroid.x;
                                points3d(1, i) = (rays[i].y / norm * sampledDepths[i]) - centroid.y;
                                points3d(2, i) = (1.0 / norm * sampledDepths[i]) - centroid.z;
                            }

                            Eigen::JacobiSVD svd(points3d, Eigen::ComputeFullU);
                            Eigen::Matrix3f uMat = svd.matrixU();

                            Eigen::Vector3f camNormal = Eigen::Vector3f(0, 0, 1);

                            Eigen::Vector3f objNormal = Eigen::Vector3f(uMat(0, 2), uMat(1, 2), uMat(2, 2));

                            Eigen::Vector3f product = camNormal.cross(objNormal);

                            Eigen::Vector4f nonnormQuaternion(0.0, product.x(), product.y(), product.z());

                            nonnormQuaternion.w() = std::sqrt(camNormal.norm() * objNormal.norm() + camNormal.dot(objNormal));

                            // RCLCPP_INFO(get_logger(), "W: %f", std::sqrt(camNormal.norm() * objNormal.norm() + camNormal.dot(objNormal)));
                            // orientationVector.angle() = 0.0;
                            // orientationVector.axis().x() = uMat(0, 2);
                            // orientationVector.axis().y() = uMat(1, 2);
                            // orientationVector.axis().z() = uMat(2, 2);
                            RCLCPP_WARN(get_logger(), "Axis Angles: %f, %f, %f", objNormal.x(), objNormal.y(), objNormal.z());

                            Eigen::Quaternionf q;

                            objHypo.pose.pose.orientation.w = nonnormQuaternion.w() / nonnormQuaternion.norm();
                            objHypo.pose.pose.orientation.x = nonnormQuaternion.x() / nonnormQuaternion.norm();
                            objHypo.pose.pose.orientation.y = nonnormQuaternion.y() / nonnormQuaternion.norm();
                            objHypo.pose.pose.orientation.z = nonnormQuaternion.z() / nonnormQuaternion.norm();
                            RCLCPP_INFO(get_logger(), "Q: %f, %f, %f, %f", objHypo.pose.pose.orientation.w,
                                        objHypo.pose.pose.orientation.x,
                                        objHypo.pose.pose.orientation.y,
                                        objHypo.pose.pose.orientation.z);
                        }
                        else
                        {
                            objHypo.pose.pose.orientation.w = 2.0;
                            objHypo.pose.pose.orientation.x = 2.0;
                            objHypo.pose.pose.orientation.y = 2.0;
                            objHypo.pose.pose.orientation.z = 2.0;
                        }
                        detection3d.results.push_back(objHypo);
                        detections3d.detections.push_back(detection3d);
                    }
                }
            }
            sensor_msgs::msg::Image bboxedImage;
            std_msgs::msg::Header hdr;
            cv_bridge::CvImage(hdr, "bgr8", copyFrame).toImageMsg(bboxedImage);
            bbox_pub->publish(bboxedImage);
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
            cv::Mat(cv::Size(3, 3), CV_64FC1, (void *)msg->k.data()).convertTo(camera_matrix, CV_32FC1);
            gotCalibrationInfo = true;
        }
    }

    bool gotCalibrationInfo = false;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs = cv::Mat().zeros(cv::Size(1, 5), CV_32FC1);
    cv::Mat depths;
    std::vector<std::string> objIds;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detection_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr bbox_pub;
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