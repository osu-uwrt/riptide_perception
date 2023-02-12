#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/exceptions.h>

static const std::string ROBOT_NAME = "tempest"; //TODO: dont hardcode

/**
 * ROS node that publishes "oriented" mapping frames based off of the parameters in oriented_frames.yaml.
 * Pairs of frames will be oriented to face the direction perpendicular to the line in the XY plane that connects them.
 * "Oriented" frames will be published under the name <frame name>_oriented
 */
class OrientedFramePublisher : public rclcpp::Node {
    public:
    OrientedFramePublisher()
     : rclcpp::Node("oriented_frame_publisher")
    {
        //declare params
        this->declare_parameter<double>("timer_period", 0);
        this->declare_parameter<std::vector<std::string>>("locked_frames", std::vector<std::string>());

        //set up TF peripherals
        this->tfBuffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        this->tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);
        this->tfBroadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        //set up timer
        double period = this->get_parameter("timer_period").as_double();
        if(period > 0) {
            timer = this->create_wall_timer(
                std::chrono::duration<double>(period), 
                std::bind(&OrientedFramePublisher::timerCb, this)
            );
        } else {
            throw std::runtime_error("Could not properly read parameters! Please ensure parameter file is specified correctly.");
        }
    }

    private:
    /**
     * @brief Looks up the transform from fromFrame to toFrame.
     * 
     * @param fromFrame The frame to look up from
     * @param toFrame the frame to look up to
     * @param transform Will be populated with the transform from fromFrame to toFrame.
     * @return true if the operation succeeds, false otherwise.
     */
    bool getTransform(const std::string& fromFrame, const std::string& toFrame, geometry_msgs::msg::TransformStamped& transform) {
        try {
            transform = tfBuffer->lookupTransform(toFrame, fromFrame, tf2::TimePointZero);
            return true;
        } catch(tf2::LookupException& ex) {
            RCLCPP_WARN(this->get_logger(), "Could not look up tranform from %s to %s.", fromFrame.c_str(), toFrame.c_str());
        }

        return false;
    }

    /**
     * @brief Converts a Quaternion to Euler angles.
     * 
     * @param orientation The quaternion to convert.
     * @return geometry_msgs::msg::Vector3 The equivalent rotation in rpy.
     */
    geometry_msgs::msg::Vector3 toRPY(geometry_msgs::msg::Quaternion orientation) {
        tf2::Quaternion tf2Orientation;
        tf2::fromMsg(orientation, tf2Orientation);

        geometry_msgs::msg::Vector3 rpy;
        tf2::Matrix3x3(tf2Orientation).getEulerYPR(rpy.z, rpy.y, rpy.x);
        return rpy;
    }

    /**
     * @brief Converts a rotation in Euler angles to a quaternion.
     * 
     * @param rpy The euler angles to convert.
     * @return geometry_msgs::msg::Quaternion The resulting rotation in quaternion.
     */
    geometry_msgs::msg::Quaternion toQuat(geometry_msgs::msg::Vector3 rpy) {
        tf2::Quaternion tf2Quat;
        tf2Quat.setRPY(rpy.x, rpy.y, rpy.z);
        tf2Quat.normalize();

        return tf2::toMsg(tf2Quat);
    }

    /**
     * @brief Applies a transform to a pose.
     * 
     * @param relative The pose to transform.
     * @param transform The transform to use.
     * @return geometry_msgs::msg::Pose the transformed pose.
     */
    geometry_msgs::msg::Pose doTransform(geometry_msgs::msg::Pose relative, geometry_msgs::msg::TransformStamped transform) {
        geometry_msgs::msg::Pose result;

        //rotate the position on the yaw based on the transform rotation to the object
        double yaw = toRPY(transform.transform.rotation).z;
        
        geometry_msgs::msg::Vector3 newRelative;
        newRelative.x = relative.position.x * cos(yaw) - relative.position.y * sin(yaw);
        newRelative.y = relative.position.x * sin(yaw) + relative.position.y * cos(yaw);
        
        //translate point to new frame
        geometry_msgs::msg::Vector3 translation = transform.transform.translation;
        result.position.x = newRelative.x + translation.x;
        result.position.y = newRelative.y + translation.y;
        result.position.z = relative.position.z + translation.z;

        //transform the orientation quaternions
        tf2::Quaternion transformQuat;
        tf2::fromMsg(transform.transform.rotation, transformQuat);

        tf2::Quaternion relativeQuat;
        tf2::fromMsg(relative.orientation, relativeQuat);

        tf2::Quaternion resultQuat = relativeQuat * transformQuat;
        resultQuat.normalize();
        result.orientation = tf2::toMsg(resultQuat);

        return result;
    }

    /**
     * @brief Called when timer elapses.
     */
    void timerCb() {
        auto frames = this->get_parameter("locked_frames").as_string_array();

        if(frames.size() % 2 != 0) {
            throw std::runtime_error(
                "The locked_frames parameter must be set to an array with an even number of elements!");
        }

        for(uint i = 0; i < frames.size(); i += 2) {
            std::string
                frame1 = frames.at(i),
                frame2 = frames.at(i + 1);
            
            geometry_msgs::msg::TransformStamped f1ToMap, f2ToMap, mapToF1, mapToF2, robotToMap;
            bool
                f12M = getTransform(frame1, "map", f1ToMap),
                f22M = getTransform(frame2, "map", f2ToMap),
                m2F1 = getTransform("map", frame1, mapToF1),
                m2F2 = getTransform("map", frame2, mapToF2),
                r2F1 = getTransform(ROBOT_NAME + "/base_link", "map", robotToMap);

            //if frames were looked up successfully...
            if(f12M && f22M && m2F1 && m2F2) {
                //figure out yaw of oriented frames
                double 
                    frameXDiff = f2ToMap.transform.translation.x - f1ToMap.transform.translation.x,
                    frameYDiff = f2ToMap.transform.translation.y - f1ToMap.transform.translation.y,
                    thetaMap = atan2(frameYDiff, frameXDiff) + (M_PI / 2); //angle between two props plus 90 degrees

                //convert thetas from world frame to their respective object frames
                geometry_msgs::msg::Vector3 f1Rpy, f2Rpy;
                f1Rpy.z = thetaMap;
                f2Rpy.z = thetaMap;

                geometry_msgs::msg::Pose f1MapPose, f2MapPose;
                f1MapPose.orientation = toQuat(f1Rpy);
                f2MapPose.orientation = toQuat(f2Rpy);
                
                geometry_msgs::msg::Pose
                    f1RelativePose = doTransform(f1MapPose, mapToF1),
                    f2RelativePose = doTransform(f2MapPose, mapToF2);

                //assemble oriented transforms to send out
                rclcpp::Time now = this->get_clock()->now();

                geometry_msgs::msg::TransformStamped newF1Transform, newF2Transform;
                newF1Transform.header.stamp = now;
                newF1Transform.header.frame_id = frame1;
                newF1Transform.child_frame_id = frame1 + "_oriented";
                newF1Transform.transform.rotation = f1RelativePose.orientation;

                newF2Transform.header.stamp = now;
                newF2Transform.header.frame_id = frame2;
                newF2Transform.child_frame_id = frame2 + "_oriented";
                newF2Transform.transform.rotation = f2RelativePose.orientation;

                //publish new frames
                tfBroadcaster->sendTransform(newF1Transform);
                tfBroadcaster->sendTransform(newF2Transform);
            }
        }
    }

    rclcpp::TimerBase::SharedPtr timer;
    std::shared_ptr<tf2_ros::Buffer> tfBuffer;
    std::shared_ptr<tf2_ros::TransformListener> tfListener;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;
};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OrientedFramePublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
}
