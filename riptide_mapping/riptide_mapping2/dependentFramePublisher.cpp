#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/quaternion.h>
#include <geometry_msgs/msg/vector3.h>
#include <geometry_msgs/msg/transform_stamped.h>

using namespace std::chrono_literals;

std::vector<std::string> frameIds = {
    "gate_frame",
    "octogon_frame"
};

class DependentFramePublisher : public rclcpp::Node {
    public:
    DependentFramePublisher()
     : Node("dependent_frame_publisher") 
    {
        //declare parameters
        for(std::string id : frameIds) {
            declare_parameter<std::string>(id + ".parent", "");
            declare_parameter<std::string>(id + ".x", "");
            declare_parameter<std::string>(id + ".y", "");
            declare_parameter<std::string>(id + ".z", "");
            declare_parameter<std::string>(id + ".roll", "");
            declare_parameter<std::string>(id + ".pitch", "");
            declare_parameter<std::string>(id + ".yaw", "");
        }

        //initialize tf things
        tfBuffer = std::make_shared<tf2_ros::Buffer>(get_clock());
        tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);
        tfBroadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        //create and start timer
        timer = this->create_wall_timer(250ms, std::bind(&DependentFramePublisher::timerCB, this));
        RCLCPP_INFO(get_logger(), "Dependent frame publisher started.");
    }

    private:
    rclcpp::TimerBase::SharedPtr timer;
    std::shared_ptr<tf2_ros::Buffer> tfBuffer;
    std::shared_ptr<tf2_ros::TransformListener> tfListener;
    std::shared_ptr<tf2_ros::TransformBroadcaster >tfBroadcaster;

    /**
     * @brief Parses a string coordinate from the parameters. Expecting a string with format [number] [relative/absolute]
     * 
     * @param s The string containing the coordinate
     * @param value Will be populated with the value of the coordinate
     * @return true if the coordinate is absolute, false otherwise.
     */
    bool parseFrameCoordinate(const std::string& s, double& value) {
        value = 0;
        size_t space = s.find(' ');
        if(space != std::string::npos) {
            std::string 
                numStr = s.substr(0, space),
                absStr = s.substr(space + 1);
            
            try {
                value = std::stod(numStr);
                return absStr == "absolute";
            } catch(std::invalid_argument& ex) {
                RCLCPP_ERROR(get_logger(), "Error parsing coordinate %s! Got std::invalid argument: %s", s.c_str(), ex.what());
            }
        }

        return false;
    }

    /**
     * @brief Converts a rotation in Quaternion to Euler angles.
     * 
     * @param orientation The rotation to convert.
     * @return geometry_msgs::msg::Vector3 The resulting vec3
     */
    geometry_msgs::msg::Vector3 toRPY(geometry_msgs::msg::Quaternion orientation) {
        tf2::Quaternion tf2Orientation;
        tf2::fromMsg(orientation, tf2Orientation);

        geometry_msgs::msg::Vector3 rpy;
        tf2::Matrix3x3(tf2Orientation).getEulerYPR(rpy.z, rpy.y, rpy.x);
        return rpy;
    }

    /**
     * @brief Converts a rotation in Euler angles to Quaternion.
     * 
     * @param rpy The rotation to convert.
     * @return geometry_msgs::msg::Quaternion The resulting Quaternion.
     */
    geometry_msgs::msg::Quaternion toQuat(geometry_msgs::msg::Vector3 rpy) {
        tf2::Quaternion tf2Quat;
        tf2Quat.setRPY(rpy.x, rpy.y, rpy.z);
        tf2Quat.normalize();

        return tf2::toMsg(tf2Quat);
    }

    /**
     * @brief Multiplies to quaternions in order: q1 * q2
     * 
     * @param q1 the first quaternion
     * @param q2 the second quaternion
     * @return geometry_msgs::msg::Quaternion the resultant quaternion
     */
    geometry_msgs::msg::Quaternion quaternionMultiply(geometry_msgs::msg::Quaternion q1, geometry_msgs::msg::Quaternion q2) {
        tf2::Quaternion tfQ1;
        tf2::fromMsg(q1, tfQ1);

        tf2::Quaternion tfQ2;
        tf2::fromMsg(q2, tfQ2);

        tf2::Quaternion resultQuat = tfQ1 * tfQ2;
        resultQuat.normalize();

        return tf2::toMsg(resultQuat);
    }

    /**
     * @brief Assembles a Pose from XYZ and RPY values.
     * 
     * @param x x-coordinate
     * @param y y-coordinate
     * @param z z-coordinate
     * @param roll rotation along the x-axis
     * @param pitch rotation along the y-axis
     * @param yaw rotation along the z-axis
     * @return geometry_msgs::msg::Pose The resultant pose
     */
    geometry_msgs::msg::Pose poseFromXYZRPY(double x, double y, double z, double roll, double pitch, double yaw) {
        geometry_msgs::msg::Pose p;
        p.position.x = x;
        p.position.y = y;
        p.position.z = z;

        geometry_msgs::msg::Vector3 rpy;
        rpy.x = roll;
        rpy.y = pitch;
        rpy.z = yaw;

        p.orientation = toQuat(rpy);
        return p;
    }

    /**
     * @brief Applies absolute coordinates to the Pose p.
     * 
     * @param p The Pose to apply absolute coordinates to
     * @param absolutes An array containing booleans notifying absolute coordinates. The order is [xAbs, yAbs, zAbs, rollAbs, pitchAbs, yawAbs]
     * @param absVals A Pose. if a coordinate is absolute its value will be set to the cooresponding value from this structure
     * @return geometry_msgs::msg::Pose A new Pose with absolute coordinates applied.
     */
    geometry_msgs::msg::Pose applyAbsoluteCoordinates(geometry_msgs::msg::Pose p, bool absolutes[6], geometry_msgs::msg::Pose absVals) {
        //easy part: apply absolute coordinates on position
        p.position.x = (absolutes[0] ? absVals.position.x : p.position.x);
        p.position.y = (absolutes[1] ? absVals.position.y : p.position.y);
        p.position.z = (absolutes[2] ? absVals.position.z : p.position.z);

        //slightly less easy part: rotation
        auto pRpy = toRPY(p.orientation);
        auto absRpy = toRPY(absVals.orientation);

        pRpy.x = (absolutes[3] ? absRpy.x : pRpy.x);
        pRpy.y = (absolutes[4] ? absRpy.y : pRpy.y);
        pRpy.z = (absolutes[5] ? absRpy.z : pRpy.z);

        p.orientation = toQuat(pRpy);

        return p;
    }

    /**
     * @brief Called when the timer elapses
     */
    void timerCB() {
        for(std::string id : frameIds) {
            std::string
                parent = get_parameter(id + ".parent").as_string(),
                xStr = get_parameter(id + ".x").as_string(),
                yStr = get_parameter(id + ".y").as_string(),
                zStr = get_parameter(id + ".z").as_string(),
                rollStr = get_parameter(id + ".roll").as_string(),
                pitchStr = get_parameter(id + ".pitch").as_string(),
                yawStr = get_parameter(id + ".yaw").as_string();
            
            double x, y, z, roll, pitch, yaw;
            bool
                xAbs = parseFrameCoordinate(xStr, x),
                yAbs = parseFrameCoordinate(yStr, y),
                zAbs = parseFrameCoordinate(zStr, z),
                rollAbs = parseFrameCoordinate(rollStr, roll),
                pitchAbs = parseFrameCoordinate(pitchStr, pitch),
                yawAbs = parseFrameCoordinate(yawStr, yaw);
            
            bool absoluteCoordinates[6] { xAbs, yAbs, zAbs, rollAbs, pitchAbs, yawAbs };
            auto coordinates = poseFromXYZRPY(x, y, z, roll, pitch, yaw);
            
            try {
                //look up transform from parent frame to world
                auto parentToWorld = tfBuffer->lookupTransform("world", parent, tf2::TimePointZero);
                
                //initialize relative pose as the raw coordinates in the parameters
                geometry_msgs::msg::Pose relPose = coordinates;

                //zero out absolute coordinates
                relPose = applyAbsoluteCoordinates(relPose, absoluteCoordinates, geometry_msgs::msg::Pose());

                //transform the relative pose to world frame
                geometry_msgs::msg::Pose transformedPose;
                tf2::doTransform(relPose, transformedPose, parentToWorld);

                //apply absolute coordinates
                transformedPose = applyAbsoluteCoordinates(transformedPose, absoluteCoordinates, coordinates);

                //assemble transform to broadcast
                geometry_msgs::msg::TransformStamped childTransform;
                childTransform.header.frame_id = "world",
                childTransform.header.stamp = get_clock()->now();
                childTransform.child_frame_id = id;
                childTransform.transform.translation.x = transformedPose.position.x;
                childTransform.transform.translation.y = transformedPose.position.y;
                childTransform.transform.translation.z = transformedPose.position.z;
                childTransform.transform.rotation = transformedPose.orientation;

                tfBroadcaster->sendTransform(childTransform);
            } catch(tf2::TransformException& ex) {
                RCLCPP_WARN(get_logger(), "Could not look up transform from world to %s: %s", parent.c_str(), ex.what());
            }
        }
    }
};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DependentFramePublisher>());
    rclcpp::shutdown();
}
