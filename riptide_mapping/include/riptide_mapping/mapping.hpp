#include <chrono>
#include <rclcpp/rclcpp.hpp>

using namespace std::chrono_literals;

/*
Subscribes to Detections3DArray Topic and determines approximate location using
GTSAM and publishes to TF.
*/
class MappingNode : public rclcpp::Node {
    public:
    // Initalizes MappingNode
    MappingNode();
    // Deconstructs MappingNode
    ~MappingNode();

    private:
    // Callback when any Detections3DArray is recieved
    void DetectionCallback();
};