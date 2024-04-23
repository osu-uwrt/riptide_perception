#include <rclcpp/rclcpp.hpp>
#include <riptide_mapping/mapping.hpp>


int main(int argc, char* argv[]) {

    rclcpp::init(argc, argv);

    rclcpp::Node::SharedPtr mapping_node = std::make_shared<MappingNode>();

    rclcpp::executors::MultiThreadedExecutor exec;

    exec.add_node(mapping_node);

    exec.spin();

    return 0;
}