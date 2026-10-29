# Original RGB image capture timestamp and camera optical frame.
std_msgs/Header header

# Individual instances from that image. Empty when none are detected.
riptide_perception_msgs/InstanceSegmentation[] instances
