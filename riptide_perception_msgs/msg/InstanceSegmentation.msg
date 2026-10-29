# Detection metadata and mask for one object instance.
# Detection and mask headers must match the source RGB header.
vision_msgs/Detection2D detection

# Full-resolution mono8 mask: background=0, object=255.
sensor_msgs/Image mask
