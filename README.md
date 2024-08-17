# riptide_perception
Riptide Perception contains all of the neccessary code for handing the perception system onboard the Riptide AUV platform. This contains much of our custom vision code for handing vision based systems as well as the onboard context based mapping system. 

|            |              |
|------------|--------------|
| OS Distro  | Ubuntu 22.04 |
| ROS Distro | ROS2 Humble  |

## tensor_detector
Riptide Yolo is a custom designed wrapper around the Sterolabs ZED SDK. It serves as our 3d vision detection solution for localizing objects in the water. It uses the YOLOv8 segmentation detection network to find objects in the pool and the ZED SDK to solve its 3d pose relative to the camera. 

## riptide_mapping
Riptide Mapping is a custom mapping system designed to take in 3d vision based detections for object and apply location and context aware filterring to the data. It is able to fuse and merge position estimates for objects relative to a map frame in the world.
