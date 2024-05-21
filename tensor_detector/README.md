# Tensor Detector

## Overview
The `tensor_detector` package is part of `riptide_perception` and contains a node designed for object detection and pose/orientation estimation. The node utilizes a YOLOv8 segmentation model to perform real-time detection and localization of objects using RGB and depth data from a ZED camera.

## Common Configuration
The common configuration for this package is specified in `tensorrt.yaml` and includes parameters used by the node.

### Configuration Parameters
| Parameter Name   | Type                | Description                                      |
|------------------|---------------------|--------------------------------------------------|
| `yolo_model_path`| `string`            | Path to the YOLO model file.                     |
| `class_id_map`   | `dict<int, string>` | Mapping of class IDs to their respective names.  |
| `threshold`      | `float`             | Confidence threshold for object detection.       |
| `iou`            | `float`             | Intersection over Union threshold for object detection.|

## Dependencies
List all the dependencies required by the package.

### Dependencies List
- `rclpy`
- `sensor_msgs`
- `cv_bridge`
- `cv2`
- `ultralytics`
- `numpy`
- `visualization_msgs`
- `vision_msgs`
- `geometry_msgs`
- `scipy`
- `collections`
- `time`
- `os`

## Usage
The node in this package is launched by the `riptide_launch` service. For more details, refer to [riptide_launch](https://github.com/osu-uwrt/riptide_launch).

## Diagnostics and Monitoring

- **Check Camera Output**: Ensure there is camera output at `/talos/zed/zed_node/left/image_rect_color` using:
```bash
ros2 topic echo /talos/zed/zed_node/left/image_rect_color
```

- **Check YOLO Output**: Verify the annotated images are published at `/talos/yolo` using:
```bash
ros2 topic echo /talos/yolo
```

- **Check Detection Output**: Ensure the detections are published at `/talos/detected_objects` using:
```bash
ros2 topic echo /talos/detected_objects
```

## Troubleshooting
### Problem: Inaccurate Detections
- **Solution**: Check the model in the configuration file to ensure it is correct and properly set up.

### Problem: Orientation Slightly Off
- **Solution**: Ensure the ZED camera is using the correct calibration file.

## Node

### YOLO Orientation Node

#### Overview
The `yolo_orientation` node utilizes a YOLOv8 segmentation model for real-time object detection, tracking, and 3D localization using images and depth data from a ZED camera. It processes both RGB and depth data to detect and localize objects.

#### ROS2 Interfaces

**Parameters**
| Parameter Name   | Type                | Default  | Description                                            |
|------------------|---------------------|----------|--------------------------------------------------------|
| `yolo_model_path`| `string`            | N/A      | Path to the YOLO model file.                           |
| `class_id_map`   | `dict<int, string>` | N/A      | Mapping of class IDs to their respective names.        |
| `threshold`      | `float`             | 0.9      | Confidence threshold for object detection.             |
| `iou`            | `float`             | 0.9      | Intersection over Union threshold for object detection.|

**Subscriptions**
| Subscription Topic                      | Message Type             | Description                                           |
|-----------------------------------------|--------------------------|-------------------------------------------------------|
| `/talos/zed/zed_node/left/camera_info`  | `sensor_msgs/CameraInfo` | Receives camera intrinsic parameters.                 |
| `/talos/zed/zed_node/depth/camera_info` | `sensor_msgs/CameraInfo` | Receives depth camera information.                    |
| `/talos/zed/zed_node/left/image_rect_color`| `sensor_msgs/Image`    | Receives the color image stream from the ZED camera.  |
| `/talos/zed/zed_node/depth/depth_registered`| `sensor_msgs/Image`   | Receives the depth image stream.                      |

**Publishers**
| Publisher Topic                    | Message Type                | Description                                         |
|------------------------------------|-----------------------------|-----------------------------------------------------|
| `/talos/visualization_marker`      | `visualization_msgs/Marker` | Publishes visualization markers for detected objects.|
| `/talos/visualization_marker_array`| `visualization_msgs/MarkerArray`| Publishes an array of visualization markers.|
| `/talos/yolo`                      | `sensor_msgs/Image`         | Publishes annotated images with detection results.  |
| `/talos/point_cloud`               | `sensor_msgs/PointCloud`    | Publishes point clouds of detected objects.         |
| `/talos/yolo_mask`                 | `sensor_msgs/Image`         | Publishes mask images for detected objects.         |
| `/talos/detected_objects`          | `vision_msgs/Detection3DArray` | Publishes detected objects with 3D positions and orientations.|

#### Functional Description
1. **Segmentation Model**:
   - Utilizes the YOLOv8 segmentation model to detect objects in the RGB image.
   - Converts `.pt` models to `.engine` if necessary.

2. **Feature Tracking**:
   - Uses `cv2.goodFeaturesToTrack` to identify 2D points within the bounding boxes of detected objects.
   - Converts 2D points to 3D points using the depth map from the ZED camera.

3. **Pose Estimation**:
   - Applies Singular Value Decomposition (SVD) to determine the orientation and pose of detected objects.
   - The bounding box center provides x/y coordinates of the centroid.
   - SVD provides the z-coordinate and orientation.

#### Data Flow
- **Input**: Subscribes to RGB and depth image streams, and camera information topics.
- **Processing**: Detects objects, tracks features, converts 2D points to 3D, and estimates pose using SVD.
- **Output**: Publishes annotated images, point clouds, and detected objects with their 3D positions and orientations.

#### Key Functions and Classes
| Function/Class                         | Description                                                  |
|----------------------------------------|--------------------------------------------------------------|
| `initialize_yolo(yolo_model_path)`     | Initializes the YOLO model.                                   |
| `camera_info_callback(msg: CameraInfo)`| Processes camera info messages.                               |
| `depth_info_callback(msg: CameraInfo)` | Processes depth camera info messages.                         |
| `depth_callback(msg: Image)`           | Processes depth images.                                       |
| `image_callback(msg: Image)`           | Processes RGB images, runs detection, and publishes results.  |
| `create_detection3d_message(header, box, cv_image, conf)` | Creates detection messages.   |
| `publish_marker(quat, centroid, class_id, bbox_width, bbox_height)` | Publishes visualization markers. |
| `publish_accumulated_point_cloud()`    | Publishes accumulated point clouds.                           |
