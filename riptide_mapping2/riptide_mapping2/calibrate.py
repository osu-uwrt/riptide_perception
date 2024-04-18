#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
import tf2_ros
from tf2_ros import TransformException
import transforms3d
import copy
import statistics
import yaml
import os

from geometry_msgs.msg import Pose
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import PoseStamped

# The number of samples to collect before calculating the variances
maxSamples = 0

# Certainty threshold
minCertainty = 1.0

# Target we are doing a covariance calibration for
targetName = ""

# Used to translate between DOPE ids and names of objects
objectIDs = {
    0 : "gate",
    1 : "cutie", 
    2 : "tommy", 
    3 : "gman", 
    4 : "bootlegger", 
    5 : "badge", 
}

# global variables for runtime
samples = []

# ROS parameter name for the yaml file
fileName = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "resource", "config.yaml")

class CalibNode(Node):

    def __init__(self):
        super().__init__('mapping_calibration')
        if(self.get_namespace() == "/"):
            self.get_logger().warning("Namespace may be incorrect, launched on {}".format(self.get_namespace()))

        # Handle startup parameters from top of this file
        self.declare_parameter('~target', "cutie")
        self.declare_parameter('~samples', 100)
        self.declare_parameter('~certainty', 0.5)

        targetName = self.get_parameter("~target").value
        maxSamples = int(self.get_parameter("~samples").value)
        minCertainty = float(self.get_parameter("~certainty").value)

        if(maxSamples < 10):
            self.get_logger().error("Calibration should use more than 10 samples")
            return

        # class variables 
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tl = tf2_ros.TransformListener(self.tf_buffer, self)
        self.worldFrame = "world"
        self.cameraFrame = "{}/stereo/left_optical".format(self.get_namespace())

        # Subscribers
        self.create_subscription(Detection3DArray, "{}/dope/detected_objects".format(self.get_namespace()), self.detectionCallback, qos_profile_system_default) # DOPE's information 

        self.covarSaved = False

        self.get_logger().info("Computing mapping system variance for target '{}' from {} data points with minimum certainty of {}".format(targetName, maxSamples, minCertainty))

    def detectionCallback(self, msg):
        if(len(samples) <= maxSamples):
            #print("got detect {}".format(msg))
            for detection in msg.detections:
                #print(detection)
                # Verify TF system is established
                now = rclpy.time.Time()
                try:
                    trans = self.tf_buffer.lookup_transform(
                        self.cameraFrame,
                        self.worldFrame,
                        now)
                except TransformException as ex:
                    self.get_logger().error(f'Could not transform {self.cameraFrame} to {self.worldFrame}: {ex}')
                    return

                # Note that we don't change the first loop to `detection in msg.detections.results` because we want the timestamp from the Detection3D object
                # Context: This loop will run <number of objects DOPE can identify> times 
                # `result` is of type ObjectHypothesisWithPose (http://docs.ros.org/en/lunar/api/vision_msgs/html/msg/ObjectHypothesisWithPose.html)
                for result in detection.results:

                    # Translate the ID that DOPE gives us to a name meaningful to us
                    name = objectIDs[result.id]

                    # Save a new sample if it meets the critereia
                    if(name == targetName and result.score >= minCertainty):

                        if(len(samples) % int(maxSamples / 10.0) == 0):
                            self.get_logger().info("Adding sample {} for {}".format(len(samples), name))

                        # convert from camera frame to world frame
                        p1 = PoseStamped()
                        p1.header.frame_id = self.cameraFrame
                        p1.header.stamp = now
                        p1.pose = result.pose.pose

                        convertedPos = self.tf_buffer.transform(p1, self.worldFrame)

                        samples.append(copy.deepcopy(convertedPos.pose))

        elif(not self.covarSaved):
            self.get_logger().info("Data collection complete.\nCalculating vision system variance")

            # We have enough data to run statistics now
            xData = []
            yData = []
            zData = []
            yawData = []

            #print(samples)

            # pool data into corresponding lists
            for sample in samples:
                xData.append(sample.position.x)
                yData.append(sample.position.y)
                zData.append(sample.position.z)

                # Convert quaternion to euler angle
                orient = [sample.orientation.x,sample.orientation.y ,sample.orientation.z, sample.orientation.w]
                (_, _, yaw) = transforms3d.euler.quat2euler(orient, 'xyzs')
                yawData.append(yaw)

            # Calculate corresponding variances
            xVar = float(statistics.variance(xData))
            yVar = float(statistics.variance(yData))
            zVar = float(statistics.variance(zData))
            yawVar = float(statistics.variance(yawData))

            # Load exisitng yaml into dict
            covarianceData = {}
            with open(fileName, 'r') as fs:
                try:
                    yamlData = yaml.safe_load(fs)
                    if(not yamlData is None):
                        covarianceData = yamlData
                    else:
                        self.get_logger().warning("File did not contain any yaml")

                except Exception as e:
                    self.get_logger().error("Exception reading yaml file: {}".format(e))
                    
            # Update covar data
            covarianceData[targetName] = {
                "x": xVar,
                "y": yVar,
                "z": zVar,
                "yaw": yawVar
            }

            self.get_logger().info("Updating mapping covariance data: {}" .format(covarianceData))

            # Save data back to yaml
            with open(fileName, 'w') as fs:
                yaml.safe_dump(covarianceData, fs, default_flow_style=False)

            covarSaved = True
            self.get_logger().info("Mapping system covariances updated")


def main(args=None):
    rclpy.init(args=args)

    calib = CalibNode()

    rclpy.spin(calib)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
    