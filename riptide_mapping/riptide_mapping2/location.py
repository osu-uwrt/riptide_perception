from geometry_msgs.msg import PoseWithCovariance, Pose, Vector3, Point
from transforms3d.euler import quat2euler, euler2quat
from math import atan2, pi

from scipy.stats import circmean, circvar

import numpy
import transforms3d as tf3d

class Location:
    
    def __init__(self, inital_pose_xyz: Point, initial_pose_rpy: Vector3, buffer_size: int, quantile: 'tuple[float, float]'):
        # If the buffer or inital pose are changed externally the Location class must be reset using the reset
        # method for the changes to take effect
        self.initial_pose_xyz = inital_pose_xyz
        self.initial_pose_rpy = initial_pose_rpy
        self.buffer_size = buffer_size
        self.quantile = quantile
        
        self.position = {
            "x": numpy.full(self.buffer_size, self.initial_pose_xyz.x, numpy.float64),
            "y": numpy.full(self.buffer_size, self.initial_pose_xyz.y, numpy.float64),
            "z": numpy.full(self.buffer_size, self.initial_pose_xyz.z, numpy.float64)
        }

        self.orientation = {
            "x": numpy.full(self.buffer_size, self.initial_pose_rpy.x * pi / 180.0, numpy.float64),
            "y": numpy.full(self.buffer_size, self.initial_pose_rpy.y * pi / 180.0, numpy.float64),
            "z": numpy.full(self.buffer_size, self.initial_pose_rpy.z * pi / 180.0, numpy.float64)
        }
        
        self.buffer_warm = False
        self.reset()

    def reset(self):
        # mark all buffers as not warmed up yet by making the last element nan
        self.position["x"][len(self.position["x"]) - 1] = numpy.nan
        self.position["y"][len(self.position["y"]) - 1] = numpy.nan
        self.position["z"][len(self.position["z"]) - 1] = numpy.nan

        self.orientation["x"][len(self.orientation["x"]) - 1] = numpy.nan
        self.orientation["y"][len(self.orientation["y"]) - 1] = numpy.nan
        self.orientation["z"][len(self.orientation["z"]) - 1] = numpy.nan
        
        self.buffer_warm = False

        # Variable is used so we can get rid of old poses in a rolling fashion instead of shifting entire array
        self.position_location = 1
        self.orientation_location = 1

    def add_pose(self, pose: Pose, update_position: bool, update_orientation: bool):
        # If we are updating position add the values from the pose and update the location tracker
        if update_position:
            self.position["x"][self.position_location] = pose.position.x
            self.position["y"][self.position_location] = pose.position.y
            self.position["z"][self.position_location] = pose.position.z

            self.position_location += 1
            
            # Check if we have reached the end of the array and if so start from the beginning
            if self.position_location >= self.buffer_size:
                self.buffer_warm = True
                self.position_location = 0

        # Same concept as the position updating
        if update_orientation:
            unit = [0, 0, 1] # Unit vector. Vision detects normal vector as Z axis so thats what we will rotate here
            
            # get rotation matrix from quaternion. IMPORTANT: quat ordered wxyz
            det_rotm = tf3d.quaternions.quat2mat(
                [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])[:3, :3]
            
            projected_unit = numpy.dot(det_rotm, unit)
            yaw = atan2(projected_unit[1], projected_unit[0])

            self.orientation["x"][self.orientation_location] = 0 # maybe someday roll and pitch will be nonzero. When that happens, do a projection with euler_angle[0]
            self.orientation["y"][self.orientation_location] = 0 # maybe someday...
            self.orientation["z"][self.orientation_location] = yaw

            self.orientation_location += 1

            if self.orientation_location >= self.buffer_size:
                self.buffer_warm = True
                self.orientation_location = 0

    def get_pose(self) -> PoseWithCovariance:
        pose = PoseWithCovariance()
        trimmed = {}
        
        # Remove the outliers if we have like 10 samples
        if self.buffer_warm:
            for key in self.position.keys():
                trimmed[key] = remove_outliers(self.position[key], self.quantile)
        else:
            trimmed = self.position
        
        # Set the position using the mean of trimmed array
        pose.pose.position.x = numpy.nanmean(trimmed["x"])
        pose.pose.position.y = numpy.nanmean(trimmed["y"])
        pose.pose.position.z = numpy.nanmean(trimmed["z"])

        # Set covariance to list for now so we can add incrementally
        cov: 'list[float]' = [0.0] * 36
        
        # Do the same steps for rotational things
        if self.buffer_warm:
            for key in self.orientation.keys():
                trimmed[key] = remove_outliers(self.orientation[key], self.quantile)
        else:
            trimmed = self.orientation
        
        quat = euler2quat(
            circmean(trimmed['x'], nan_policy="omit"),
            circmean(trimmed['y'], nan_policy="omit"),
            circmean(trimmed['z'], nan_policy="omit")
        )
        
        pose.pose.orientation.w = quat[0]
        pose.pose.orientation.x = quat[1]
        pose.pose.orientation.y = quat[2]
        pose.pose.orientation.z = quat[3]
                
        if self.buffer_warm:
            cov[0] = numpy.nanvar(self.position["x"])
            cov[7] = numpy.nanvar(self.position["y"])
            cov[14] = numpy.nanvar(self.position["z"])
            cov[21] = circvar(self.orientation["x"])
            cov[28] = circvar(self.orientation["y"])
            cov[35] = circvar(self.orientation["z"])
        else:
            #publish initial covariances
            cov[0] =  1.0
            cov[7] =  1.0
            cov[14] = 1.0
            cov[21] = 0.1
            cov[28] = 0.1
            cov[35] = 0.1

        pose.covariance = tuple(cov)

        return pose


    def cool_buffer(self):
        self.buffer_warm = False


# Remove any outliers using quantiles
def remove_outliers(arr: numpy.ndarray, quantile: 'tuple[float, float]') -> numpy.ndarray:
    return arr[numpy.where(
        (arr >= numpy.nanquantile(arr, quantile[0])) &
        (arr <= numpy.nanquantile(arr, quantile[1]))
    )]
