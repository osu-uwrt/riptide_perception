from geometry_msgs.msg import PoseWithCovarianceStamped, PoseWithCovariance, Pose
from transforms3d.euler import quat2euler, euler2quat

import numpy

class Location:
    
    def __init__(self, inital_pose: Pose, buffer_size: int, quantile: tuple[float, float]):
        
        # If the buffer or inital pose are changed externally the Location class must be reset using the reset
        # method for the changes to take effect
        self.inital_pose = inital_pose
        self.buffer_size = buffer_size
        self.quantile = quantile

        self.reset()

    def reset(self):
        
        self.position = {
            "x": numpy.full(self.buffer_size, None, numpy.float64),
            "y": numpy.full(self.buffer_size, None, numpy.float64),
            "z": numpy.full(self.buffer_size, None, numpy.float64)
        }

        self.orientation = {
            "x": numpy.full(self.buffer_size, None, numpy.float64),
            "y": numpy.full(self.buffer_size, None, numpy.float64),
            "z": numpy.full(self.buffer_size, None, numpy.float64)
        }

        self.position["x"][0] = self.inital_pose.position.x
        self.position["y"][0] = self.inital_pose.position.y
        self.position["z"][0] = self.inital_pose.position.z

        self.orientation["x"][0] = self.inital_pose.orientation.x
        self.orientation["y"][0] = self.inital_pose.orientation.y
        self.orientation["z"][0] = self.inital_pose.orientation.z
        

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
                self.position_location = 0

        # Same concept as the position updating
        if update_orientation:
            
            # Convert the angles to euler for the purpose of calculating covariance
            euler_angle = quat2euler((
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
            ))

            self.orientation["x"][self.orientation_location] = euler_angle[0]
            self.orientation["y"][self.orientation_location] = euler_angle[1]
            self.orientation["z"][self.orientation_location] = euler_angle[2]

            self.orientation_location += 1

            if self.position_location >= self.buffer_size:
                self.position_location = 0

    def get_pose(self) -> PoseWithCovariance:

        pose = PoseWithCovariance()

        trimmed = {}

        # Remove the outliers if we have like 10 samples
        if self.buffer_size > 10 and self.position["x"][10] != numpy.nan:
            for key in self.position.keys():
                trimmed[key] = remove_outliers(self.position[key], self.quantile)
        else:
            trimmed = self.position
        
        # Set the position using the mean of trimmed array
        pose.pose.position.x = numpy.nanmean(trimmed["x"])
        pose.pose.position.y = numpy.nanmean(trimmed["y"])
        pose.pose.position.z = numpy.nanmean(trimmed["z"])

        # Set covariance to list for now so we can add incrementally
        cov: list[float] = [0.0] * 36

        cov[0] = numpy.nanvar(trimmed["x"])
        cov[7] = numpy.nanvar(trimmed["y"])
        cov[14] = numpy.nanvar(trimmed["z"])

        # Do the same steps for rotational things
        if self.buffer_size > 10 and self.orientation["x"][10] != numpy.nan:
            for key in self.orientation.keys():
                trimmed[key] = remove_outliers(self.orientation[key], self.quantile)
        else:
            trimmed = self.orientation

        quat = euler2quat(
            numpy.nanmean(trimmed["x"]),
            numpy.nanmean(trimmed["y"]),
            numpy.nanmean(trimmed["z"])
        )

        pose.pose.orientation.w = quat[0]
        pose.pose.orientation.x = quat[1]
        pose.pose.orientation.y = quat[2]
        pose.pose.orientation.z = quat[3]

        cov[21] = numpy.nanvar(trimmed["x"])
        cov[28] = numpy.nanvar(trimmed["y"])
        cov[35] = numpy.nanvar(trimmed["z"])

        pose.covariance = tuple(cov)

        return pose
        
        
    # arr[numpy.where((arr >= numpy.quantile(arr, 0.1)) & (arr <= numpy.quantile(arr, 0.99)))]

# Remove any outliers using quantiles
def remove_outliers(arr: numpy.ndarray, quantile: tuple[float, float]) -> numpy.ndarray:
    return arr[numpy.where(
        (arr >= numpy.nanquantile(arr, quantile[0])) &
        (arr <= numpy.nanquantile(arr, quantile[1]))
    )]