from typing import Tuple
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseWithCovariance
from transforms3d.euler import quat2euler, euler2quat
from math import pi
import numpy as np

DEG_TO_RAD = (pi/180)
RAD_TO_DEG = (180/pi) # Used for debug output
ORIGIN_DEVIATION_LIMIT = 50

class KalmanEstimate:
    def __init__(self, initPoseWithCov: PoseWithCovarianceStamped, covStep: float, covMin: float):
        self.lastPose = initPoseWithCov
        self.covStep = covStep
        self.covMin = covMin

    # Takes in a new estimate in the map frame and attempts to add it to the current estimate
    def addPosEstim(self, poseWithCov: PoseWithCovarianceStamped) -> Tuple[bool, str]:
        # find the eigienvals of the last pose
        lastCovDist = covarDist(self.lastPose)

        # look at the data compared to the world
        eucDist = self.poseDist(poseWithCov, self.lastPose)

        posDiffEucNorm = euclideanDist(eucDist[0:2])
        posCovEucNorm = euclideanDist(lastCovDist[0:2])
        
        newCov = np.array(self.lastPose.pose.covariance)
        covs = [0, 7, 14, 21, 28, 35]

        # compare the euclidean position distance to the covariance
        # also compare the rpy distance
        # because zero isnt less than zero the roll and pitch checks are removed
        # eucDist[3] < lastCovDist[3] and eucDist[4] < lastCovDist[4]
        if(posDiffEucNorm < posCovEucNorm and eucDist[5] < lastCovDist[5]):
            #decrease covariance
            newCov *= 1.0 - self.covStep

            # lower bound newCov, assumes covariance is positive definite vector
            for i in covs:
                newCov[i] = newCov[i] if newCov[i] > self.covMin else self.covMin    
            
            #update the covariance on our estimate
            # self.lastPose.pose.covariance = newCov
            poseWithCov.pose.covariance = self.lastPose.pose.covariance * 7
            
            # merge the estimates in a proper weighted manner
            self.lastPose.pose = self.updatePose(self.lastPose.pose, poseWithCov.pose)
            self.lastPose.pose.covariance = newCov
            
            # update the timestamp
            self.lastPose.header.stamp = poseWithCov.header.stamp

            return (True, "")
        else:
            #increase covariance
            newCov *= 1.0 / (1.0 - self.covStep)
            self.lastPose.pose.covariance = newCov
            
            # if outside, reject the detection
            if(posDiffEucNorm >= posCovEucNorm):
                return(False, f"Detection position {posDiffEucNorm} observed outside covariance elipsoid {posCovEucNorm}")
            # elif(eucDist[3] >= lastCovDist[3]):
            #     return(False, "Detection roll observed outside covariance elipsoid")
            # elif(eucDist[4] >= lastCovDist[4]):
            #     return(False, "Detection pitch observed outside covariance elipsoid")
            elif(eucDist[5] >= lastCovDist[5]):
                return(False, "Detection yaw observed outside covariance elipsoid")
            else:
                return(False, "Unknown condition")

    def getPoseEstim(self) -> PoseWithCovarianceStamped:
        return self.lastPose

    # returns a 6x1 vector containing the distance
    def poseDist(self, pose1: PoseWithCovarianceStamped, pose2: PoseWithCovarianceStamped) -> np.ndarray:
        # compute euler distance
        xDist = abs(pose1.pose.pose.position.x - pose2.pose.pose.position.x)
        yDist = abs(pose1.pose.pose.position.y - pose2.pose.pose.position.y)
        zDist = abs(pose1.pose.pose.position.z - pose2.pose.pose.position.z)

        # convert quat to RPY
        pose1RPY = quat2euler([
            pose1.pose.pose.orientation.w, pose1.pose.pose.orientation.x,
            pose1.pose.pose.orientation.y, pose1.pose.pose.orientation.z
        ])
        pose2RPY = quat2euler([
            pose2.pose.pose.orientation.w, pose2.pose.pose.orientation.x,
            pose2.pose.pose.orientation.y, pose2.pose.pose.orientation.z
        ])

        # euler angle distance
        rollDist = abs(pose1RPY[0] - pose2RPY[0])
        pitchDist = abs(pose1RPY[1] - pose2RPY[1])
        yawDist = abs(pose1RPY[2] - pose2RPY[2])

        return np.array([xDist, yDist, zDist, rollDist, pitchDist, yawDist])

    # compute a fused pose based on the covariance of the position vector as well
    # as the orientation in RPY format
    def updatePose(self, pose1: PoseWithCovariance, pose2: PoseWithCovariance) -> PoseWithCovariance:
        newPose = PoseWithCovariance()
        newPose.pose.position.x, newPose.covariance[0] = updateValue(
            pose1.pose.position.x, pose2.pose.position.x,
            pose1.covariance[0], pose2.covariance[0]
            )
        newPose.pose.position.y, newPose.covariance[7] = updateValue(
            pose1.pose.position.y, pose2.pose.position.y,
            pose1.covariance[7], pose2.covariance[7]
            )
        newPose.pose.position.z, newPose.covariance[14] = updateValue(
            pose1.pose.position.z, pose2.pose.position.z,
            pose1.covariance[14], pose2.covariance[14]
            )

        # convert quat to RPY
        pose1RPY = quat2euler([
            pose1.pose.orientation.w, pose1.pose.orientation.x,
            pose1.pose.orientation.y, pose1.pose.orientation.z
        ])
        pose2RPY = quat2euler([
            pose2.pose.orientation.w, pose2.pose.orientation.x,
            pose2.pose.orientation.y, pose2.pose.orientation.z
        ])

        rpy = [0.0, 0.0, 0.0]

        # update RPY covars and estimates
        # rpy[0], newPose.covariance[21] = updateValue(
        #     pose1RPY[0], pose2RPY[0],
        #     pose1.covariance[21], pose2.covariance[21]
        #     )
        # rpy[1], newPose.covariance[28] = updateValue(
        #     pose1RPY[1], pose2RPY[1],
        #     pose1.covariance[28], pose2.covariance[28]
        #     )
        rpy[2], newPose.covariance[35] = updateValue(
            pose1RPY[2], pose2RPY[2],
            pose1.covariance[35], pose2.covariance[35]
            )

        # rebuild the quat
        (newPose.pose.orientation.w, newPose.pose.orientation.x, 
        newPose.pose.orientation.y, newPose.pose.orientation.z) = euler2quat(rpy[0], rpy[1], rpy[2]) #order defaults to sxyz
        
        return newPose

# Compute the size of the covariance elipsoids of a poseWithCovarianceStamped
# takes the eigenvalues of the cov matrix and euclidean norms them to get the 
# elipsoidal radius terms
def covarDist(poseWithCov: PoseWithCovarianceStamped) -> np.ndarray:
    # convert ROS message to numpy array
    covArr = np.array(poseWithCov.pose.covariance)

    # make the array 6x6
    covArr = covArr.reshape((6,6))

    # square root the matrix to compute the distance
    sqtrMat = np.sqrt(covArr)

    # compute and return the eigenvalues
    eigenVals = np.linalg.eigvals(sqtrMat)
    return eigenVals

# Compute the euclidean norm of a vector. 
# Vector is assumed to be 3x1, but will work for higher dims
def euclideanDist(vect: np.ndarray) -> float:
        sumSq = np.dot(vect.T, vect)
        return np.sqrt(sumSq)

# Reconciles two estimates, each with a given estimated value and covariance
# From https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html,
# Where estimate #1 is our current estimate and estimate #2 is the reading we just got in. 
def updateValue(val1, val2, cov1, cov2):
    new_mean = (val1 * (cov2**2) + val2 * (cov1**2)) / (cov1**2 + cov2**2)
    # new_cov = (cov1 * cov2) / (cov1 + cov2)

    return (new_mean, cov1)
