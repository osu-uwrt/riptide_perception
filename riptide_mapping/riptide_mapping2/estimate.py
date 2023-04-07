from typing import Tuple
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseWithCovariance
from transforms3d.euler import quat2euler, euler2quat
from math import pi
import numpy as np

DEG_TO_RAD = (pi/180)
RAD_TO_DEG = (180/pi) # Used for debug output
ORIGIN_DEVIATION_LIMIT = 50

class KalmanEstimate:
    def __init__(self, initPoseWithCov: PoseWithCovarianceStamped, covStep: float, covMin: float, detectionCovFactor: float):
        self.lastPose = initPoseWithCov
        self.covStep = covStep
        self.covMin = covMin
        self.detectionCovFactor = detectionCovFactor

    # Takes in a new estimate in the map frame and attempts to add it to the current estimate
    def addPosEstim(self, poseWithCov: PoseWithCovarianceStamped, withOrientation: bool) -> Tuple[bool, str]:
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
        # yaw error will not be considered if orientation is not being merged (withOrientation == True)
        if(posDiffEucNorm < posCovEucNorm and (eucDist[5] < lastCovDist[5] or not withOrientation)):            
            #widen the covariance on our estimate so that it converges correctly
            poseWithCov.pose.covariance = self.lastPose.pose.covariance * self.detectionCovFactor
            
            # merge the estimates in a weighted manner
            self.lastPose.pose = self.updatePose(self.lastPose.pose, poseWithCov.pose, withOrientation)
            
            #decrease covariance
            newCov *= 1.0 - self.covStep
            
            #if orientation is not being merged, its covariance should not change
            if not withOrientation:
                newCov[35] = self.lastPose.pose.covariance[35]

            # lower bound newCov, assumes covariance is positive definite vector
            for i in covs:
                newCov[i] = newCov[i] if newCov[i] > self.covMin else self.covMin
                
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
                return(False, f"Detection yaw {eucDist[5]} observed outside covariance elipsoid {lastCovDist[5]}")
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
        yawDist = abs(pose1RPY[2] - makeContinuous(pose2RPY[2], pose1RPY[2]))

        return np.array([xDist, yDist, zDist, rollDist, pitchDist, yawDist])

    # compute a fused pose based on the covariance of the position vector as well
    # as the orientation in RPY format
    def updatePose(self, pose1: PoseWithCovariance, pose2: PoseWithCovariance, updateOrientation: bool) -> PoseWithCovariance:
        #can update later to also update covariances (currently put into _) when updateValue can properly give them
        newPose = PoseWithCovariance()
        newPose.pose.position.x, _ = updateValue(
            pose1.pose.position.x, pose2.pose.position.x,
            pose1.covariance[0], pose2.covariance[0]
            )
        newPose.pose.position.y, _ = updateValue(
            pose1.pose.position.y, pose2.pose.position.y,
            pose1.covariance[7], pose2.covariance[7]
            )
        newPose.pose.position.z, _ = updateValue(
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
        
        #vision will send an invalid quaternion (components greater than 1) if the orientation could not be determined.
        #so, only merge the orientation if it has a w component less than 1
        if updateOrientation:
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
                pose1RPY[2], makeContinuous(pose2RPY[2], pose1RPY[2]),
                pose1.covariance[35], pose2.covariance[35]
                )

            # rebuild the quat
            (newPose.pose.orientation.w, newPose.pose.orientation.x, 
            newPose.pose.orientation.y, newPose.pose.orientation.z) = euler2quat(rpy[0], rpy[1], rpy[2]) #order defaults to sxyz
        else:
            newPose.pose.orientation = pose1.pose.orientation
            newPose.covariance[35] = pose1.covariance[35]
        
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
    
    
def makeContinuous(angle: float, base: float):
    angle -= 2 * pi
    while abs(angle - base) > pi:
        angle += 2 * pi
    
    if abs(angle - base) > pi:
        angle = (2 * pi) - angle
    
    return angle

# Reconciles two estimates, each with a given estimated value and covariance
# From https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html,
# Where estimate #1 is our current estimate and estimate #2 is the reading we just got in. 
# returns new_mean, cov1. Use to return a properly updated variance value, but it
# didn't work very well, so instead we fudge the covariance in another part of the code
def updateValue(val1, val2, cov1, cov2):
    new_mean = (val1 * (cov2**2) + val2 * (cov1**2)) / (cov1**2 + cov2**2)
    return (new_mean, cov1)
