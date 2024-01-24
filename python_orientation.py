import cv2

# numpy.set_printoptions(threshold=sys.maxsize)

camera_matrix = [[697.235, 0, 648.635], [0, 697.235, 363.702], [0, 0, 1]]
dist_coeffs = [-0.1734, 0.0278, 0, 0, 0]

depthIm = cv2.imread("/home/markc/Rotated.pfm", cv2.IMREAD_UNCHANGED)
# depthIm = cv2.rotate(depthIm, cv2.ROTATE_180)
depthIm = cv2.flip(depthIm, 0)
colorIm = cv2.imread("/home/markc/Rotated.png")
grayIm = cv2.imread("/home/markc/Rotated.png", cv2.IMREAD_GRAYSCALE)


#library imports
import numpy as np
import math
import matplotlib.pyplot    as     plt

xyz = np.empty((20, 3))

corners = cv2.goodFeaturesToTrack(grayIm[430:710,880:1410], 20, 0.2, 10)
i = 0
for corner in corners:
    x, y = corner.ravel()
    x = int(x) + 880
    y = int(y) + 430
    if depthIm[y, x] != 0: 
        xyz[i] = [(x - 967.35) * depthIm[y, x] / 1078.96, (y - 544.594) * depthIm[y, x] / 1079, depthIm[y, x]]
        i += 1
        cv2.circle(colorIm, (x, y), 3, (0, 0, 255), -1)
        print(depthIm[y, x])
xyz.resize((i, 3))
print(i)
print(len(xyz))

zMean = np.average(xyz[:,2])
zMedian = np.median(xyz[:,2])
zMax = np.max(xyz[:,2])
# for i in range(20):
#     if xyz[i, 2] <

print(zMean, zMedian)

cv2.namedWindow("test", cv2.WINDOW_FREERATIO)
cv2.imshow("test", colorIm)
cv2.waitKey(0)


# i = 0

# for x in range(600, 700):
#     for y in range(300, 350):
#         if (depthIm[y, x] != 0):
#             xyz[i] = [(x - 648.635) * depthIm[y, x] / 697.235, (y - 363.702) * depthIm[y, x] / 697.235, depthIm[y,x]]
#             i += 1

''' best plane fit'''
#1.calculate centroid of points and make points relative to it
centroid         = xyz.mean(axis = 0)
xyzT             = np.transpose(xyz)
xyzR             = xyz - centroid                         #points relative to centroid
xyzRT            = np.transpose(xyzR)                       

#2. calculate the singular value decomposition of the xyzT matrix and get the normal as the last column of u matrix
u, sigma, v       = np.linalg.svd(xyzR)
normal            = v[2]                                 
normal            = normal / np.linalg.norm(normal)       #we want normal vectors normalized to unity

'''matplotlib display'''
#prepare normal vector for display
forGraphs = list()
forGraphs.append(np.array([centroid[0],centroid[1],centroid[2],normal[0],normal[1], normal[2]]))
print((normal[0],normal[1], normal[2]))

#get d coefficient to plane for display
d = normal[0] * centroid[0] + normal[1] * centroid[1] + normal[2] * centroid[2]

# create x,y for display
minPlane = int(math.floor(min(min(xyzT[0]), min(xyzT[1]), min(xyzT[2]))))
maxPlane = int(math.ceil(max(max(xyzT[0]), max(xyzT[1]), max(xyzT[2]))))
xx, yy = np.meshgrid(range(minPlane,maxPlane), range(minPlane,maxPlane))

# calculate corresponding z for display
z = (-normal[0] * xx - normal[1] * yy + d) * 1. /normal[2]

#matplotlib display code
forGraphs = np.asarray(forGraphs)
X, Y, Z, U, V, W = zip(*forGraphs)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, z, alpha=0.2)
ax.scatter(xyzT[0],xyzT[1],xyzT[2])
ax.quiver(X, Y, Z, U, V, W)
ax.set_xlim([min(xyzT[0])- 0.1, max(xyzT[0]) + 0.1])
ax.set_ylim([min(xyzT[1])- 0.1, max(xyzT[1]) + 0.1])
ax.set_zlim([min(xyzT[2])- 0.1, max(xyzT[2]) + 0.1])   
plt.show() 
