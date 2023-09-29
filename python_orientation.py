import cv2

# numpy.set_printoptions(threshold=sys.maxsize)

camera_matrix = [[697.235, 0, 648.635], [0, 697.235, 363.702], [0, 0, 1]]
dist_coeffs = [-0.1734, 0.0278, 0, 0, 0]

image = cv2.imread("..\ZED\\flat_surface.pfm", cv2.IMREAD_UNCHANGED)

#library imports
import numpy as np
import math
import matplotlib.pyplot    as     plt

xyz = np.empty((5000, 3))

i = 0

for x in range(600, 700):
    for y in range(300, 350):
        if (image[y, x] != 0):
            xyz[i] = [(x - 648.635) * image[y, x] / 697.235, (y - 363.702) * image[y, x] / 697.235, image[y,x]]
            i += 1

print(xyz.shape)
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

print(image.shape)