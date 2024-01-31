import numpy as np
import matplotlib.pyplot as plt

# Generate random points close to a plane with more noise
np.random.seed(0)
n_points = 100
x = np.random.uniform(-10, 10, n_points)
y = np.random.uniform(-10, 10, n_points)
noise_level = 5.5
a, b, c = 1, -2, 3
z = a*x + b*y + c + np.random.normal(0, noise_level, n_points)

points = np.vstack((x, y, z)).T

# Use SVD to find the best fit plane
centroid = np.mean(points, axis=0)
centered_points = points - centroid
U, S, Vt = np.linalg.svd(centered_points)
normal_vector = Vt[-1]

# Visualize the points, the plane, and the normal vector
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the noisy points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b')

# Plot the plane
xx, yy = np.meshgrid(np.linspace(-10, 10, 50),
                     np.linspace(-10, 10, 50))

d = -centroid.dot(normal_vector)
zz = (-normal_vector[0] * xx - normal_vector[1] * yy - d) / normal_vector[2]
ax.plot_surface(xx, yy, zz, color='cyan', alpha=0.3)

# Plot the normal vector as an arrow
normal_length = 10
if normal_vector[2] < 0:
    normal_vector = -normal_vector
ax.quiver(centroid[0], centroid[1], centroid[2],
          normal_vector[0], normal_vector[1], normal_vector[2],
          length=normal_length, color='r', linewidth=2.5, arrow_length_ratio=0.1)

# Manually setting the same scale for all axes
axis_limit = 5
ax.set_xlim(-axis_limit, axis_limit)
ax.set_ylim(-axis_limit, axis_limit)
ax.set_zlim(-axis_limit, axis_limit)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Noisy Points, Best Fit Plane, and Normal Vector')
plt.show()
