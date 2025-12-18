#!/usr/bin/env python3


def get_hole_size(self, hole): #move
		x_min, y_min, x_max, y_max = map(int, hole.xyxy[0])
		hole_width = x_max - x_min
		hole_height = y_max - y_min
		hole_size = hole_height*hole_width
		return hole_size
 
def find_top_and_bottom_holes(self): #move
    if self.plane_normal is None:
        self.get_logger().warning("Plane normal not defined. Cannot compute hole positions.")
        return
    if self.torpedo_centroid is None:
        self.get_logger().warning("Centroid not defined. Cannot compute hole positions.")
        return

    hole_positions = []
    for hole in self.torpedo_holes:
        x_min, y_min, x_max, y_max = map(int, hole.xyxy[0])
        bbox_center_x = (x_min + x_max) / 2
        bbox_center_y = (y_min + y_max) / 2
        
        # Calculate 3D position using plane intersection
        d = np.linalg.inv(self.intrinsic_matrix) @ np.array([bbox_center_x, bbox_center_y, 1.0])
        d = d / np.linalg.norm(d)
        
        n = self.plane_normal
        p0 = self.torpedo_centroid

        numerator = np.dot(n, p0)
        denominator = np.dot(n, d)
        if denominator == 0:
            self.get_logger().warning(f"Denominator zero for hole center ({bbox_center_x}, {bbox_center_y}). Skipping this hole.")
            continue
        t = numerator / denominator
        if t <= 0:
            self.get_logger().warning(f"Intersection behind the camera for hole center ({bbox_center_x}, {bbox_center_y}). Skipping this hole.")
            continue
        point_3d = t * d
        
        # Store hole with its 3D position and Y coordinate for sorting
        hole_positions.append((hole, point_3d, point_3d[1]))
        #self.get_logger().info(f"Hole position computed: Y={point_3d[1]}")

    if not hole_positions:
        self.get_logger().warning("No valid hole positions computed.")
        return

    # Sort holes based on Y coordinate (height)
    hole_positions.sort(key=lambda x: x[2])

    self.torpedo_bottom_hole = hole_positions[-1][0]  # Lower Y = bottom
    self.torpedo_top_hole = hole_positions[0][0]    # Higher Y = top

    #self.get_logger().info(f"Bottom hole Y: {hole_positions[0][2]}")
    #self.get_logger().info(f"Top hole Y: {hole_positions[-1][2]}")

def find_smallest_and_largest_holes(self): #move
    if self.plane_normal is None or self.mapping_map_centroid is None:
        self.get_logger().warning("Plane normal or centroid not defined. Cannot compute hole sizes.")
        return

    hole_sizes = []
    for hole in self.mapping_holes:
        x_min, y_min, x_max, y_max = map(int, hole.xyxy[0])
        corners_2d = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max)
        ]
        corners_3d = []
        for (u, v) in corners_2d:
            d = np.linalg.inv(self.intrinsic_matrix) @ np.array([u, v, 1.0])
            n = self.plane_normal
            p0 = self.mapping_map_centroid

            numerator = np.dot(n, p0)
            denominator = np.dot(n, d)
            if denominator == 0:
                self.get_logger().warning(f"Denominator zero for point ({u}, {v}). Skipping this corner.")
                continue
            t = numerator / denominator
            if t <= 0:
                self.get_logger().warning(f"Intersection behind the camera for point ({u}, {v}). Skipping this corner.")
                continue
            point_3d = t * d
            corners_3d.append(point_3d)

        if len(corners_3d) == 4:
            width_vector = corners_3d[1] - corners_3d[0]
            height_vector = corners_3d[3] - corners_3d[0]
            width = np.linalg.norm(width_vector)
            height = np.linalg.norm(height_vector)
            hole_size = width * height
            hole_sizes.append((hole, hole_size))
            #self.get_logger().info(f"Hole size computed: {hole_size}")
        else:
            self.get_logger().warning(f"Not enough valid corners for hole. Expected 4, got {len(corners_3d)}")
            continue

    if not hole_sizes:
        self.get_logger().warning("No valid hole sizes computed.")
        return

    # Sort holes based on hole_size
    hole_sizes.sort(key=lambda x: x[1])

    self.smallest_hole = hole_sizes[0][0]
    self.largest_hole = hole_sizes[-1][0]

    #self.get_logger().info(f"Smallest hole size: {hole_sizes[0][1]}")
    #self.get_logger().info(f"Largest hole size: {hole_sizes[-1][1]}")

def cleanup_old_holes(self, age_threshold=2.0): #move
    # Get the current time as a builtin_interfaces.msg.Time object
    current_time = self.get_clock().now().to_msg()

    # for bbox, timestamp in self.holes:
    # 	self.get_logger().info(f"Time for cleanup: {((current_time.sec - timestamp.sec) + (current_time.nanosec - timestamp.nanosec) * 1e-9)}")
    # Filter out holes older than the specified age threshold
    self.holes = [(bbox, timestamp) for bbox, timestamp in self.holes
                if ((current_time.sec - timestamp.sec) + (current_time.nanosec - timestamp.nanosec) * 1e-9) < age_threshold]