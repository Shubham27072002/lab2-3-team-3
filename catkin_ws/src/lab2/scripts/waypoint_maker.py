#!/usr/bin/env python3

import rospy
import math

from nav_msgs.msg import Path, GridCells
from geometry_msgs.msg import PoseStamped, Point
from tf.transformations import quaternion_from_euler


class WaypointOptimizer:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("waypoint_optimizer")

        # Publisher for optimized waypoint path
        self.optimized_path_pub = rospy.Publisher(
            "/optimized_path",
            Path,
            queue_size=1,
            latch=True
        )

        # Publisher used to visualize optimized waypoints in RViz
        self.optimized_cells_pub = rospy.Publisher(
            "/optimized_path_cells",
            GridCells,
            queue_size=1,
            latch=True
        )

        # Subscribe to the original A* path
        rospy.Subscriber(
            "/planned_path",
            Path,
            self.path_callback
        )

        rospy.loginfo("Waypoint optimizer node started.")

        rospy.spin()

    def path_callback(self, msg):
        # Ignore paths that are too short
        if len(msg.poses) < 2:
            rospy.logwarn("Path too short to optimize.")
            return

        # Remove redundant waypoints from the A* path
        optimized_poses = self.optimize_path(msg.poses)

        # Assign orientation to each waypoint
        # Each waypoint faces toward the next waypoint
        for i in range(len(optimized_poses) - 1):

            curr = optimized_poses[i].pose.position
            nxt = optimized_poses[i + 1].pose.position

            # Calculate yaw angle between consecutive waypoints
            yaw = math.atan2(
                nxt.y - curr.y,
                nxt.x - curr.x
            )

            # Convert yaw angle to quaternion
            q = quaternion_from_euler(0, 0, yaw)

            optimized_poses[i].pose.orientation.x = q[0]
            optimized_poses[i].pose.orientation.y = q[1]
            optimized_poses[i].pose.orientation.z = q[2]
            optimized_poses[i].pose.orientation.w = q[3]

        # Final waypoint uses the same orientation
        # as the previous waypoint
        if len(optimized_poses) > 1:

            optimized_poses[-1].pose.orientation = \
                optimized_poses[-2].pose.orientation

            # Create optimized path message
            path_msg = Path()
            path_msg.header = msg.header
            path_msg.poses = optimized_poses

            # Publish optimized waypoint path
            self.optimized_path_pub.publish(path_msg)

            # Publish optimized waypoint cells for RViz visualization
            self.publish_gridcells(
                optimized_poses,
                msg.header.frame_id
            )

        rospy.loginfo(
            "Path optimized: %d points -> %d waypoints",
            len(msg.poses),
            len(optimized_poses)
        )

    def optimize_path(self, poses):
        # Store final optimized waypoints
        optimized = []

        # Always keep the first waypoint
        optimized.append(poses[0])

        previous_direction = None

        # Traverse through the original path
        for i in range(1, len(poses)):

            prev = poses[i - 1].pose.position
            curr = poses[i].pose.position

            # Calculate movement direction
            dx = curr.x - prev.x
            dy = curr.y - prev.y

            # Normalize direction
            direction = self.get_direction(dx, dy)

            # Initialize previous direction
            if previous_direction is None:
                previous_direction = direction
                continue

            # Keep a waypoint only when the direction changes
            if direction != previous_direction:
                optimized.append(poses[i - 1])

                previous_direction = direction

        # Always keep the final waypoint
        optimized.append(poses[-1])

        return optimized

    def get_direction(self, dx, dy):
        """
        Converts motion direction into a normalized direction.
        Works for horizontal, vertical, and diagonal movement.
        """

        tolerance = 1e-4

        # Remove numerical noise
        if abs(dx) < tolerance:
            dx = 0.0

        if abs(dy) < tolerance:
            dy = 0.0

        # Normalize x direction
        if dx > 0:
            sx = 1

        elif dx < 0:
            sx = -1

        else:
            sx = 0

        # Normalize y direction
        if dy > 0:
            sy = 1

        elif dy < 0:
            sy = -1

        else:
            sy = 0

        return sx, sy

    def publish_gridcells(self, poses, frame_id):
        # Publish optimized waypoints as GridCells for RViz
        msg = GridCells()

        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id

        # Grid cell size
        # Adjust based on map resolution if needed
        msg.cell_width = 0.05
        msg.cell_height = 0.05

        # Convert waypoints into GridCells points
        for pose in poses:

            point = Point()

            point.x = pose.pose.position.x
            point.y = pose.pose.position.y
            point.z = 0.0

            msg.cells.append(point)

        # Publish waypoint visualization
        self.optimized_cells_pub.publish(msg)


if __name__ == "__main__":
    try:
        WaypointOptimizer()

    except rospy.ROSInterruptException:
        pass