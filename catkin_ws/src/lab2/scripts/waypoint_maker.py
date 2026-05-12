#!/usr/bin/env python3

import rospy
import math
from nav_msgs.msg import Path, GridCells
from geometry_msgs.msg import PoseStamped, Point
from tf.transformations import quaternion_from_euler


class WaypointOptimizer:
    def __init__(self):
        rospy.init_node("waypoint_optimizer")

        self.optimized_path_pub = rospy.Publisher(
            "/optimized_path", Path, queue_size=1, latch=True
        )

        self.optimized_cells_pub = rospy.Publisher(
            "/optimized_path_cells", GridCells, queue_size=1, latch=True
        )

        rospy.Subscriber("/planned_path", Path, self.path_callback)

        rospy.loginfo("Waypoint optimizer node started.")
        rospy.spin()

    def path_callback(self, msg):
        if len(msg.poses) < 2:
            rospy.logwarn("Path too short to optimize.")
            return

        optimized_poses = self.optimize_path(msg.poses)

        for i in range(len(optimized_poses) - 1):
            curr = optimized_poses[i].pose.position
            nxt = optimized_poses[i + 1].pose.position

            yaw = math.atan2(nxt.y - curr.y, nxt.x - curr.x)
            q = quaternion_from_euler(0, 0, yaw)

            optimized_poses[i].pose.orientation.x = q[0]
            optimized_poses[i].pose.orientation.y = q[1]
            optimized_poses[i].pose.orientation.z = q[2]
            optimized_poses[i].pose.orientation.w = q[3]

        # last waypoint → same orientation as previous
        if len(optimized_poses) > 1:
            optimized_poses[-1].pose.orientation = optimized_poses[-2].pose.orientation

            path_msg = Path()
            path_msg.header = msg.header
            path_msg.poses = optimized_poses

            self.optimized_path_pub.publish(path_msg)
            self.publish_gridcells(optimized_poses, msg.header.frame_id)

        rospy.loginfo(
            "Path optimized: %d points -> %d waypoints",
            len(msg.poses),
            len(optimized_poses)
        )

    def optimize_path(self, poses):
        optimized = []

        optimized.append(poses[0])

        previous_direction = None

        for i in range(1, len(poses)):
            prev = poses[i - 1].pose.position
            curr = poses[i].pose.position

            dx = curr.x - prev.x
            dy = curr.y - prev.y

            direction = self.get_direction(dx, dy)

            if previous_direction is None:
                previous_direction = direction
                continue

            # If direction changes, keep the previous point as a waypoint
            if direction != previous_direction:
                optimized.append(poses[i - 1])
                previous_direction = direction

        optimized.append(poses[-1])

        return optimized

    def get_direction(self, dx, dy):
        """
        Converts motion direction into a normalized direction.
        Works for horizontal, vertical, and diagonal lines.
        """

        tolerance = 1e-4

        if abs(dx) < tolerance:
            dx = 0.0
        if abs(dy) < tolerance:
            dy = 0.0

        if dx > 0:
            sx = 1
        elif dx < 0:
            sx = -1
        else:
            sx = 0

        if dy > 0:
            sy = 1
        elif dy < 0:
            sy = -1
        else:
            sy = 0

        return sx, sy

    def publish_gridcells(self, poses, frame_id):
        msg = GridCells()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id

        # adjust if needed based on your map resolution
        msg.cell_width = 0.05
        msg.cell_height = 0.05

        for pose in poses:
            point = Point()
            point.x = pose.pose.position.x
            point.y = pose.pose.position.y
            point.z = 0.0
            msg.cells.append(point)

        self.optimized_cells_pub.publish(msg)


if __name__ == "__main__":
    try:
        WaypointOptimizer()
    except rospy.ROSInterruptException:
        pass