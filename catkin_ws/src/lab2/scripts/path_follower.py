#!/usr/bin/env python3

import rospy
import math
import tf

from nav_msgs.msg import Path
from geometry_msgs.msg import Twist


class PathFollower:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("path_follower")

        # Store the optimized waypoint path
        self.path = []

        # Current waypoint index the robot is trying to reach
        self.current_index = 0

        # Radius around a waypoint.
        # If the robot enters this imaginary circle,
        # the waypoint is considered reached.
        self.waypoint_tolerance = 0.05

        # Linear forward speed of the robot
        self.linear_speed = 0.12

        # Gain used for proportional angular control
        self.angular_gain = 1.5

        # Maximum allowed angular velocity
        self.max_angular_speed = 0.6

        # Publisher used to command the robot
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Subscribe to optimized waypoint path
        rospy.Subscriber("/optimized_path", Path, self.path_callback)

        # TF listener is used to get the robot pose
        # from map -> base_footprint
        self.listener = tf.TransformListener()

        rospy.loginfo("Path follower started.")

        # Start the main control loop
        self.run()

    def path_callback(self, msg):
        # Store the optimized waypoint list
        self.path = msg.poses

        # Reset waypoint index whenever a new path arrives
        self.current_index = 0

        rospy.loginfo(
            "Received optimized path with %d waypoints.",
            len(self.path)
        )

    def get_robot_pose(self):
        # Get current robot pose from TF
        try:
            self.listener.waitForTransform(
                "map",
                "base_footprint",
                rospy.Time(0),
                rospy.Duration(1.0)
            )

            (trans, rot) = self.listener.lookupTransform(
                "map",
                "base_footprint",
                rospy.Time(0)
            )

            # Extract x and y position
            x = trans[0]
            y = trans[1]

            # Convert quaternion orientation to yaw angle
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(rot)

            return x, y, yaw

        except Exception as e:
            rospy.logwarn("Could not get robot pose: %s", str(e))
            return None

    def normalize_angle(self, angle):
        # Normalize angle between -pi and pi
        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    def stop_robot(self):
        # Publish zero velocity command to stop the robot
        self.cmd_pub.publish(Twist())

    def run(self):
        # Main control loop frequency
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():

            # If there is no path or all waypoints are completed
            if not self.path or self.current_index >= len(self.path):
                self.stop_robot()
                rate.sleep()
                continue

            # Get robot pose
            pose = self.get_robot_pose()

            # Skip iteration if pose is unavailable
            if pose is None:
                rate.sleep()
                continue

            robot_x, robot_y, robot_yaw = pose

            # Current target waypoint
            target = self.path[self.current_index].pose.position

            target_x = target.x
            target_y = target.y

            # Position error between robot and waypoint
            dx = target_x - robot_x
            dy = target_y - robot_y

            # Euclidean distance to waypoint
            distance = math.sqrt(dx**2 + dy**2)

            # If robot enters waypoint tolerance circle,
            # mark waypoint as reached
            if distance < self.waypoint_tolerance:

                rospy.loginfo(
                    "Waypoint %d reached.",
                    self.current_index
                )

                # Move to next waypoint
                self.current_index += 1

                # Stop robot once final waypoint is reached
                if self.current_index >= len(self.path):
                    rospy.loginfo(
                        "Final waypoint reached. Stopping robot."
                    )

                    self.stop_robot()

                rate.sleep()
                continue

            # Desired heading angle toward the waypoint
            target_angle = math.atan2(dy, dx)

            # Difference between desired heading and current robot heading
            angle_error = self.normalize_angle(
                target_angle - robot_yaw
            )

            # Velocity command message
            cmd = Twist()

            # If robot is not facing the waypoint,
            # rotate first before moving forward
            if abs(angle_error) > 0.25:
                cmd.linear.x = 0.0
                cmd.angular.z = self.angular_gain * angle_error

            # Otherwise move forward while correcting heading
            else:
                cmd.linear.x = self.linear_speed
                cmd.angular.z = self.angular_gain * angle_error

            # Limit angular velocity
            cmd.angular.z = max(
                min(cmd.angular.z, self.max_angular_speed),
                -self.max_angular_speed
            )

            # Publish velocity command to robot
            self.cmd_pub.publish(cmd)

            rate.sleep()


if __name__ == "__main__":
    try:
        PathFollower()

    except rospy.ROSInterruptException:
        pass