#!/usr/bin/env python3

import rospy
import math
import tf

from nav_msgs.msg import Path
from geometry_msgs.msg import Twist


class PathFollower:
    def __init__(self):
        rospy.init_node("path_follower")

        self.path = []
        self.current_index = 0

        self.waypoint_tolerance = 0.05   # imaginary circle radius in meters
        self.linear_speed = 0.12
        self.angular_gain = 1.5
        self.max_angular_speed = 0.6

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/optimized_path", Path, self.path_callback)

        self.listener = tf.TransformListener()

        rospy.loginfo("Path follower started.")
        self.run()

    def path_callback(self, msg):
        self.path = msg.poses
        self.current_index = 0
        rospy.loginfo("Received optimized path with %d waypoints.", len(self.path))

    def get_robot_pose(self):
        try:
            self.listener.waitForTransform("map", "base_footprint", rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("map", "base_footprint", rospy.Time(0))

            x = trans[0]
            y = trans[1]

            roll, pitch, yaw = tf.transformations.euler_from_quaternion(rot)

            return x, y, yaw

        except Exception as e:
            rospy.logwarn("Could not get robot pose: %s", str(e))
            return None

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if not self.path or self.current_index >= len(self.path):
                self.stop_robot()
                rate.sleep()
                continue

            pose = self.get_robot_pose()
            if pose is None:
                rate.sleep()
                continue

            robot_x, robot_y, robot_yaw = pose

            target = self.path[self.current_index].pose.position
            target_x = target.x
            target_y = target.y

            dx = target_x - robot_x
            dy = target_y - robot_y

            distance = math.sqrt(dx**2 + dy**2)

            if distance < self.waypoint_tolerance:
                rospy.loginfo("Waypoint %d reached.", self.current_index)
                self.current_index += 1

                if self.current_index >= len(self.path):
                    rospy.loginfo("Final waypoint reached. Stopping robot.")
                    self.stop_robot()

                rate.sleep()
                continue

            target_angle = math.atan2(dy, dx)
            angle_error = self.normalize_angle(target_angle - robot_yaw)

            cmd = Twist()

            # If robot is not facing the waypoint, rotate first
            if abs(angle_error) > 0.25:
                cmd.linear.x = 0.0
                cmd.angular.z = self.angular_gain * angle_error
            else:
                cmd.linear.x = self.linear_speed
                cmd.angular.z = self.angular_gain * angle_error

            # limit angular velocity
            cmd.angular.z = max(
                min(cmd.angular.z, self.max_angular_speed),
                -self.max_angular_speed
            )

            self.cmd_pub.publish(cmd)
            rate.sleep()


if __name__ == "__main__":
    try:
        PathFollower()
    except rospy.ROSInterruptException:
        pass