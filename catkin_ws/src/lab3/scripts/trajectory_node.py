#!/usr/bin/env python3

import sys
import rospy
import moveit_commander

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, PoseArray
from trajectory_msgs.msg import JointTrajectory


class OpenManipulatorTrajectoryNode:
    def __init__(self):
        rospy.init_node("task8_trajectory_node")

        moveit_commander.roscpp_initialize(sys.argv)

        self.group = moveit_commander.MoveGroupCommander("arm")
        self.group.set_max_velocity_scaling_factor(0.2)
        self.group.set_max_acceleration_scaling_factor(0.2)

        self.trajectory_pub = rospy.Publisher(
            "/lab3/planned_trajectory",
            JointTrajectory,
            queue_size=10
        )

        rospy.Subscriber(
            "/lab3/joint_goal",
            Float64MultiArray,
            self.joint_goal_callback
        )

        rospy.Subscriber(
            "/lab3/task_goal",
            PoseStamped,
            self.task_goal_callback
        )

        rospy.Subscriber(
            "/lab3/waypoints",
            PoseArray,
            self.waypoints_callback
        )

        rospy.loginfo("Task 8 trajectory node started.")
        rospy.loginfo("Listening to /lab3/joint_goal")
        rospy.loginfo("Listening to /lab3/task_goal")
        rospy.loginfo("Listening to /lab3/waypoints")

    def execute_plan(self, plan):
        if isinstance(plan, tuple):
            success = plan[0]
            trajectory = plan[1]
        else:
            trajectory = plan
            success = len(trajectory.joint_trajectory.points) > 0

        if not success or len(trajectory.joint_trajectory.points) == 0:
            rospy.logwarn("No valid trajectory found.")
            return

        self.trajectory_pub.publish(trajectory.joint_trajectory)

        rospy.loginfo("Executing trajectory...")
        self.group.execute(trajectory, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        rospy.loginfo("Trajectory execution complete.")

    def joint_goal_callback(self, msg):
        if len(msg.data) != 4:
            rospy.logwarn("Joint goal must contain 4 values: joint1 joint2 joint3 joint4")
            return

        joint_goal = list(msg.data)

        rospy.loginfo("Received joint-space goal:")
        rospy.loginfo(joint_goal)

        self.group.set_joint_value_target(joint_goal)
        plan = self.group.plan()
        self.execute_plan(plan)

    def task_goal_callback(self, msg):
        pose_goal = msg.pose

        rospy.loginfo("Received task-space goal:")
        rospy.loginfo("x: %.4f, y: %.4f, z: %.4f",
                      pose_goal.position.x,
                      pose_goal.position.y,
                      pose_goal.position.z)

        self.group.set_pose_target(pose_goal)
        plan = self.group.plan()
        self.execute_plan(plan)

    def waypoints_callback(self, msg):
        if len(msg.poses) == 0:
            rospy.logwarn("Waypoint list is empty.")
            return

        waypoints = []

        for pose in msg.poses:
            waypoints.append(pose)

        rospy.loginfo("Received %d task-space waypoints.", len(waypoints))

        plan, fraction = self.group.compute_cartesian_path(
            waypoints,
            0.01,
            0.0
        )

        rospy.loginfo("Cartesian path fraction: %.2f", fraction)

        if fraction < 0.9:
            rospy.logwarn("Only %.2f of the waypoint path was planned.", fraction)

        if len(plan.joint_trajectory.points) == 0:
            rospy.logwarn("No valid waypoint trajectory found.")
            return

        self.trajectory_pub.publish(plan.joint_trajectory)

        rospy.loginfo("Executing waypoint trajectory...")
        self.group.execute(plan, wait=True)
        self.group.stop()
        rospy.loginfo("Waypoint trajectory execution complete.")

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = OpenManipulatorTrajectoryNode()
    node.spin()