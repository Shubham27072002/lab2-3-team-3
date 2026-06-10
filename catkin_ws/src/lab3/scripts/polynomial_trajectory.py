#!/usr/bin/env python3

import sys
import copy
import rospy
import numpy as np
import moveit_commander

from geometry_msgs.msg import PoseStamped, TwistStamped, PoseArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory


class PolynomialTrajectory:
    def __init__(self):
        rospy.init_node("polynomial_trajectory", anonymous=True)
        moveit_commander.roscpp_initialize(sys.argv)

        self.arm = moveit_commander.MoveGroupCommander("arm")
        self.arm.set_planning_time(10.0)
        self.arm.allow_replanning(True)
        self.arm.set_max_velocity_scaling_factor(0.15)
        self.arm.set_max_acceleration_scaling_factor(0.15)

        self.desired_pose_pub = rospy.Publisher("/lab3/poly/desired_ee_pose", PoseStamped, queue_size=10)
        self.desired_velocity_pub = rospy.Publisher("/lab3/poly/desired_ee_velocity", TwistStamped, queue_size=10)
        self.desired_joint_pub = rospy.Publisher("/lab3/poly/desired_joint_trajectory", JointTrajectory, queue_size=10)
        self.taskspace_waypoints_pub = rospy.Publisher("/lab3/poly/taskspace_waypoints", PoseArray, queue_size=10)

        rospy.loginfo("Polynomial trajectory node ready.")

    def quintic(self, q0, qf, T, N):
        a0 = q0
        a1 = 0.0
        a2 = 0.0
        a3 = 10.0 * (qf - q0) / T**3
        a4 = -15.0 * (qf - q0) / T**4
        a5 = 6.0 * (qf - q0) / T**5

        t = np.linspace(0.0, T, N)

        q = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
        qd = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
        qdd = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3

        return t, q, qd, qdd

    def publish_desired_task_state(self, pose, vx, vy, vz):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self.arm.get_planning_frame()
        pose_msg.pose = copy.deepcopy(pose)
        self.desired_pose_pub.publish(pose_msg)

        vel_msg = TwistStamped()
        vel_msg.header.stamp = rospy.Time.now()
        vel_msg.header.frame_id = self.arm.get_planning_frame()
        vel_msg.twist.linear.x = float(vx)
        vel_msg.twist.linear.y = float(vy)
        vel_msg.twist.linear.z = float(vz)
        self.desired_velocity_pub.publish(vel_msg)

    def run_q9_task_space(self):
        rospy.sleep(2.0)

        start_pose = self.arm.get_current_pose().pose

        start = np.array([
            start_pose.position.x,
            start_pose.position.y,
            start_pose.position.z
        ])

        # Safe relative task-space goal
        goal = np.array([
            0.097048,
            -0.019909,
            0.2216522
        ])

        T = 6.0
        N = 80

        rospy.loginfo("TASK 9: Task-space quintic polynomial")
        rospy.loginfo("Start: x=%.6f y=%.6f z=%.6f", start[0], start[1], start[2])
        rospy.loginfo("Goal:  x=%.6f y=%.6f z=%.6f", goal[0], goal[1], goal[2])

        t, x, xd, _ = self.quintic(start[0], goal[0], T, N)
        _, y, yd, _ = self.quintic(start[1], goal[1], T, N)
        _, z, zd, _ = self.quintic(start[2], goal[2], T, N)

        waypoints = []
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = self.arm.get_planning_frame()

        for i in range(N):
            pose = copy.deepcopy(start_pose)
            pose.position.x = float(x[i])
            pose.position.y = float(y[i])
            pose.position.z = float(z[i])

            waypoints.append(copy.deepcopy(pose))
            pose_array.poses.append(copy.deepcopy(pose))

            self.publish_desired_task_state(pose, xd[i], yd[i], zd[i])

        self.taskspace_waypoints_pub.publish(pose_array)

        plan, fraction = self.arm.compute_cartesian_path(waypoints, 0.005, 0.0)

        rospy.loginfo("Cartesian path fraction: %.3f", fraction)

        if fraction < 0.90 or len(plan.joint_trajectory.points) == 0:
            rospy.logwarn("Task-space Cartesian path failed. Try a smaller goal.")
            return

        rospy.loginfo("Executing task-space trajectory.")
        success = self.arm.execute(plan, wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()

        if not success:
            rospy.logwarn("Execution reported failure or timeout.")

        rospy.sleep(1.0)

        actual_pose = self.arm.get_current_pose().pose
        actual = np.array([
            actual_pose.position.x,
            actual_pose.position.y,
            actual_pose.position.z
        ])

        error = goal - actual

        rospy.loginfo("Task 9 final comparison:")
        rospy.loginfo("Desired: x=%.6f y=%.6f z=%.6f", goal[0], goal[1], goal[2])
        rospy.loginfo("Actual:  x=%.6f y=%.6f z=%.6f", actual[0], actual[1], actual[2])
        rospy.loginfo("Error:   x=%.6f y=%.6f z=%.6f", error[0], error[1], error[2])
        rospy.loginfo("Total error = %.6f m", np.linalg.norm(error))

    def run_q10_joint_space(self):
        rospy.sleep(2.0)

        rospy.loginfo("TASK 10: Joint-space quintic polynomial")

        q0 = np.array(self.arm.get_current_joint_values()[:4])

        qf = np.array([
            -0.111981,
            -0.615126,
            0.690291,
            0.455592
        ])

        T = 6.0
        N = 80

        rospy.loginfo("Start joints: %s", q0)
        rospy.loginfo("Goal joints:  %s", qf)

        joint_traj = JointTrajectory()
        joint_traj.joint_names = ["joint1", "joint2", "joint3", "joint4"]

        all_q = []
        all_qd = []

        for j in range(4):
            t, q, qd, _ = self.quintic(q0[j], qf[j], T, N)
            all_q.append(q)
            all_qd.append(qd)

        for i in range(N):
            point = JointTrajectoryPoint()
            point.positions = [
                float(all_q[0][i]),
                float(all_q[1][i]),
                float(all_q[2][i]),
                float(all_q[3][i])
            ]
            point.velocities = [
                float(all_qd[0][i]),
                float(all_qd[1][i]),
                float(all_qd[2][i]),
                float(all_qd[3][i])
            ]
            point.time_from_start = rospy.Duration.from_sec(float(t[i] + 0.5))
            joint_traj.points.append(point)

        self.desired_joint_pub.publish(joint_traj)

        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = joint_traj

        rospy.loginfo("Executing joint-space polynomial trajectory.")
        success = self.arm.execute(robot_traj, wait=True)

        self.arm.stop()
        self.arm.clear_pose_targets()

        if not success:
            rospy.logwarn("Full joint trajectory failed. Executing final joint goal instead.")
            self.arm.set_start_state_to_current_state()
            self.arm.set_joint_value_target(qf.tolist())
            self.arm.go(wait=True)
            self.arm.stop()
            self.arm.clear_pose_targets()

        rospy.sleep(1.0)

        actual_joints = np.array(self.arm.get_current_joint_values()[:4])
        joint_error = qf - actual_joints

        actual_pose = self.arm.get_current_pose().pose

        rospy.loginfo("Task 10 final joint comparison:")
        rospy.loginfo("Desired joints: %s", qf)
        rospy.loginfo("Actual joints:  %s", actual_joints)
        rospy.loginfo("Joint error:    %s", joint_error)
        rospy.loginfo("Total joint error = %.6f rad", np.linalg.norm(joint_error))

        rospy.loginfo("Final measured EE position:")
        rospy.loginfo(
            "x=%.6f y=%.6f z=%.6f",
            actual_pose.position.x,
            actual_pose.position.y,
            actual_pose.position.z
        )


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  rosrun lab3 polynomial_trajectory.py q9")
        print("  rosrun lab3 polynomial_trajectory.py q10")
        return

    mode = sys.argv[1].lower()
    node = PolynomialTrajectory()

    if mode == "q9":
        node.run_q9_task_space()
    elif mode == "q10":
        node.run_q10_joint_space()
    else:
        print("Invalid argument. Use q9 or q10.")


if __name__ == "__main__":
    main()