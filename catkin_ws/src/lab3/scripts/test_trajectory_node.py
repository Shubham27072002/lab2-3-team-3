#!/usr/bin/env python3

import time
import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, PoseArray, Pose


HOME = [0.0, -1.05, 0.35, 0.70]


def publish_joint_goal(pub, joints):
    msg = Float64MultiArray()
    msg.data = joints
    pub.publish(msg)


def make_pose(x, y, z):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z

    # Keep fixed orientation
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0

    return pose


def publish_task_goal(pub, x, y, z):
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "base_footprint"
    msg.pose = make_pose(x, y, z)
    pub.publish(msg)


def publish_waypoints(pub):
    msg = PoseArray()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "base_footprint"

    msg.poses.append(make_pose(0.10, 0.00, 0.22))
    msg.poses.append(make_pose(0.5, 0.03, 0.22))
    msg.poses.append(make_pose(0.15, 0.00, 0.23))

    pub.publish(msg)


def wait_for_motion(seconds=8):
    rospy.sleep(seconds)


def main():
    rospy.init_node("test_trajectory_node")

    joint_pub = rospy.Publisher(
        "/lab3/joint_goal",
        Float64MultiArray,
        queue_size=10
    )

    task_pub = rospy.Publisher(
        "/lab3/task_goal",
        PoseStamped,
        queue_size=10
    )

    waypoint_pub = rospy.Publisher(
        "/lab3/waypoints",
        PoseArray,
        queue_size=10
    )

    rospy.sleep(2.0)

    rospy.loginfo("Test 1: Joint-space goal")
    publish_joint_goal(joint_pub, [0.0, -0.80, 0.70, 0.35])
    wait_for_motion()

    rospy.loginfo("Returning to home")
    publish_joint_goal(joint_pub, HOME)
    wait_for_motion()

    rospy.loginfo("Test 2: Task-space goal")
    publish_task_goal(task_pub, 0.10, 0.00, 0.22)
    wait_for_motion()

    rospy.loginfo("Returning to home")
    publish_joint_goal(joint_pub, HOME)
    wait_for_motion()

    rospy.loginfo("Test 3: Waypoint list")
    publish_waypoints(waypoint_pub)
    wait_for_motion(12)

    rospy.loginfo("Returning to home")
    publish_joint_goal(joint_pub, HOME)
    wait_for_motion()

    rospy.loginfo("All Task 8 tests completed")


if __name__ == "__main__":
    main()