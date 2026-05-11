#!/usr/bin/env python3

import rospy
import sys
import random
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
import math
import time

class myTurtle():
    def __init__(self) -> None:
        """_summary_     
        create all the nessary pubs/subs here and all the nessary other things
        """
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_cb)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.nav_to_pose)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.rate = rospy.Rate(10)

    def nav_to_pose(self, goal):
        # type: (PoseStamped) -> None
        """
        This is a callback function. It should extract data from goal, drive in a striaght line to reach the goal and
        then spin to match the goal orientation.
        :param goal: PoseStamped
        :return:
        """
        goal_x = goal.pose.position.x
        goal_y = goal.pose.position.y
        dx = goal_x - self.x
        dy = goal_y - self.y
        target_angle = math.atan2(dy, dx)
        angle_to_turn = target_angle - self.theta
        # normalize angle between -pi and pi
        if angle_to_turn > math.pi:
            angle_to_turn -= 2 * math.pi
        elif angle_to_turn < -math.pi:
            angle_to_turn += 2 * math.pi
        distance = math.sqrt(dx**2 + dy**2)
        self.rotate(angle_to_turn)
        self.drive_straight(distance, 0.15)
        final_theta = self.convert_to_euler(goal.pose.orientation)
        final_turn = final_theta - self.theta
        # normalize final angle
        if final_turn > math.pi:
            final_turn -= 2 * math.pi
        elif final_turn < -math.pi:
            final_turn += 2 * math.pi
        self.rotate(final_turn)

    def odom_cb(self,msg:Odometry) ->None:
        """_summary_
        Get the odom and update the internal location of the robot
        Args:
            msg (Odometry): _description_
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.theta = self.convert_to_euler(msg.pose.pose.orientation)
        self.linear_vel = msg.twist.twist.linear.x
        self.angular_vel = msg.twist.twist.angular.z
    
    def stop(self)->None:
        """_summary_
        Stop moving
        """
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
   
    def drive_straight(self, dist: float, vel: float)->None:
        """_summary_
        Args:
            dist (_type_): _description_
        """
        start_x = self.x
        start_y = self.y
        msg = Twist()
        msg.linear.x = vel
        msg.angular.z = 0.0
        travelled = 0.0
        while travelled < dist and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(msg)
            travelled = math.sqrt((self.x - start_x) ** 2 + (self.y - start_y) ** 2)
            self.rate.sleep()
        self.stop()

    def spin_wheels(self, u1, u2, time_duration):
        """
        Spin the two wheels
        :param u1: wheel 1 speed
        :param u2: wheel 2 speed
        :param time: time to drive
        :return: None
        """
        msg = Twist()
        # Convert wheel speeds to linear and angular velocity
        msg.linear.x = (u1 + u2) / 2.0
        msg.angular.z = (u2 - u1)
        start_time = rospy.Time.now().to_sec()
        while (rospy.Time.now().to_sec() - start_time) < time_duration:
            self.cmd_vel_pub.publish(msg)
            self.rate.sleep()
        self.stop()

    def rotate(self, angle):
        """
        Rotate in place
        :param angle: angle to rotate
        :return: None
        """
        start_theta = self.theta
        msg = Twist()
        msg.linear.x = 0.0
        angular_speed = 0.3
        if angle < 0:
            angular_speed = -0.3
        msg.angular.z = angular_speed
        rotated = 0.0
        while abs(rotated) < abs(angle) and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(msg)
            rotated = self.theta - start_theta
            # handle angle wraparound
            if rotated > math.pi:
                rotated -= 2 * math.pi
            elif rotated < -math.pi:
                rotated += 2 * math.pi
            self.rate.sleep()
        self.stop()
    
    def convert_to_euler(self, quat: Quaternion) -> float:
        # type: (Quaternion) -> float
        """
        This might be helpful to have
        :param quat: quaternion 
        :return: euler angles
        """
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w
        # Yaw calculation
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def drive_circle(self, radius) -> None:
        """
        Drive the robot in a circle of given radius
        """
        msg = Twist()
        linear_speed = 0.15
        angular_speed = linear_speed / radius
        msg.linear.x = linear_speed
        msg.angular.z = angular_speed
        duration = (2 * math.pi) / angular_speed
        start_time = rospy.Time.now().to_sec()
        while (rospy.Time.now().to_sec() - start_time) < duration and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(msg)
            self.rate.sleep()
        self.stop()

    def spinning_wheels(self, duration: float) -> None:
        """
        Spins the TurtleBot3 in place for a given duration using spin_wheels().
        Args:
            duration (float): Time in seconds for which the robot should spin
        """
        wheel_speed = 0.2
        self.spin_wheels(wheel_speed, -wheel_speed, duration)

def main():
    """_summary_
    create all the node start up here
    """
    rospy.init_node('my_turtlebot_node')
    robot = myTurtle()
    rospy.sleep(2)
    rospy.loginfo("TurtleBot node started.")
    if len(sys.argv) < 2:
        rospy.loginfo("No task provided.")
        rospy.loginfo("Usage: rosrun lab1 my_turtlebot.py [q5|q6|q7|q8]")
        return
    task = sys.argv[1]
    if task == "q5":
        rospy.loginfo("Running Question 5: drive in a circle of radius 0.5 m")
        robot.drive_circle(0.5)
    elif task == "q6":
        rospy.loginfo("Running Question 6: square path with side 0.5 m")
        for _ in range(4):
            robot.drive_straight(0.5, 0.15)
            rospy.sleep(1)
            robot.rotate(math.pi / 2)
            rospy.sleep(1)
        robot.stop()
    elif task == "q7":
        rospy.loginfo("Running Question 7: use RViz 2D Nav Goal")
        rospy.spin()
    elif task == "q8":
        rospy.loginfo("Running Question 8: random dance")
        for _ in range(10):
            move = random.choice(["straight", "rotate_left", "rotate_right", "spin"])
            if move == "straight":
                rospy.loginfo("Dance move: drive straight")
                robot.drive_straight(0.3, 0.15)
            elif move == "rotate_left":
                rospy.loginfo("Dance move: rotate left")
                robot.rotate(math.pi / 2)
            elif move == "rotate_right":
                rospy.loginfo("Dance move: rotate right")
                robot.rotate(-math.pi / 2)
            elif move == "spin":
                rospy.loginfo("Dance move: spin in place")
                robot.spinning_wheels(2)
            rospy.sleep(1)
        robot.stop()
    else:
        rospy.loginfo("Invalid task argument.")
        rospy.loginfo("Usage: rosrun lab1 my_turtlebot.py [q5|q6|q7|q8]")

if __name__ == '__main__':
    main()