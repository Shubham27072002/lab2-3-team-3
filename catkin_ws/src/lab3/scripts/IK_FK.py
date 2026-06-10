#!/usr/bin/env python3

import math
import rospy
import moveit_commander
import numpy as np
import xml.etree.ElementTree as ET
from sensor_msgs.msg import JointState


ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4"]


def rpy_to_matrix(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,   0,  1]])

    Ry = np.array([[cp, 0, sp],
                   [0,  1, 0],
                   [-sp, 0, cp]])

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr,  cr]])

    return Rz @ Ry @ Rx


def axis_angle_matrix(axis, angle):
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c

    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ])


def make_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def parse_xyz(text):
    if text is None:
        return [0.0, 0.0, 0.0]
    return [float(v) for v in text.split()]


def parse_urdf_chain(base_link, ee_link):
    urdf_text = rospy.get_param("/robot_description")
    root = ET.fromstring(urdf_text)

    joints = []

    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        joint_type = joint.attrib["type"]

        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]

        origin = joint.find("origin")
        if origin is not None:
            xyz = parse_xyz(origin.attrib.get("xyz"))
            rpy = parse_xyz(origin.attrib.get("rpy"))
        else:
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]

        axis_tag = joint.find("axis")
        if axis_tag is not None:
            axis = parse_xyz(axis_tag.attrib.get("xyz"))
        else:
            axis = [0.0, 0.0, 1.0]

        lower = -math.pi
        upper = math.pi
        limit = joint.find("limit")
        if limit is not None:
            if "lower" in limit.attrib:
                lower = float(limit.attrib["lower"])
            if "upper" in limit.attrib:
                upper = float(limit.attrib["upper"])

        joints.append({
            "name": name,
            "type": joint_type,
            "parent": parent,
            "child": child,
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
            "lower": lower,
            "upper": upper
        })

    chain = []
    current = ee_link

    while current != base_link:
        found = None
        for j in joints:
            if j["child"] == current:
                found = j
                break

        if found is None:
            raise RuntimeError("Could not find chain from {} to {}".format(base_link, ee_link))

        chain.insert(0, found)
        current = found["parent"]

    return chain


def fk_from_urdf(chain, joint_values):
    T = np.eye(4)

    for j in chain:
        R_origin = rpy_to_matrix(j["rpy"][0], j["rpy"][1], j["rpy"][2])
        T_origin = make_transform(R_origin, np.array(j["xyz"]))

        T = T @ T_origin

        if j["type"] in ["revolute", "continuous"]:
            angle = joint_values.get(j["name"], 0.0)
            R_joint = axis_angle_matrix(j["axis"], angle)
            T = T @ make_transform(R_joint, np.zeros(3))

        elif j["type"] == "prismatic":
            d = joint_values.get(j["name"], 0.0)
            axis = np.array(j["axis"], dtype=float)
            axis = axis / np.linalg.norm(axis)
            T = T @ make_transform(np.eye(3), axis * d)

    return T


def get_joint_dict():
    msg = rospy.wait_for_message("/joint_states", JointState)

    joint_dict = {}
    for name, pos in zip(msg.name, msg.position):
        joint_dict[name] = pos

    return joint_dict


def numerical_ik(chain, target_position, initial_guess):
    q = np.array(initial_guess, dtype=float)

    damping = 0.01
    step = 1e-5
    max_iters = 300
    tol = 1e-5

    for _ in range(max_iters):
        joint_dict = {ARM_JOINTS[i]: q[i] for i in range(4)}
        T = fk_from_urdf(chain, joint_dict)
        current_position = T[:3, 3]

        error = target_position - current_position

        if np.linalg.norm(error) < tol:
            break

        J = np.zeros((3, 4))

        for i in range(4):
            q_step = q.copy()
            q_step[i] += step

            joint_dict_step = {ARM_JOINTS[k]: q_step[k] for k in range(4)}
            T_step = fk_from_urdf(chain, joint_dict_step)
            pos_step = T_step[:3, 3]

            J[:, i] = (pos_step - current_position) / step

        dq = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(3)) @ error
        q = q + dq

        for i in range(4):
            q[i] = math.atan2(math.sin(q[i]), math.cos(q[i]))

    return q


def print_position(title, p):
    print("\n" + title)
    print("x = {:.6f} m".format(p[0]))
    print("y = {:.6f} m".format(p[1]))
    print("z = {:.6f} m".format(p[2]))


def print_joints(title, q):
    print("\n" + title)
    for i in range(4):
        print("{} = {:.6f} rad".format(ARM_JOINTS[i], q[i]))


def main():
    rospy.init_node("task_2_custom_ik_fk")

    moveit_commander.roscpp_initialize([])
    group = moveit_commander.MoveGroupCommander("arm")

    base_frame = group.get_planning_frame()
    ee_link = group.get_end_effector_link()
    chain = parse_urdf_chain(base_frame, ee_link)

    joint_dict = get_joint_dict()

    measured_joints = np.array([
        joint_dict["joint1"],
        joint_dict["joint2"],
        joint_dict["joint3"],
        joint_dict["joint4"]
    ])

    measured_pose = group.get_current_pose().pose
    measured_position = np.array([
        measured_pose.position.x,
        measured_pose.position.y,
        measured_pose.position.z
    ])

    fk_T = fk_from_urdf(chain, joint_dict)
    fk_position = fk_T[:3, 3]
    fk_error = measured_position - fk_position

    initial_guess = measured_joints.copy()
    ik_solution = numerical_ik(chain, measured_position, initial_guess)

    joint_error = measured_joints - ik_solution
    total_ik_error = np.linalg.norm(joint_error)

    ik_joint_dict = {
        "joint1": ik_solution[0],
        "joint2": ik_solution[1],
        "joint3": ik_solution[2],
        "joint4": ik_solution[3],
    }

    fk_from_ik_T = fk_from_urdf(chain, ik_joint_dict)
    fk_from_ik_position = fk_from_ik_T[:3, 3]
    ik_fk_error = measured_position - fk_from_ik_position

    print_joints("Measured joint angles from /joint_states:", measured_joints)
    print_position("Measured end-effector position from MoveIt:", measured_position)

    print_position("FK position from measured joint angles:", fk_position)
    print_position("FK component error measured - calculated:", fk_error)
    print("\nTotal FK error = {:.6f} m".format(np.linalg.norm(fk_error)))

    print_joints("IK joint angles:", ik_solution)
    print_joints("IK component error measured - calculated:", joint_error)
    print("\nTotal IK joint error = {:.6f} rad".format(total_ik_error))

    print_position("FK position using IK joint angles:", fk_from_ik_position)
    print_position("IK-FK component error measured - calculated:", ik_fk_error)
    print("\nTotal IK-FK error = {:.6f} m".format(np.linalg.norm(ik_fk_error)))


if __name__ == "__main__":
    main()