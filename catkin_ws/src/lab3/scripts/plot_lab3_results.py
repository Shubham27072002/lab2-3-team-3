#!/usr/bin/env python3

import os
import math
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
import rosbag
import rospy
import numpy as np
import matplotlib.pyplot as plt


ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4"]


TASK4_TIMES = {
    "pose1": {"RRT": 0.013764, "PRM": 0.016941, "KPIECE": 0.021854},
    "pose2": {"RRT": 0.002536, "PRM": 0.020056, "KPIECE": 0.055540},
    "pose3": {"RRT": 0.014456, "PRM": 0.016705, "KPIECE": 0.054233},
}

TASK5_TIMES = {
    "pose1": {"RRT": 0.015034, "PRM": 0.026005, "KPIECE": 0.033625},
    "pose2": {"RRT": 0.013797, "PRM": 0.040484, "KPIECE": 0.065369},
    "pose3": {"RRT": 0.012292, "PRM": 0.013954, "KPIECE": 0.017725},
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def rpy_to_matrix(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    return Rz @ Ry @ Rx


def axis_angle_matrix(axis, angle):
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c

    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C]
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


def parse_urdf_chain(base_link="base_footprint", ee_link="end_effector_link"):
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
        axis = parse_xyz(axis_tag.attrib.get("xyz")) if axis_tag is not None else [0.0, 0.0, 1.0]

        joints.append({
            "name": name,
            "type": joint_type,
            "parent": parent,
            "child": child,
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis
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
            raise RuntimeError("Could not build URDF chain.")

        chain.insert(0, found)
        current = found["parent"]

    return chain


def fk_from_urdf(chain, joint_values):
    T = np.eye(4)

    for j in chain:
        R_origin = rpy_to_matrix(j["rpy"][0], j["rpy"][1], j["rpy"][2])
        T = T @ make_transform(R_origin, np.array(j["xyz"]))

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


def read_joint_states(bag_path):
    times = []
    pos = []
    vel = []
    eff = []

    with rosbag.Bag(bag_path) as bag:
        for _, msg, t in bag.read_messages(topics=["/joint_states"]):
            jd = {name: i for i, name in enumerate(msg.name)}

            if not all(j in jd for j in ARM_JOINTS):
                continue

            times.append(t.to_sec())

            pos.append([msg.position[jd[j]] for j in ARM_JOINTS])

            if len(msg.velocity) == len(msg.name):
                vel.append([msg.velocity[jd[j]] for j in ARM_JOINTS])
            else:
                vel.append([0.0] * 4)

            if len(msg.effort) == len(msg.name):
                eff.append([msg.effort[jd[j]] for j in ARM_JOINTS])
            else:
                eff.append([0.0] * 4)

    if len(times) == 0:
        return None

    times = np.array(times)
    times = times - times[0]

    return {
        "t": times,
        "pos": np.array(pos),
        "vel": np.array(vel),
        "eff": np.array(eff)
    }


def compute_ee_from_joint_states(joint_data, chain):
    ee = []

    for q in joint_data["pos"]:
        jd = {
            "joint1": q[0],
            "joint2": q[1],
            "joint3": q[2],
            "joint4": q[3],
        }
        T = fk_from_urdf(chain, jd)
        ee.append(T[:3, 3])

    ee = np.array(ee)
    ee_vel = np.gradient(ee, joint_data["t"], axis=0)

    return ee, ee_vel


def get_desired_final_from_controller_goal(bag_path, chain):
    final_q = None

    with rosbag.Bag(bag_path) as bag:
        for _, msg, _ in bag.read_messages(topics=["/arm_controller/follow_joint_trajectory/goal"]):
            traj = msg.goal.trajectory
            if len(traj.points) == 0:
                continue

            names = list(traj.joint_names)
            p = traj.points[-1].positions
            q = {}

            for j in ARM_JOINTS:
                if j in names:
                    q[j] = p[names.index(j)]

            if all(j in q for j in ARM_JOINTS):
                final_q = q

    if final_q is None:
        return None

    T = fk_from_urdf(chain, final_q)
    return T[:3, 3]


def read_desired_ee_q9(bag_path):
    t_pose = []
    p = []
    t_vel = []
    v = []

    with rosbag.Bag(bag_path) as bag:
        for topic, msg, t in bag.read_messages(topics=[
            "/lab3/poly/desired_ee_pose",
            "/lab3/task9/desired_ee_pose",
            "/lab3/poly/desired_ee_velocity",
            "/lab3/task9/desired_ee_velocity"
        ]):
            if "desired_ee_pose" in topic:
                t_pose.append(t.to_sec())
                p.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

            if "desired_ee_velocity" in topic:
                t_vel.append(t.to_sec())
                v.append([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

    if len(t_pose) == 0:
        return None

    t_pose = np.array(t_pose)
    t_pose = t_pose - t_pose[0]
    p = np.array(p)

    if len(t_vel) > 0:
        t_vel = np.array(t_vel)
        t_vel = t_vel - t_vel[0]
        v = np.array(v)
    else:
        t_vel = t_pose
        v = np.gradient(p, t_pose, axis=0)

    return {"t_pos": t_pose, "pos": p, "t_vel": t_vel, "vel": v}


def read_desired_joint_q10(bag_path, chain):
    traj = None

    with rosbag.Bag(bag_path) as bag:
        for _, msg, _ in bag.read_messages(topics=["/lab3/poly/desired_joint_trajectory"]):
            traj = msg

    if traj is None or len(traj.points) == 0:
        return None

    names = list(traj.joint_names)
    t = []
    q = []
    qd = []

    for pt in traj.points:
        t.append(pt.time_from_start.to_sec())

        q_row = []
        qd_row = []

        for j in ARM_JOINTS:
            idx = names.index(j)
            q_row.append(pt.positions[idx])

            if len(pt.velocities) > idx:
                qd_row.append(pt.velocities[idx])
            else:
                qd_row.append(0.0)

        q.append(q_row)
        qd.append(qd_row)

    t = np.array(t)
    t = t - t[0]
    q = np.array(q)
    qd = np.array(qd)

    ee = []
    for row in q:
        jd = {
            "joint1": row[0],
            "joint2": row[1],
            "joint3": row[2],
            "joint4": row[3],
        }
        T = fk_from_urdf(chain, jd)
        ee.append(T[:3, 3])

    ee = np.array(ee)
    ee_vel = np.gradient(ee, t, axis=0)

    return {
        "t": t,
        "q": q,
        "qd": qd,
        "ee": ee,
        "ee_vel": ee_vel
    }


def plot_joint_data(joint_data, title, out_dir, prefix):
    labels = ["joint1", "joint2", "joint3", "joint4"]

    for key, ylabel, suffix in [
        ("pos", "Joint position (rad)", "joint_position"),
        ("vel", "Joint velocity (rad/s)", "joint_velocity"),
        ("eff", "Joint effort", "joint_effort")
    ]:
        plt.figure(figsize=(10, 6))
        for i in range(4):
            plt.plot(joint_data["t"], joint_data[key][:, i], label=labels[i])
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(f"{title} - {ylabel}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_{suffix}.png"), dpi=200)
        plt.close()


def plot_ee_trajectory(t, ee, title, out_dir, prefix):
    plt.figure(figsize=(10, 6))
    plt.plot(t, ee[:, 0], label="x")
    plt.plot(t, ee[:, 1], label="y")
    plt.plot(t, ee[:, 2], label="z")
    plt.xlabel("Time (s)")
    plt.ylabel("End-effector position (m)")
    plt.title(f"{title} - End-effector trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_ee_position.png"), dpi=200)
    plt.close()


def plot_final_error_bar(errors, title, out_dir, prefix):
    names = list(errors.keys())
    vals = [errors[k] for k in names]

    plt.figure(figsize=(10, 5))
    plt.bar(names, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Final position error (m)")
    plt.title(title)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_final_error_comparison.png"), dpi=200)
    plt.close()


def plot_planning_times(time_dict, title, out_dir, prefix):
    poses = list(time_dict.keys())
    planners = ["RRT", "PRM", "KPIECE"]

    x = np.arange(len(poses))
    width = 0.25

    plt.figure(figsize=(10, 6))

    for i, planner in enumerate(planners):
        vals = [time_dict[p][planner] for p in poses]
        bars = plt.bar(x + (i - 1) * width, vals, width, label=planner)

        # Add time value above each bar
        for bar, val in zip(bars, vals):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                "{:.4f}".format(val),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0
            )

    plt.xticks(x, poses)
    plt.ylabel("Planning time (s)")
    plt.title(title)
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_planning_time_comparison.png"), dpi=200)
    plt.close()


def plot_desired_vs_measured(t_meas, meas, t_des, des, title, ylabel, out_dir, prefix):
    plt.figure(figsize=(10, 6))
    axes = ["x", "y", "z"]

    for i in range(3):
        des_interp = np.interp(t_meas, t_des, des[:, i])
        plt.plot(t_meas, des_interp, "--", label=f"desired {axes[i]}")
        plt.plot(t_meas, meas[:, i], label=f"measured {axes[i]}")

    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}.png"), dpi=200)
    plt.close()


def plot_tracking_error(t_meas, meas, t_des, des, title, out_dir, prefix):
    des_interp = np.column_stack([
        np.interp(t_meas, t_des, des[:, i]) for i in range(3)
    ])

    err = des_interp - meas
    err_norm = np.linalg.norm(err, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(t_meas, err[:, 0], label="x error")
    plt.plot(t_meas, err[:, 1], label="y error")
    plt.plot(t_meas, err[:, 2], label="z error")
    plt.plot(t_meas, err_norm, label="total error", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Tracking error (m)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}.png"), dpi=200)
    plt.close()


def process_task_planner_bags(task_dir, task_name, out_dir, chain):
    if not os.path.isdir(task_dir):
        print(f"Missing folder: {task_dir}")
        return

    task_out = os.path.join(out_dir, task_name)
    ensure_dir(task_out)

    errors = {}

    for fname in sorted(os.listdir(task_dir)):
        if not fname.endswith(".bag"):
            continue

        bag_path = os.path.join(task_dir, fname)
        prefix = fname.replace(".bag", "")
        title = f"{task_name.upper()} {prefix}"

        jd = read_joint_states(bag_path)
        if jd is None:
            continue

        ee, _ = compute_ee_from_joint_states(jd, chain)

        plot_joint_data(jd, title, task_out, prefix)
        plot_ee_trajectory(jd["t"], ee, title, task_out, prefix)

        desired_final = get_desired_final_from_controller_goal(bag_path, chain)
        if desired_final is not None:
            actual_final = ee[-1]
            errors[prefix] = np.linalg.norm(desired_final - actual_final)

    if errors:
        plot_final_error_bar(errors, f"{task_name.upper()} final position error", task_out, task_name)


def process_q9(bag_path, out_dir, chain):
    if not os.path.exists(bag_path):
        print(f"Missing q9 bag: {bag_path}")
        return

    q9_out = os.path.join(out_dir, "task9")
    ensure_dir(q9_out)

    jd = read_joint_states(bag_path)
    if jd is None:
        return

    ee_meas, ee_vel_meas = compute_ee_from_joint_states(jd, chain)
    des = read_desired_ee_q9(bag_path)

    plot_joint_data(jd, "Task 9", q9_out, "task9")
    plot_ee_trajectory(jd["t"], ee_meas, "Task 9 measured", q9_out, "task9_measured")

    if des is not None:
        plot_desired_vs_measured(
            jd["t"], ee_meas,
            des["t_pos"], des["pos"],
            "Task 9 desired vs measured end-effector position",
            "Position (m)",
            q9_out,
            "task9_desired_vs_measured_position"
        )

        plot_desired_vs_measured(
            jd["t"], ee_vel_meas,
            des["t_vel"], des["vel"],
            "Task 9 desired vs measured end-effector velocity",
            "Velocity (m/s)",
            q9_out,
            "task9_desired_vs_measured_velocity"
        )

        plot_tracking_error(
            jd["t"], ee_meas,
            des["t_pos"], des["pos"],
            "Task 9 trajectory tracking error",
            q9_out,
            "task9_tracking_error"
        )


def process_q10(bag_path, out_dir, chain):
    if not os.path.exists(bag_path):
        print(f"Missing q10 bag: {bag_path}")
        return

    q10_out = os.path.join(out_dir, "task10")
    ensure_dir(q10_out)

    jd = read_joint_states(bag_path)
    if jd is None:
        return

    ee_meas, ee_vel_meas = compute_ee_from_joint_states(jd, chain)
    des = read_desired_joint_q10(bag_path, chain)

    plot_joint_data(jd, "Task 10 measured", q10_out, "task10_measured")
    plot_ee_trajectory(jd["t"], ee_meas, "Task 10 measured", q10_out, "task10_measured")

    if des is not None:
        plot_desired_vs_measured(
            jd["t"], ee_meas,
            des["t"], des["ee"],
            "Task 10 desired vs measured end-effector position",
            "Position (m)",
            q10_out,
            "task10_desired_vs_measured_position"
        )

        plot_desired_vs_measured(
            jd["t"], ee_vel_meas,
            des["t"], des["ee_vel"],
            "Task 10 desired vs measured end-effector velocity",
            "Velocity (m/s)",
            q10_out,
            "task10_desired_vs_measured_velocity"
        )

        plot_tracking_error(
            jd["t"], ee_meas,
            des["t"], des["ee"],
            "Task 10 trajectory tracking error",
            q10_out,
            "task10_tracking_error"
        )

        plt.figure(figsize=(10, 6))
        for i, j in enumerate(ARM_JOINTS):
            plt.plot(des["t"], des["q"][:, i], "--", label=f"desired {j}")
            plt.plot(jd["t"], jd["pos"][:, i], label=f"measured {j}")
        plt.xlabel("Time (s)")
        plt.ylabel("Joint position (rad)")
        plt.title("Task 10 desired vs measured joint position")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(q10_out, "task10_desired_vs_measured_joint_position.png"), dpi=200)
        plt.close()

def process_task8(bag_path, out_dir, chain):
    if not os.path.exists(bag_path):
        print(f"Missing task8 bag: {bag_path}")
        return

    task8_out = os.path.join(out_dir, "task8")
    ensure_dir(task8_out)

    jd = read_joint_states(bag_path)
    if jd is None:
        print("No joint states found in Task 8 bag.")
        return

    ee, ee_vel = compute_ee_from_joint_states(jd, chain)

    # Joint position velocity effort
    plot_joint_data(jd, "Task 8", task8_out, "task8")

    # End-effector trajectory
    plot_ee_trajectory(jd["t"], ee, "Task 8 measured", task8_out, "task8_measured")

    # 3D end-effector path
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], label="Measured EE path")
    ax.scatter(ee[0, 0], ee[0, 1], ee[0, 2], label="Start")
    ax.scatter(ee[-1, 0], ee[-1, 1], ee[-1, 2], label="End")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Task 8 end-effector 3D trajectory")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(task8_out, "task8_ee_3d_trajectory.png"), dpi=200)
    plt.close()

    # EE velocity
    plt.figure(figsize=(10, 6))
    plt.plot(jd["t"], ee_vel[:, 0], label="vx")
    plt.plot(jd["t"], ee_vel[:, 1], label="vy")
    plt.plot(jd["t"], ee_vel[:, 2], label="vz")
    plt.xlabel("Time (s)")
    plt.ylabel("End-effector velocity (m/s)")
    plt.title("Task 8 measured end-effector velocity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(task8_out, "task8_ee_velocity.png"), dpi=200)
    plt.close()


def main():
    rospy.init_node("plot_lab3_results", anonymous=True)

    cwd = os.getcwd()
    if os.path.basename(cwd) == "catkin_ws":
        project_root = os.path.dirname(cwd)
    else:
        project_root = cwd

    bag_root = os.path.join(project_root, "lab3_rosbags")
    out_dir = os.path.join(project_root, "lab3_all_plots")
    ensure_dir(out_dir)

    print("Project root:", project_root)
    print("Bag root:", bag_root)
    print("Output folder:", out_dir)

    chain = parse_urdf_chain("base_footprint", "end_effector_link")

    process_task_planner_bags(os.path.join(bag_root, "task4"), "task4", out_dir, chain)
    process_task_planner_bags(os.path.join(bag_root, "task5"), "task5", out_dir, chain)

    process_task8(os.path.join(bag_root, "task8", "task8_test.bag"), out_dir, chain)

    plot_planning_times(TASK4_TIMES, "Task 4 task-space planner timing comparison", out_dir, "task4")
    plot_planning_times(TASK5_TIMES, "Task 5 joint-space planner timing comparison", out_dir, "task5")

    process_q9(os.path.join(bag_root, "task9", "q9_taskspace_poly.bag"), out_dir, chain)
    process_q10(os.path.join(bag_root, "task10", "q10_jointspace_poly.bag"), out_dir, chain)

    print("All plots saved to:", out_dir)


if __name__ == "__main__":
    main()