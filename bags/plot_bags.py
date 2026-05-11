#!/usr/bin/env python3

from pathlib import Path
from bagpy import bagreader
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
BAG_DIR = BASE_DIR
OUT_DIR = BASE_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

print("Bag folder:", BAG_DIR)
print("Plot folder:", OUT_DIR)

bag_files = sorted(BAG_DIR.glob("*.bag"))
print("Bag files found:", [f.name for f in bag_files])

def get_csv_df(bag, topics):
    for topic in topics:
        try:
            csv_path = bag.message_by_topic(topic)
            if csv_path:
                return pd.read_csv(csv_path)
        except Exception:
            pass
    return None

title_map = {
    "q5_circle": "Task 5: Circular Motion",
    "q6_square": "Task 6: Square Motion",
    "q7_nav_goal_CmdLine": "Task 7: Navigation (Command Line)",
    "q7_nav_goal_Rviz": "Task 7: Navigation (RViz)",
    "q8_random_dance": "Task 8: Random Dance"
}

for bag_file in bag_files:
    print(f"Processing {bag_file.name}")
    b = bagreader(str(bag_file))

    df_odom = get_csv_df(b, ["/odom"])
    df_imu = get_csv_df(b, ["/imu", "/imu/data"])
    df_cmd = get_csv_df(b, ["/cmd_vel"])

    if df_odom is None:
        print(f"Skipping {bag_file.name}: /odom not found")
        continue

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    title = title_map.get(bag_file.stem, bag_file.stem)
    fig.suptitle(title, fontsize=14, y=0.98)

    x_col = "field.pose.pose.position.x" if "field.pose.pose.position.x" in df_odom.columns else "pose.pose.position.x"
    y_col = "field.pose.pose.position.y" if "field.pose.pose.position.y" in df_odom.columns else "pose.pose.position.y"
    axes[0].plot(df_odom[x_col], df_odom[y_col])
    axes[0].set_title("Odometry Path")
    axes[0].set_xlabel("X Position (m)")
    axes[0].set_ylabel("Y Position (m)")
    axes[0].grid(True)
    axes[0].axis("equal")

    if df_cmd is not None:
        lin_col = "field.linear.x" if "field.linear.x" in df_cmd.columns else "linear.x"
        ang_col = "field.angular.z" if "field.angular.z" in df_cmd.columns else "angular.z"
        axes[1].plot(df_cmd[lin_col], label="Linear Velocity")
        axes[1].plot(df_cmd[ang_col], label="Angular Velocity")
        axes[1].set_title("Commanded Velocity")
    else:
        lin_col = "field.twist.twist.linear.x" if "field.twist.twist.linear.x" in df_odom.columns else "twist.twist.linear.x"
        ang_col = "field.twist.twist.angular.z" if "field.twist.twist.angular.z" in df_odom.columns else "twist.twist.angular.z"
        axes[1].plot(df_odom[lin_col], label="Linear Velocity")
        axes[1].plot(df_odom[ang_col], label="Angular Velocity")
        axes[1].set_title("Velocity from Odom")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Velocity")
    axes[1].grid(True)
    axes[1].legend()

    if df_imu is not None:
        # Angular velocity (Z)
        ang_col = "field.angular_velocity.z" if "field.angular_velocity.z" in df_imu.columns else "angular_velocity.z"
        
        # Linear acceleration (X, Y, Z)
        ax_col = "field.linear_acceleration.x" if "field.linear_acceleration.x" in df_imu.columns else "linear_acceleration.x"
        ay_col = "field.linear_acceleration.y" if "field.linear_acceleration.y" in df_imu.columns else "linear_acceleration.y"
        az_col = "field.linear_acceleration.z" if "field.linear_acceleration.z" in df_imu.columns else "linear_acceleration.z"

        # Plot angular velocity
        axes[2].plot(df_imu[ang_col], label="Angular Vel Z", linewidth=2)

        # Plot acceleration (lighter lines so it doesn't clutter)
        axes[2].plot(df_imu[ax_col], label="Accel X", alpha=0.6)
        axes[2].plot(df_imu[ay_col], label="Accel Y", alpha=0.6)
        axes[2].plot(df_imu[az_col], label="Accel Z", alpha=0.6)

        axes[2].set_title("IMU Data (Angular Velocity + Acceleration)")
        axes[2].set_xlabel("Sample")
        axes[2].set_ylabel("Value")
        axes[2].grid(True)
        axes[2].legend()

    else:
        axes[2].text(0.5, 0.5, "IMU topic not found", ha="center", va="center")
        axes[2].set_title("IMU")
        axes[2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_file = OUT_DIR / f"{bag_file.stem}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print("Saved:", out_file)

print("Done.")