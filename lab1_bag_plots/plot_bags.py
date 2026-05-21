#!/usr/bin/env python3

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from bagpy import bagreader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def smooth(series, window=100):
    return pd.Series(series).rolling(window=window, center=True, min_periods=1).mean().to_numpy()

def smooth_xy(x, y, window=20):
    x_s = x.rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    y_s = y.rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    return x_s, y_s

def compute_linear_velocity_from_odom(df, x_col, y_col):
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    if "Time" in df.columns:
        t = df["Time"].to_numpy()
    elif "time" in df.columns:
        t = df["time"].to_numpy()
    else:
        t = np.arange(len(x))

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)

    dt[dt == 0] = np.nan

    velocity = np.sqrt(dx**2 + dy**2) / dt
    velocity = np.insert(velocity, 0, 0)

    velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)

    return velocity

title_map = {
    "q5_circle_lab1": "Lab 1 - Task 5: Circular Motion",
    "q6_square_lab1": "Lab 1 - Task 6: Square Motion",
    "q7_nav_goal_CmdLine_lab1": "Lab 1 - Task 7: Navigation (Command Line)",
}

for bag_file in bag_files:
    print(f"Processing {bag_file.name}")
    b = bagreader(str(bag_file))

    df_odom = get_csv_df(b, ["/odom"])
    df_imu = get_csv_df(b, ["/imu", "/imu/data"])

    if df_odom is None:
        print(f"Skipping {bag_file.name}: /odom not found")
        continue

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    title = title_map.get(bag_file.stem, bag_file.stem)
    fig.suptitle(title, fontsize=14, y=0.98)

    # -------- ODOM PATH --------
    x_col = "field.pose.pose.position.x" if "field.pose.pose.position.x" in df_odom.columns else "pose.pose.position.x"
    y_col = "field.pose.pose.position.y" if "field.pose.pose.position.y" in df_odom.columns else "pose.pose.position.y"

    x_s, y_s = smooth_xy(df_odom[x_col], df_odom[y_col], window=20)

    axes[0].plot(x_s, y_s)
    axes[0].set_title("Odometry Path")
    axes[0].set_xlabel("X Position (m)")
    axes[0].set_ylabel("Y Position (m)")
    axes[0].grid(True)
    axes[0].axis("equal")

    # -------- LINEAR VELOCITY FROM ODOM --------
    linear_velocity = compute_linear_velocity_from_odom(df_odom, x_col, y_col)
    linear_velocity_smooth = smooth(linear_velocity, window=100)

    axes[1].plot(linear_velocity_smooth, label="Linear Velocity")
    axes[1].set_title("Linear Velocity from Odom")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Velocity")
    axes[1].grid(True)
    axes[1].legend()

    # -------- IMU --------
    if df_imu is not None:
        ang_col = "field.angular_velocity.z" if "field.angular_velocity.z" in df_imu.columns else "angular_velocity.z"

        ax_col = "field.linear_acceleration.x" if "field.linear_acceleration.x" in df_imu.columns else "linear_acceleration.x"
        ay_col = "field.linear_acceleration.y" if "field.linear_acceleration.y" in df_imu.columns else "linear_acceleration.y"
        az_col = "field.linear_acceleration.z" if "field.linear_acceleration.z" in df_imu.columns else "linear_acceleration.z"

        axes[2].plot(smooth(df_imu[ang_col], window=100), label="Angular Vel Z", linewidth=2)
        axes[2].plot(smooth(df_imu[ax_col], window=100), label="Accel X", alpha=0.6)
        axes[2].plot(smooth(df_imu[ay_col], window=100), label="Accel Y", alpha=0.6)
        axes[2].plot(smooth(df_imu[az_col], window=100), label="Accel Z", alpha=0.6)

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