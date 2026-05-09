import csv
import math
from pathlib import Path

import numpy as np
import yaml
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav_msgs.msg import Odometry, Path as PathMsg


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw):
    return Quaternion(0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def pose_msg_to_xy_yaw(msg):
    pose = msg.pose.pose if isinstance(msg, Odometry) else msg.pose
    return (
        float(pose.position.x),
        float(pose.position.y),
        yaw_from_quaternion(pose.orientation),
    )


def rotate_map_to_base(dx, dy, base_yaw):
    c = math.cos(base_yaw)
    s = math.sin(base_yaw)
    return c * dx + s * dy, -s * dx + c * dy


def map_point_to_base(point_xy, base_pose):
    bx, by, byaw = base_pose
    return rotate_map_to_base(point_xy[0] - bx, point_xy[1] - by, byaw)


def base_point_to_map(point_xy, base_pose):
    bx, by, byaw = base_pose
    c = math.cos(byaw)
    s = math.sin(byaw)
    x = bx + c * point_xy[0] - s * point_xy[1]
    y = by + s * point_xy[0] + c * point_xy[1]
    return x, y


def make_pose_stamped(x, y, yaw, frame_id, stamp):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = 0.0
    msg.pose.orientation = quaternion_from_yaw(yaw)
    return msg


def make_path_msg(points, frame_id, stamp):
    path = PathMsg()
    path.header.frame_id = frame_id
    path.header.stamp = stamp
    for point in points:
        yaw = float(point[2]) if len(point) > 2 else 0.0
        path.poses.append(make_pose_stamped(point[0], point[1], yaw, frame_id, stamp))
    return path


def make_point(x, y, z=0.0):
    msg = Point()
    msg.x = float(x)
    msg.y = float(y)
    msg.z = float(z)
    return msg


def load_path_file(path_file):
    path = Path(path_file)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r") as f:
            data = yaml.safe_load(f)
        rows = data["points"] if isinstance(data, dict) and "points" in data else data
        points = []
        for row in rows:
            if isinstance(row, dict):
                points.append([float(row["x"]), float(row["y"]), float(row.get("yaw", 0.0))])
            else:
                yaw = float(row[2]) if len(row) > 2 else 0.0
                points.append([float(row[0]), float(row[1]), yaw])
        return np.asarray(points, dtype=np.float64)

    points = []
    with path.open("r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                x = float(row[0])
                y = float(row[1])
            except ValueError:
                continue
            yaw = float(row[2]) if len(row) > 2 and row[2] != "" else 0.0
            points.append([x, y, yaw])
    return np.asarray(points, dtype=np.float64)


def nearest_path_index(points_xy, xy, start_index=0, search_window=200):
    if len(points_xy) == 0:
        return 0
    start = max(0, int(start_index) - search_window // 4)
    end = min(len(points_xy), start + search_window)
    if end <= start:
        start, end = 0, len(points_xy)
    d = points_xy[start:end] - np.asarray(xy, dtype=np.float64)
    return int(start + np.argmin(np.einsum("ij,ij->i", d, d)))


def lookahead_index(points_xy, nearest_idx, distance):
    if len(points_xy) == 0:
        return 0
    accum = 0.0
    idx = int(nearest_idx)
    while idx < len(points_xy) - 1 and accum < distance:
        step = np.linalg.norm(points_xy[idx + 1] - points_xy[idx])
        accum += float(step)
        idx += 1
    return idx
