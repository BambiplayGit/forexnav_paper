#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ideal Path A* Node

Subscribes to the full ground-truth global point cloud, builds a 2D occupancy
grid, and computes the shortest collision-free path via A*.  The resulting
ideal path length is published once as a reference metric for benchmarking.

Published topics
  ~ideal_path         (nav_msgs/Path)          A* waypoints
  ~ideal_path_marker  (visualization_msgs/Marker) thick cyan line for RViz
  ~ideal_path_length  (std_msgs/Float64)       total path length in metres

Parameters
  ~start_x / ~start_y         start position (metres, world frame)
  ~goal_x  / ~goal_y          goal  position
  ~planning_height             z used for Path/Marker visualisation
  ~resolution                  grid cell size   (default 0.1 m)
  ~inflate_radius              obstacle dilation (default 0.25 m)
  ~obstacle_z_min              points with z > this are obstacles (default 0.12)
"""

import heapq
import math

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float64, ColorRGBA
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker

# ─────────────────────── ANSI helpers ───────────────────────
_CYAN  = "\033[36m"
_BCYAN = "\033[1;36m"
_RESET = "\033[0m"


def cyan(t):
    return "{}{}{}".format(_CYAN, t, _RESET)


def bcyan(t):
    return "{}{}{}".format(_BCYAN, t, _RESET)


# ─────────────────────── A* on 2-D grid ────────────────────
_NBRS  = [(-1, -1), (-1, 0), (-1, 1),
          ( 0, -1),          ( 0, 1),
          ( 1, -1), ( 1, 0), ( 1, 1)]
_COSTS = [1.414, 1.0, 1.414,
          1.0,        1.0,
          1.414, 1.0, 1.414]


def astar_grid(occ, si, sj, gi, gj):
    """Return list of (i, j) from start to goal, or None."""
    rows, cols = occ.shape
    if occ[si, sj] or occ[gi, gj]:
        return None

    def h(i, j):
        return math.hypot(i - gi, j - gj)

    open_set = [(h(si, sj), 0.0, si, sj)]
    g_score = {(si, sj): 0.0}
    came_from = {}
    closed = set()

    while open_set:
        _f, g, ci, cj = heapq.heappop(open_set)
        if (ci, cj) in closed:
            continue
        closed.add((ci, cj))
        if ci == gi and cj == gj:
            path = [(gi, gj)]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        for (di, dj), cost in zip(_NBRS, _COSTS):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < rows and 0 <= nj < cols \
               and not occ[ni, nj] and (ni, nj) not in closed:
                ng = g + cost
                if ng < g_score.get((ni, nj), float('inf')):
                    g_score[(ni, nj)] = ng
                    came_from[(ni, nj)] = (ci, cj)
                    heapq.heappush(open_set, (ng + h(ni, nj), ng, ni, nj))
    return None


# ─────────────────────── ROS node ──────────────────────────
class IdealPathAstar:
    def __init__(self):
        rospy.init_node("ideal_path_astar")

        self.start_x = rospy.get_param("~start_x", 0.0)
        self.start_y = rospy.get_param("~start_y", 0.0)
        self.goal_x  = rospy.get_param("~goal_x", 0.0)
        self.goal_y  = rospy.get_param("~goal_y", 0.0)
        self.plan_z  = rospy.get_param("~planning_height", 0.4)
        self.res     = rospy.get_param("~resolution", 0.1)
        self.inflate = rospy.get_param("~inflate_radius", 0.25)
        self.z_min   = rospy.get_param("~obstacle_z_min", 0.12)

        self.pub_path   = rospy.Publisher("~ideal_path", Path,
                                          queue_size=1, latch=True)
        self.pub_marker = rospy.Publisher("~ideal_path_marker", Marker,
                                          queue_size=1, latch=True)
        self.pub_length = rospy.Publisher("~ideal_path_length", Float64,
                                          queue_size=1, latch=True)

        self.computed = False
        rospy.Subscriber("/map_loader/global_cloud", PointCloud2,
                         self._cloud_cb, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped,
                         self._goal_cb, queue_size=5)

        self._cloud_msg = None

        print(bcyan("[IdealPathAstar] waiting for /map_loader/global_cloud …"))
        print(cyan("  start=({:.2f}, {:.2f})  goal=({:.2f}, {:.2f})  "
                   "res={:.2f}  inflate={:.2f}".format(
                       self.start_x, self.start_y,
                       self.goal_x, self.goal_y,
                       self.res, self.inflate)))

    # ── callbacks ──────────────────────────────────────────
    def _cloud_cb(self, msg):
        self._cloud_msg = msg
        if not self.computed:
            self._compute()

    def _goal_cb(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        print(cyan("[IdealPathAstar] new goal ({:.2f}, {:.2f}) – "
                   "recomputing …".format(self.goal_x, self.goal_y)))
        self.computed = False
        if self._cloud_msg is not None:
            self._compute()

    # ── main pipeline ─────────────────────────────────────
    def _compute(self):
        cloud_msg = self._cloud_msg
        res = self.res
        inflate_cells = max(int(round(self.inflate / res)), 0)

        # --- 1. Extract obstacle XY from point cloud ---
        pts = np.array(list(pc2.read_points(
            cloud_msg, field_names=("x", "y", "z"), skip_nans=True)))
        if pts.size == 0:
            rospy.logwarn("[IdealPathAstar] empty point cloud")
            return
        obs_mask = pts[:, 2] > self.z_min
        obs_xy = pts[obs_mask, :2]
        all_xy = pts[:, :2]

        # --- 2. Build 2-D occupancy grid ---
        xy_min = all_xy.min(axis=0) - res
        xy_max = all_xy.max(axis=0) + res
        origin_x, origin_y = xy_min[0], xy_min[1]
        cols = int(math.ceil((xy_max[0] - origin_x) / res))
        rows = int(math.ceil((xy_max[1] - origin_y) / res))

        grid = np.zeros((rows, cols), dtype=np.bool_)

        if obs_xy.shape[0] > 0:
            ci = np.clip(((obs_xy[:, 1] - origin_y) / res).astype(int),
                         0, rows - 1)
            cj = np.clip(((obs_xy[:, 0] - origin_x) / res).astype(int),
                         0, cols - 1)
            grid[ci, cj] = True

        # --- 3. Inflate obstacles ---
        if inflate_cells > 0:
            from scipy.ndimage import binary_dilation
            struct_size = 2 * inflate_cells + 1
            struct = np.zeros((struct_size, struct_size), dtype=np.bool_)
            cy = cx = inflate_cells
            for di in range(struct_size):
                for dj in range(struct_size):
                    if (di - cy) ** 2 + (dj - cx) ** 2 <= inflate_cells ** 2:
                        struct[di, dj] = True
            grid = binary_dilation(grid, structure=struct).astype(np.bool_)

        print(cyan("[IdealPathAstar] grid {}x{} (res={:.2f}m)  "
                   "obstacles {:.1f}%".format(
                       rows, cols, res,
                       100.0 * grid.sum() / grid.size)))

        # --- 4. Convert start / goal to grid indices ---
        def world_to_grid(wx, wy):
            j = int(round((wx - origin_x) / res))
            i = int(round((wy - origin_y) / res))
            return (np.clip(i, 0, rows - 1), np.clip(j, 0, cols - 1))

        si, sj = world_to_grid(self.start_x, self.start_y)
        gi, gj = world_to_grid(self.goal_x, self.goal_y)

        if grid[si, sj]:
            print(cyan("[IdealPathAstar] WARNING: start in obstacle – "
                       "clearing 3x3 neighbourhood"))
            for di in range(-3, 4):
                for dj in range(-3, 4):
                    ni, nj = si + di, sj + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        grid[ni, nj] = False

        if grid[gi, gj]:
            print(cyan("[IdealPathAstar] WARNING: goal in obstacle – "
                       "clearing 3x3 neighbourhood"))
            for di in range(-3, 4):
                for dj in range(-3, 4):
                    ni, nj = gi + di, gj + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        grid[ni, nj] = False

        # --- 5. A* search ---
        print(cyan("[IdealPathAstar] running A* from grid({},{}) "
                   "to grid({},{}) …".format(si, sj, gi, gj)))
        path_idx = astar_grid(grid, si, sj, gi, gj)

        if path_idx is None:
            rospy.logwarn("[IdealPathAstar] A* FAILED – no path found")
            self.pub_length.publish(Float64(-1.0))
            self.computed = True
            return

        # --- 6. Convert to world coordinates & compute length ---
        path_world = []
        for (pi, pj) in path_idx:
            wx = origin_x + pj * res
            wy = origin_y + pi * res
            path_world.append((wx, wy))

        total_len = 0.0
        for k in range(1, len(path_world)):
            dx = path_world[k][0] - path_world[k - 1][0]
            dy = path_world[k][1] - path_world[k - 1][1]
            total_len += math.hypot(dx, dy)

        euclid = math.hypot(self.goal_x - self.start_x,
                            self.goal_y - self.start_y)

        print(bcyan("=" * 44))
        print(bcyan("     Ideal Path (A* on ground truth)"))
        print(bcyan("=" * 44))
        print(cyan("  Start        : ({:.2f}, {:.2f})".format(
            self.start_x, self.start_y)))
        print(cyan("  Goal         : ({:.2f}, {:.2f})".format(
            self.goal_x, self.goal_y)))
        print(cyan("  Euclidean    : {:.2f} m".format(euclid)))
        print(cyan("  A* path len  : {:.2f} m".format(total_len)))
        if euclid > 1e-3:
            print(cyan("  Length scale : {:.2f}".format(total_len / euclid)))
        print(cyan("  Waypoints    : {}".format(len(path_world))))
        print(bcyan("=" * 44))

        # --- 7. Publish ---
        self.pub_length.publish(Float64(total_len))
        self._publish_path(path_world)
        self._publish_marker(path_world)
        self.computed = True

    # ── publishers ─────────────────────────────────────────
    def _publish_path(self, path_world):
        msg = Path()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        for (wx, wy) in path_world:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = self.plan_z
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def _publish_marker(self, path_world):
        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = "world"
        m.ns = "ideal_path"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.15
        m.color = ColorRGBA(0.0, 0.9, 0.9, 0.85)
        m.pose.orientation.w = 1.0

        from geometry_msgs.msg import Point
        for (wx, wy) in path_world:
            p = Point()
            p.x = wx
            p.y = wy
            p.z = self.plan_z + 0.05
            m.points.append(p)
        self.pub_marker.publish(m)


def main():
    try:
        node = IdealPathAstar()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
