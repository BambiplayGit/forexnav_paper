#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark Recorder Node  (with integrated A* ideal-path computation)

Directly subscribes to the ground-truth global point cloud, builds a 2-D
occupancy grid, and runs A* to obtain the ideal shortest path length.
No dependency on any external node for this metric.

Recorded metrics (all rounded to 2 d.p.):
  1. Travel time           -- goal receipt → arrival
  2. Path length           -- accumulated odometry
  3. Ideal path length     -- A* on full ground-truth map
  4. Length scale           -- path_length / Euclidean(start, goal)
  5. Path ratio            -- path_length / ideal_path_length
  6. Average linear velocity
  7. Known area ratio
"""

import os
import csv
import math
import heapq
from datetime import datetime

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


# ───────────────────────── ANSI helpers ──────────────────────
_PURPLE      = "\033[35m"
_BOLD_PURPLE = "\033[1;35m"
_CYAN        = "\033[36m"
_BOLD_CYAN   = "\033[1;36m"
_RESET       = "\033[0m"

def purple(t):      return "{}{}{}".format(_PURPLE, t, _RESET)
def bold_purple(t): return "{}{}{}".format(_BOLD_PURPLE, t, _RESET)
def cyan(t):        return "{}{}{}".format(_CYAN, t, _RESET)
def bold_cyan(t):   return "{}{}{}".format(_BOLD_CYAN, t, _RESET)


# ───────────────────────── A* on 2-D grid ────────────────────
_NBRS  = [(-1, -1), (-1, 0), (-1, 1),
          ( 0, -1),          ( 0, 1),
          ( 1, -1), ( 1, 0), ( 1, 1)]
_COSTS = [1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414]


def astar_grid(occ, si, sj, gi, gj):
    """8-connected A* on boolean grid.  Returns [(i,j), …] or None."""
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


# ───────────────────────── node ──────────────────────────────

class BenchmarkRecorder:
    IDLE = 0
    NAVIGATING = 1

    def __init__(self):
        rospy.init_node("benchmark_recorder")

        # ── params ──
        self.algorithm   = rospy.get_param("~algorithm", "unknown")
        self.map_name    = rospy.get_param("~map_name", "unknown")
        self.goal_thresh = rospy.get_param("~goal_reach_threshold", 0.2)
        self.plan_z      = rospy.get_param("~planning_height", 0.4)
        self._grid_res   = rospy.get_param("~grid_resolution", 0.1)
        self._inflate_r  = rospy.get_param("~inflate_radius", 0.25)
        self._slice_z_lo = rospy.get_param("~slice_z_lo", 0.3)
        self._slice_z_hi = rospy.get_param("~slice_z_hi", 0.6)
        output_dir = rospy.get_param("~output_dir",
                                     os.path.expanduser("~/benchmark_results"))
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # ── navigation state ──
        self.state      = self.IDLE
        self.goal_xy    = None
        self.start_xy   = None
        self.start_time = None
        self.path_length = 0.0
        self.prev_xy    = None
        self.vel_samples = []
        self.known_area_ratio = 0.0

        # ── ideal-path state ──
        self._occ_grid    = None   # np.bool_ 2-D array (built once)
        self._grid_origin = None   # (ox, oy)
        self._grid_rows   = 0
        self._grid_cols   = 0
        self.ideal_path_length = -1.0
        self._ideal_computed   = False

        # ── publishers (ideal path visualisation) ──
        self.pub_ideal_path = rospy.Publisher(
            "~ideal_path", Path, queue_size=1, latch=True)
        self.pub_ideal_marker = rospy.Publisher(
            "~ideal_path_marker", Marker, queue_size=1, latch=True)

        # ── subscribers ──
        rospy.Subscriber("/map_loader/global_cloud", PointCloud2,
                         self._cloud_cb, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped,
                         self._goal_cb, queue_size=5)
        rospy.Subscriber("/odom", Odometry,
                         self._odom_cb, queue_size=50)
        rospy.Subscriber("/local_sensing/occupancy_grid", OccupancyGrid,
                         self._map_cb, queue_size=2)

        print(purple("[BenchmarkRecorder] ready  |  algorithm={}  map={}  "
                     "output_dir={}".format(self.algorithm, self.map_name,
                                            os.path.abspath(self.output_dir))))

    # ══════════════════ grid construction ══════════════════════

    def _cloud_cb(self, msg):
        if self._occ_grid is not None:
            return
        self._build_grid(msg)
        if self.start_xy is not None and self.goal_xy is not None \
           and not self._ideal_computed:
            self._compute_ideal_path()

    def _build_grid(self, cloud_msg):
        res = self._grid_res
        inflate_cells = max(int(round(self._inflate_r / res)), 0)

        pts = np.array(list(pc2.read_points(
            cloud_msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float32)
        if pts.size == 0:
            rospy.logwarn("[BenchmarkRecorder] empty global cloud!")
            return

        obs_mask = (pts[:, 2] >= self._slice_z_lo) & (pts[:, 2] <= self._slice_z_hi)
        obs_xy  = pts[obs_mask, :2]
        all_xy  = pts[:, :2]

        xy_min = all_xy.min(axis=0) - res
        xy_max = all_xy.max(axis=0) + res
        ox, oy = float(xy_min[0]), float(xy_min[1])
        cols = int(math.ceil((xy_max[0] - ox) / res))
        rows = int(math.ceil((xy_max[1] - oy) / res))

        grid = np.zeros((rows, cols), dtype=np.bool_)
        if obs_xy.shape[0] > 0:
            ci = np.clip(((obs_xy[:, 1] - oy) / res).astype(int), 0, rows - 1)
            cj = np.clip(((obs_xy[:, 0] - ox) / res).astype(int), 0, cols - 1)
            grid[ci, cj] = True

        # inflate (pure numpy, no scipy)
        if inflate_cells > 0:
            inflated = grid.copy()
            for di in range(-inflate_cells, inflate_cells + 1):
                for dj in range(-inflate_cells, inflate_cells + 1):
                    if di * di + dj * dj > inflate_cells * inflate_cells:
                        continue
                    sr0, sr1 = max(0, -di), min(rows, rows - di)
                    sc0, sc1 = max(0, -dj), min(cols, cols - dj)
                    dr0, dr1 = max(0,  di), min(rows, rows + di)
                    dc0, dc1 = max(0,  dj), min(cols, cols + dj)
                    inflated[dr0:dr1, dc0:dc1] |= grid[sr0:sr1, sc0:sc1]
            grid = inflated

        self._occ_grid    = grid
        self._grid_origin = (ox, oy)
        self._grid_rows   = rows
        self._grid_cols   = cols

        occ_pct = 100.0 * grid.sum() / grid.size
        print(cyan("[BenchmarkRecorder] global grid built: {}x{} "
                   "(res={:.2f}m, slice z=[{:.2f},{:.2f}], "
                   "inflate={:.2f}m, occ={:.1f}%)".format(
                       rows, cols, res,
                       self._slice_z_lo, self._slice_z_hi,
                       self._inflate_r, occ_pct)))

    # ══════════════════ ideal-path A* ═════════════════════════

    def _world_to_grid(self, wx, wy):
        ox, oy = self._grid_origin
        res = self._grid_res
        j = int(round((wx - ox) / res))
        i = int(round((wy - oy) / res))
        return (np.clip(i, 0, self._grid_rows - 1),
                np.clip(j, 0, self._grid_cols - 1))

    def _clear_around(self, ci, cj, radius=3):
        """Clear a small neighbourhood so start/goal are not inside obstacle."""
        grid = self._occ_grid
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ni, nj = ci + di, cj + dj
                if 0 <= ni < self._grid_rows and 0 <= nj < self._grid_cols:
                    grid[ni, nj] = False

    def _compute_ideal_path(self):
        sx, sy = self.start_xy
        gx, gy = self.goal_xy
        si, sj = self._world_to_grid(sx, sy)
        gi, gj = self._world_to_grid(gx, gy)

        if self._occ_grid[si, sj]:
            print(cyan("[BenchmarkRecorder] A*: start in obstacle, "
                       "clearing neighbourhood"))
            self._clear_around(si, sj)
        if self._occ_grid[gi, gj]:
            print(cyan("[BenchmarkRecorder] A*: goal in obstacle, "
                       "clearing neighbourhood"))
            self._clear_around(gi, gj)

        print(cyan("[BenchmarkRecorder] running A*  ({:.1f},{:.1f}) → "
                   "({:.1f},{:.1f}) …".format(sx, sy, gx, gy)))

        path_idx = astar_grid(self._occ_grid, si, sj, gi, gj)

        if path_idx is None:
            rospy.logwarn("[BenchmarkRecorder] A* FAILED: no path found")
            self.ideal_path_length = -1.0
            self._ideal_computed = True
            return

        # convert to world & measure
        res = self._grid_res
        ox, oy = self._grid_origin
        path_w = [(ox + pj * res, oy + pi * res) for (pi, pj) in path_idx]

        total = 0.0
        for k in range(1, len(path_w)):
            total += math.hypot(path_w[k][0] - path_w[k - 1][0],
                                path_w[k][1] - path_w[k - 1][1])

        self.ideal_path_length = total
        self._ideal_computed = True

        euclid = math.hypot(gx - sx, gy - sy)
        print(bold_cyan("=" * 44))
        print(bold_cyan("    Ideal Path (A* ground truth)"))
        print(bold_cyan("=" * 44))
        print(cyan("  Start        : ({:.2f}, {:.2f})".format(sx, sy)))
        print(cyan("  Goal         : ({:.2f}, {:.2f})".format(gx, gy)))
        print(cyan("  Euclidean    : {:.2f} m".format(euclid)))
        print(cyan("  A* path len  : {:.2f} m".format(total)))
        if euclid > 1e-3:
            print(cyan("  Length scale : {:.2f}".format(total / euclid)))
        print(cyan("  Waypoints    : {}".format(len(path_w))))
        print(bold_cyan("=" * 44))

        self._publish_ideal_vis(path_w)

    def _publish_ideal_vis(self, path_w):
        # Path msg
        pmsg = Path()
        pmsg.header.stamp = rospy.Time.now()
        pmsg.header.frame_id = "world"
        for (wx, wy) in path_w:
            ps = PoseStamped()
            ps.header = pmsg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = self.plan_z
            ps.pose.orientation.w = 1.0
            pmsg.poses.append(ps)
        self.pub_ideal_path.publish(pmsg)

        # Marker (thick cyan line)
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
        for (wx, wy) in path_w:
            p = Point()
            p.x = wx
            p.y = wy
            p.z = self.plan_z + 0.05
            m.points.append(p)
        self.pub_ideal_marker.publish(m)

    # ══════════════════ navigation callbacks ═══════════════════

    def _goal_cb(self, msg):
        gx = msg.pose.position.x
        gy = msg.pose.position.y

        self.goal_xy     = (gx, gy)
        self.start_time  = rospy.Time.now()
        self.path_length = 0.0
        self.prev_xy     = None
        self.vel_samples = []
        self.start_xy    = None
        self.state       = self.NAVIGATING
        self.ideal_path_length = -1.0
        self._ideal_computed   = False

        print(purple("[BenchmarkRecorder] goal received ({:.2f}, {:.2f}) -- "
                     "recording started".format(gx, gy)))

    def _odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.state != self.NAVIGATING:
            return

        if self.start_xy is None:
            self.start_xy = (x, y)
            self.prev_xy  = (x, y)
            # grid ready → compute ideal path now
            if self._occ_grid is not None and not self._ideal_computed:
                self._compute_ideal_path()

        dx = x - self.prev_xy[0]
        dy = y - self.prev_xy[1]
        self.path_length += math.hypot(dx, dy)
        self.prev_xy = (x, y)

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.vel_samples.append(math.hypot(vx, vy))

        if math.hypot(x - self.goal_xy[0], y - self.goal_xy[1]) < self.goal_thresh:
            self._on_goal_reached()

    def _map_cb(self, msg):
        data = msg.data
        total = len(data)
        if total == 0:
            return
        known = sum(1 for v in data if v >= 0)
        self.known_area_ratio = known / total

    # ══════════════════ result output ═════════════════════════

    def _on_goal_reached(self):
        travel_time = (rospy.Time.now() - self.start_time).to_sec()
        sx, sy = self.start_xy
        gx, gy = self.goal_xy

        euclid = math.hypot(gx - sx, gy - sy)
        length_scale_str = ("{:.2f}".format(round(self.path_length / euclid, 2))
                            if euclid > 1e-3 else "N/A")

        ideal_len = self.ideal_path_length
        if ideal_len > 1e-3:
            ideal_len_r    = round(ideal_len, 2)
            path_ratio_str = "{:.2f}".format(round(self.path_length / ideal_len, 2))
        else:
            ideal_len_r    = "N/A"
            path_ratio_str = "N/A"

        avg_vel = (sum(self.vel_samples) / len(self.vel_samples)
                   if self.vel_samples else 0.0)

        travel_time_r  = round(travel_time, 2)
        path_length_r  = round(self.path_length, 2)
        avg_vel_r      = round(avg_vel, 2)
        known_ratio_r  = round(self.known_area_ratio, 2)

        print(bold_purple("=" * 46))
        print(bold_purple("         Benchmark Result"))
        print(bold_purple("=" * 46))
        print(purple("  Algorithm      : {}".format(self.algorithm)))
        print(purple("  Map            : {}".format(self.map_name)))
        print(purple("  Start          : ({:.2f}, {:.2f})".format(sx, sy)))
        print(purple("  Goal           : ({:.2f}, {:.2f})".format(gx, gy)))
        print(purple("  Travel time    : {:.2f} s".format(travel_time_r)))
        print(purple("  Path length    : {:.2f} m".format(path_length_r)))
        print(purple("  Ideal path len : {}".format(ideal_len_r)))
        print(purple("  Length scale   : {} (actual/euclid)".format(
            length_scale_str)))
        print(purple("  Path ratio     : {} (actual/ideal)".format(
            path_ratio_str)))
        print(purple("  Avg velocity   : {:.2f} m/s".format(avg_vel_r)))
        print(purple("  Known area     : {:.2f}".format(known_ratio_r)))
        print(bold_purple("=" * 46))

        now_str  = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = ("{map}_{alg}_({sx:.1f},{sy:.1f})_to_"
                    "({gx:.1f},{gy:.1f})_{t}.csv").format(
                        map=self.map_name, alg=self.algorithm,
                        sx=sx, sy=sy, gx=gx, gy=gy, t=now_str)
        csv_path = os.path.join(self.output_dir, csv_name)

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "algorithm", "map_name",
                "start_x", "start_y", "goal_x", "goal_y",
                "travel_time_s", "path_length_m", "ideal_path_length_m",
                "length_scale", "path_ratio",
                "avg_velocity_m_s", "known_area_ratio",
            ])
            w.writerow([
                self.algorithm, self.map_name,
                round(sx, 2), round(sy, 2),
                round(gx, 2), round(gy, 2),
                travel_time_r, path_length_r, ideal_len_r,
                length_scale_str, path_ratio_str,
                avg_vel_r, known_ratio_r,
            ])

        print(purple("[BenchmarkRecorder] saved -> {}".format(
            os.path.abspath(csv_path))))
        self.state = self.IDLE


# ─────────────────────────────────────────────────────────────

def main():
    try:
        BenchmarkRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
