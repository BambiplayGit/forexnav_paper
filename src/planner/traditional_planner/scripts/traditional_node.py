#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traditional Path Planner with B-spline Trajectory Command (ROS1)

A* frontend + B-spline trajectory backend.
Replans when the current path collides with newly observed obstacles
or when the robot approaches the local-goal end of the trajectory.

Each replan cycle:
  1. A* from robot to global goal  →  full grid path
  2. Truncate to ~traj_length m    →  local path (waypoints)
  3. B-spline control-point QP optimisation:
       min  λ_fit·||B·Q−P||² + λ_curv·||Δ²Q||² + λ_jerk·||Δ³Q||²
       s.t. Q[0]=start, Q[-3:]=end  (v=0, a=0 at local goal)
     + post-opt collision check → re-opt with tighter fit if needed
  4. Curvature-limited velocity profile (v=0 at local goal)
  5. 50 Hz PositionCommand output  →  /planning/pos_cmd

Subscribes:
    /local_sensing/occupancy_grid_inflate  (nav_msgs/OccupancyGrid)  -- inflated
    /odom                                  (nav_msgs/Odometry)
    /move_base_simple/goal                 (geometry_msgs/PoseStamped)

Publishes:
    /planning/pos_cmd              (quadrotor_msgs/PositionCommand)
    /planning/traditional_path     (nav_msgs/Path)   -- full A* vis
    /planning/bspline_path         (nav_msgs/Path)   -- B-spline vis
"""

import math
import time
import heapq
import threading
from collections import deque

import numpy as np
import rospy

from scipy.interpolate import BSpline, CubicSpline

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped
from quadrotor_msgs.msg import PositionCommand
from std_msgs.msg import Header


def _quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class TraditionalPlanner(object):

    def __init__(self):
        rospy.init_node("traditional_planner", anonymous=False)

        # ---- A* parameters ----
        self.robot_radius = rospy.get_param("~robot_radius", 0.3)
        self.replan_rate = rospy.get_param("~replan_rate", 2.0)
        self.goal_reach_thresh = rospy.get_param("~goal_reach_thresh", 0.5)
        self.obstacle_thresh = rospy.get_param("~obstacle_thresh", 50)
        self.w_heuristic = rospy.get_param("~w_heuristic", 1.5)
        self.cost_near_obs = rospy.get_param("~cost_near_obs", 3.0)
        self.obs_cost_radius = rospy.get_param("~obs_cost_radius", 5)
        self.use_4connected = rospy.get_param("~use_4connected", True)
        self.stop_and_go = rospy.get_param("~stop_and_go", True)
        self.replan_delay = rospy.get_param("~replan_delay", 0.3)
        self.naive_local_goal = rospy.get_param("~naive_local_goal", False)
        self.planning_horizon = rospy.get_param("~planning_horizon", 10.0)

        # ---- Trajectory parameters ----
        self.max_vel = rospy.get_param("~max_vel", 2.0)
        self.max_acc = rospy.get_param("~max_acc", 2.0)
        self.publish_rate = rospy.get_param("~publish_rate", 50.0)
        self.fixed_z = rospy.get_param("~fixed_z", 1.0)
        self.traj_length = rospy.get_param("~traj_length", 5.0)
        self.lookahead_dist = rospy.get_param("~lookahead_dist", 0.3)
        self.n_search = rospy.get_param("~n_search", 200)
        self.w_smooth = rospy.get_param("~w_smooth", 1.0)
        self.end_zone_arc = rospy.get_param("~end_zone_arc", 0.3)
        self.replan_ahead = rospy.get_param("~replan_ahead", 1.5)

        # ---- Velocity profile tuning (legged-robot feasible) ----
        self.alloc_speed_ratio = rospy.get_param("~alloc_speed_ratio", 0.6)
        self.acc_ratio = rospy.get_param("~acc_ratio", 0.4)
        self.lat_acc_ratio = rospy.get_param("~lat_acc_ratio", 0.3)
        self.vel_smooth_window = rospy.get_param("~vel_smooth_window", 0.4)

        # ---- Occupancy grid state ----
        self.occ_data = None
        self.occ_origin = None
        self.occ_res = None
        self.occ_width = 0
        self.occ_height = 0

        # ---- Robot state ----
        self.robot_pos = None
        self.robot_vel = np.zeros(2)
        self.robot_yaw = 0.0
        self.last_yaw = 0.0
        self._yaw_initialized = False

        # ---- Planning state ----
        self.goal_pos = None
        self.current_path = None
        self.planning_active = False
        self._replan_stop_until = None  # stop-and-go: hold until this time

        # ---- B-spline trajectory state (protected by lock) ----
        self.lock = threading.Lock()
        self.cs_x = None
        self.cs_y = None
        self.total_arc = 0.0
        self.traj_valid = False
        self.vel_profile_s = None
        self.vel_profile_v = None

        # ---- Timing statistics ----
        self._plan_times = []
        self._astar_times = []
        self._bspline_times = []

        # ---- ROS I/O ----
        self.pub_cmd = rospy.Publisher(
            "/planning/pos_cmd", PositionCommand, queue_size=1)
        self.pub_vis_path = rospy.Publisher(
            "/planning/traditional_path", Path, queue_size=1, latch=True)
        self.pub_bspline_path = rospy.Publisher(
            "/planning/bspline_path", Path, queue_size=1, latch=True)

        rospy.Subscriber("/local_sensing/occupancy_grid_inflate",
                         OccupancyGrid, self._cb_occ, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self._cb_odom, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped,
                         self._cb_goal, queue_size=1)

        self.replan_timer = None
        self.cmd_timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), self._cmd_timer_cb)

        rospy.loginfo("[Traditional] A*+B-spline PositionCmd (inflated grid). "
                      "v_max=%.1f cruise=%.1f a_ramp=%.2f traj_len=%.1fm "
                      "w_smooth=%.2f",
                      self.max_vel,
                      self.max_vel * self.alloc_speed_ratio,
                      self.max_acc * self.acc_ratio,
                      self.traj_length, self.w_smooth)

    # ===================================================================
    # Callbacks
    # ===================================================================
    def _cb_occ(self, msg):
        w, h = msg.info.width, msg.info.height
        self.occ_data = np.array(msg.data, dtype=np.int8).reshape((h, w))
        self.occ_origin = (msg.info.origin.position.x,
                           msg.info.origin.position.y)
        self.occ_res = msg.info.resolution
        self.occ_width = w
        self.occ_height = h

    def _cb_odom(self, msg):
        self.robot_pos = np.array([msg.pose.pose.position.x,
                                   msg.pose.pose.position.y])
        self.robot_vel = np.array([msg.twist.twist.linear.x,
                                   msg.twist.twist.linear.y])
        self.robot_yaw = _quat_to_yaw(msg.pose.pose.orientation)
        if not self._yaw_initialized:
            self.last_yaw = self.robot_yaw
            self._yaw_initialized = True

    def _cb_goal(self, msg):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        rospy.loginfo("[Traditional] Received goal (%.2f, %.2f)", gx, gy)
        self.goal_pos = (gx, gy)
        self.planning_active = True
        self._do_plan()

        if self.replan_timer is not None:
            self.replan_timer.shutdown()
        self.replan_timer = rospy.Timer(
            rospy.Duration(1.0 / self.replan_rate), self._replan_cb)

    # ===================================================================
    # Occupancy helpers
    # ===================================================================
    def _world_to_grid(self, x, y):
        col = int((x - self.occ_origin[0]) / self.occ_res)
        row = int((y - self.occ_origin[1]) / self.occ_res)
        return row, col

    def _grid_to_world(self, row, col):
        x = self.occ_origin[0] + (col + 0.5) * self.occ_res
        y = self.occ_origin[1] + (row + 0.5) * self.occ_res
        return x, y

    def _in_bounds(self, row, col):
        return 0 <= row < self.occ_height and 0 <= col < self.occ_width

    def _is_occupied_world(self, x, y):
        if self.occ_data is None:
            return False
        row, col = self._world_to_grid(x, y)
        if not self._in_bounds(row, col):
            return False
        return self.occ_data[row, col] != 0

    # ===================================================================
    # A* algorithm  (grid is already inflated by local_sensing)
    # ===================================================================
    def _astar(self, start_rc, goal_rc):
        if self.occ_data is None:
            return None

        t_total_start = time.time()

        occupied = (self.occ_data > self.obstacle_thresh).astype(np.uint8)

        sr, sc = start_rc
        gr, gc = goal_rc
        sr = max(0, min(sr, self.occ_height - 1))
        sc = max(0, min(sc, self.occ_width - 1))
        gr = max(0, min(gr, self.occ_height - 1))
        gc = max(0, min(gc, self.occ_width - 1))

        if occupied[sr, sc]:
            sr, sc = self._nearest_free(occupied, sr, sc)
            if sr is None:
                return None
        if occupied[gr, gc]:
            gr, gc = self._nearest_free(occupied, gr, gc)
            if gr is None:
                return None

        from scipy.ndimage import distance_transform_edt
        t_edt_start = time.time()
        obs_dist = distance_transform_edt(1 - occupied)
        t_edt = time.time() - t_edt_start
        rad = self.obs_cost_radius
        prox_cost = np.where(obs_dist < rad,
                             self.cost_near_obs * (1.0 - obs_dist / rad),
                             0.0)

        w_h = self.w_heuristic
        eucl_dist = math.sqrt((sr - gr) ** 2 + (sc - gc) ** 2)

        def h(r, c):
            return math.sqrt((r - gr) ** 2 + (c - gc) ** 2)

        if self.use_4connected:
            DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            COSTS = [1.0, 1.0, 1.0, 1.0]
        else:
            DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
            COSTS = [1.0, 1.0, 1.0, 1.0,
                     1.414, 1.414, 1.414, 1.414]

        open_set = []
        heapq.heappush(open_set, (w_h * h(sr, sc), 0.0, sr, sc))
        g_score = {(sr, sc): 0.0}
        came_from = {}
        closed = set()

        t_search_start = time.time()

        while open_set:
            f, g, r, c = heapq.heappop(open_set)
            if (r, c) in closed:
                continue
            closed.add((r, c))

            if r == gr and c == gc:
                path = [(r, c)]
                while (r, c) in came_from:
                    r, c = came_from[(r, c)]
                    path.append((r, c))
                path.reverse()

                t_search = time.time() - t_search_start
                t_total = time.time() - t_total_start
                path_cost = g
                ratio = path_cost / max(eucl_dist, 1e-6)
                rospy.loginfo(
                    "[A*-BENCH] w=%.1f | grid=%dx%d (%d cells) | "
                    "eucl=%.0f cells | expanded=%d | path_len=%d | "
                    "path_cost=%.1f | cost/eucl=%.2f | "
                    "t_edt=%.3fs | t_search=%.3fs | t_total=%.3fs",
                    w_h, self.occ_width, self.occ_height,
                    self.occ_width * self.occ_height,
                    eucl_dist, len(closed), len(path),
                    path_cost, ratio,
                    t_edt, t_search, t_total)
                return path

            for (dr, dc), cost in zip(DIRS, COSTS):
                nr, nc = r + dr, c + dc
                if not self._in_bounds(nr, nc):
                    continue
                if occupied[nr, nc]:
                    continue
                if (nr, nc) in closed:
                    continue
                ng = g + cost + prox_cost[nr, nc]
                if ng < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = ng
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_set, (ng + w_h * h(nr, nc), ng, nr, nc))

        t_total = time.time() - t_total_start
        rospy.logwarn("[A*-BENCH] w=%.1f | FAILED | expanded=%d | t_total=%.3fs",
                      w_h, len(closed), t_total)
        return None

    def _nearest_free(self, inflated, row, col, max_radius=50):
        queue = deque([(row, col)])
        visited = {(row, col)}
        while queue:
            r, c = queue.popleft()
            if not inflated[r, c]:
                return r, c
            if abs(r - row) > max_radius or abs(c - col) > max_radius:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return None, None

    # ===================================================================
    # Local path extraction (truncate A* path to traj_length)
    # ===================================================================
    def _extract_local_path(self, world_path):
        if len(world_path) < 2:
            return list(world_path)

        local = [world_path[0]]
        cum_dist = 0.0
        for i in range(1, len(world_path)):
            dx = world_path[i][0] - world_path[i - 1][0]
            dy = world_path[i][1] - world_path[i - 1][1]
            seg = math.sqrt(dx * dx + dy * dy)
            if cum_dist + seg >= self.traj_length:
                remain = self.traj_length - cum_dist
                ratio = remain / max(seg, 1e-9)
                px = world_path[i - 1][0] + ratio * dx
                py = world_path[i - 1][1] + ratio * dy
                local.append((px, py))
                return local
            cum_dist += seg
            local.append(world_path[i])
        return local

    # ===================================================================
    # Planning: A* → local path → B-spline optimisation
    # ===================================================================
    def _do_plan(self):
        if self.robot_pos is None:
            rospy.logwarn("[Traditional] No odom yet, cannot plan.")
            return False
        if self.occ_data is None:
            rospy.logwarn("[Traditional] No occupancy grid yet, cannot plan.")
            return False
        if self.goal_pos is None:
            return False

        t_plan_start = time.time()

        robot_xy = (float(self.robot_pos[0]), float(self.robot_pos[1]))
        start_rc = self._world_to_grid(robot_xy[0], robot_xy[1])

        if self.naive_local_goal:
            dx = self.goal_pos[0] - robot_xy[0]
            dy = self.goal_pos[1] - robot_xy[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > self.planning_horizon:
                ratio = self.planning_horizon / dist
                lg_x = robot_xy[0] + ratio * dx
                lg_y = robot_xy[1] + ratio * dy
            else:
                lg_x, lg_y = self.goal_pos
            goal_rc = self._world_to_grid(lg_x, lg_y)
        else:
            goal_rc = self._world_to_grid(self.goal_pos[0], self.goal_pos[1])

        grid_path = self._astar(start_rc, goal_rc)
        if grid_path is None:
            rospy.logwarn("[Traditional] No A* path found!")
            self.current_path = None
            return False
        t_astar = time.time() - t_plan_start

        world_path = [self._grid_to_world(r, c) for r, c in grid_path]
        world_path[0] = robot_xy
        self.current_path = world_path
        self._publish_vis_path(world_path)

        local_path = self._extract_local_path(world_path)

        local_len = 0.0
        for i in range(1, len(local_path)):
            dx = local_path[i][0] - local_path[i - 1][0]
            dy = local_path[i][1] - local_path[i - 1][1]
            local_len += math.sqrt(dx * dx + dy * dy)

        rospy.loginfo("[Traditional] A* %d cells -> local %d wps (%.1fm)",
                      len(grid_path), len(local_path), local_len)

        wps = [np.array(p) for p in local_path]
        t_bspline_start = time.time()
        self._fit_bspline(wps)
        t_bspline = time.time() - t_bspline_start
        t_plan_total = time.time() - t_plan_start

        self._astar_times.append(t_astar)
        self._bspline_times.append(t_bspline)
        self._plan_times.append(t_plan_total)

        rospy.loginfo(
            "[PLAN-BENCH] t_astar=%.3fs | t_bspline=%.3fs | t_total=%.3fs",
            t_astar, t_bspline, t_plan_total)

        if len(self._plan_times) % 5 == 0:
            at = np.array(self._astar_times)
            bt = np.array(self._bspline_times)
            pt = np.array(self._plan_times)
            rospy.loginfo(
                "[PLAN-STATS] n=%d | astar: mean=%.3f std=%.3f max=%.3f | "
                "bspline: mean=%.3f std=%.3f max=%.3f | "
                "total: mean=%.3f std=%.3f max=%.3f (seconds)",
                len(pt),
                at.mean(), at.std(), at.max(),
                bt.mean(), bt.std(), bt.max(),
                pt.mean(), pt.std(), pt.max())
        return True

    # ===================================================================
    # B-spline control-point optimisation (ego-planner style)
    #
    #   N_ctrl control points Q  (N_ctrl << N_waypoints → naturally smooth)
    #
    #   QP cost:
    #     λ_fit  · ||B·Q − P||²       (waypoint fitting)
    #   + λ_curv · ||Δ²Q||²           (curvature / smoothness)
    #   + λ_jerk · ||Δ³Q||²           (jerk / control effort)
    #
    #   Endpoint constraints:
    #     Q[0]  = start_pos
    #     Q[-3] = Q[-2] = Q[-1] = end_pos   →  v(L)=0, a(L)=0
    #
    #   Post-optimisation collision check against inflated grid;
    #   if collision detected, re-solve with tighter fitting weight.
    # ===================================================================
    def _fit_bspline(self, wps):
        if len(wps) < 2:
            return
        if self.robot_pos is not None:
            wps[0] = self.robot_pos.copy()

        pts = np.array(wps)
        n_wps = len(pts)

        # --- arc-length of A* waypoints ---
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        arc_wp = np.zeros(n_wps)
        arc_wp[1:] = np.cumsum(seg_lens)
        L = arc_wp[-1]
        if L < 0.05:
            return

        # --- cubic clamped B-spline setup ---
        n_ctrl = min(max(10, n_wps // 3), 20)
        if n_ctrl > n_wps:
            n_ctrl = n_wps
        if n_ctrl < 6:
            return
        k = 3

        n_internal = max(n_ctrl - k - 1, 0)
        if n_internal > 0:
            internal = np.linspace(0, L, n_internal + 2)[1:-1]
        else:
            internal = np.array([])
        t = np.concatenate([np.full(k + 1, 0.0), internal,
                            np.full(k + 1, L)])

        # initial control points from linear interpolation
        s_init = np.linspace(0, L, n_ctrl)
        c0x = np.interp(s_init, arc_wp, pts[:, 0])
        c0y = np.interp(s_init, arc_wp, pts[:, 1])

        # --- basis matrix at waypoint arc-lengths ---
        eps = 1e-10
        s_wp = np.clip(arc_wp, t[k] + eps, t[-k - 1] - eps)
        B_fit = np.zeros((n_wps, n_ctrl))
        for j in range(n_ctrl):
            ej = np.zeros(n_ctrl)
            ej[j] = 1.0
            B_fit[:, j] = BSpline(t, ej, k, extrapolate=False)(s_wp)

        # --- smoothness matrices ---
        n2 = max(n_ctrl - 2, 0)
        D2 = np.zeros((n2, n_ctrl))
        for i in range(n2):
            D2[i, i] = 1.0; D2[i, i + 1] = -2.0; D2[i, i + 2] = 1.0

        n3 = max(n_ctrl - 3, 0)
        D3 = np.zeros((n3, n_ctrl))
        for i in range(n3):
            D3[i, i] = 1.0; D3[i, i + 1] = -3.0
            D3[i, i + 2] = 3.0; D3[i, i + 3] = -1.0

        # pin indices: first ctrl-pt → start; last 1-3 → end (v=0, a=0)
        n_pin_end = min(3, n_ctrl - 3)
        pin_end = list(range(n_ctrl - n_pin_end, n_ctrl))

        def _solve_qp(lf, lc, lj):
            H_base = lf * (B_fit.T @ B_fit) + \
                     lc * (D2.T @ D2) + lj * (D3.T @ D3) + \
                     np.eye(n_ctrl) * 1e-8
            cx_o = np.zeros(n_ctrl)
            cy_o = np.zeros(n_ctrl)
            for dim in range(2):
                H = H_base.copy()
                rhs = lf * (B_fit.T @ pts[:, dim])
                H[0, :] = 0.0; H[0, 0] = 1.0
                rhs[0] = pts[0, dim]
                for idx in pin_end:
                    H[idx, :] = 0.0; H[idx, idx] = 1.0
                    rhs[idx] = pts[-1, dim]
                try:
                    c = np.linalg.solve(H, rhs)
                except np.linalg.LinAlgError:
                    c = c0x if dim == 0 else c0y
                if dim == 0:
                    cx_o[:] = c
                else:
                    cy_o[:] = c
            return cx_o, cy_o

        lam_fit = 1.0
        lam_curv = self.w_smooth * 5.0
        lam_jerk = self.w_smooth * 2.0

        cx, cy = _solve_qp(lam_fit, lam_curv, lam_jerk)

        # --- collision check on optimised trajectory ---
        n_check = max(int(L / 0.05), 40)
        s_check = np.linspace(0, L, n_check)
        s_check_c = np.clip(s_check, t[k] + eps, t[-k - 1] - eps)

        def _eval_traj(cx_v, cy_v):
            return (BSpline(t, cx_v, k, extrapolate=False)(s_check_c),
                    BSpline(t, cy_v, k, extrapolate=False)(s_check_c))

        def _has_collision(tx, ty):
            if self.occ_data is None:
                return False
            for i in range(len(tx)):
                if self._is_occupied_world(float(tx[i]), float(ty[i])):
                    return True
            return False

        tx, ty = _eval_traj(cx, cy)
        status = "ok"

        if _has_collision(tx, ty):
            cx, cy = _solve_qp(lam_fit * 5.0,
                                lam_curv * 0.2, lam_jerk * 0.2)
            tx, ty = _eval_traj(cx, cy)
            status = "reopt1"
            if _has_collision(tx, ty):
                cx, cy = _solve_qp(lam_fit * 20.0,
                                    lam_curv * 0.05, lam_jerk * 0.05)
                tx, ty = _eval_traj(cx, cy)
                status = "reopt2"

        # --- dense sample → CubicSpline arc-length refit ---
        n_dense = max(int(L / 0.03), 60)
        s_dense = np.linspace(0, L, n_dense)
        s_dense_c = np.clip(s_dense, t[k] + eps, t[-k - 1] - eps)
        bsx = BSpline(t, cx, k, extrapolate=False)
        bsy = BSpline(t, cy, k, extrapolate=False)
        opt_x = bsx(s_dense_c)
        opt_y = bsy(s_dense_c)
        opt_x[0] = pts[0, 0]; opt_y[0] = pts[0, 1]
        opt_x[-1] = pts[-1, 0]; opt_y[-1] = pts[-1, 1]

        opt_pts = np.column_stack([opt_x, opt_y])
        diffs2 = np.diff(opt_pts, axis=0)
        seg2 = np.linalg.norm(diffs2, axis=1)
        arc = np.zeros(len(opt_pts))
        arc[1:] = np.cumsum(seg2)
        total_arc = arc[-1]
        if total_arc < 0.02:
            return

        try:
            cs_x = CubicSpline(arc, opt_pts[:, 0], bc_type='natural')
            cs_y = CubicSpline(arc, opt_pts[:, 1], bc_type='natural')
        except Exception as e:
            rospy.logwarn("[Traditional] CubicSpline refit failed: %s", e)
            return

        # --- velocity profile (v=0, a=0 at local goal) ---
        robot_speed = float(np.linalg.norm(self.robot_vel))
        prof_s, prof_v = self._build_velocity_profile(
            cs_x, cs_y, total_arc, robot_speed)

        with self.lock:
            self.cs_x = cs_x
            self.cs_y = cs_y
            self.total_arc = total_arc
            self.vel_profile_s = prof_s
            self.vel_profile_v = prof_v
            self.traj_valid = True

        rospy.loginfo("[Traditional] Opt: arc=%.2fm ctrl=%d [%s]",
                      total_arc, n_ctrl, status)
        self._publish_bspline_vis(cs_x, cs_y, total_arc)

    # ===================================================================
    # Curvature-limited velocity profile (legged-robot feasible)
    #
    # Matches ForexNav-MINCO behaviour by:
    #   1. Cruise speed = max_vel * alloc_speed_ratio   (MINCO uses 0.7)
    #   2. Ramp accel   = max_acc * acc_ratio           (gentle accel/decel)
    #   3. Lateral accel = max_acc * lat_acc_ratio       (curvature limit)
    #   4. Gaussian smoothing on final profile           (no jerk spikes)
    #   5. Backward pass enforces v(L)=0
    # ===================================================================
    def _build_velocity_profile(self, cs_x, cs_y, total_arc, start_speed=0.0):
        n_samples = max(int(total_arc / 0.02), 50)
        s_arr = np.linspace(0.0, total_arc, n_samples)
        ds = s_arr[1] - s_arr[0] if n_samples > 1 else total_arc

        dx = cs_x(s_arr, 1)
        dy = cs_y(s_arr, 1)
        ddx = cs_x(s_arr, 2)
        ddy = cs_y(s_arr, 2)

        speed_sq = dx * dx + dy * dy
        speed_32 = np.power(np.maximum(speed_sq, 1e-12), 1.5)
        kappa = np.abs(dx * ddy - dy * ddx) / speed_32

        cruise_vel = self.max_vel * self.alloc_speed_ratio
        a_ramp = self.max_acc * self.acc_ratio
        a_lat = self.max_acc * self.lat_acc_ratio

        v_curv = np.where(kappa > 1e-6,
                          np.sqrt(a_lat / np.maximum(kappa, 1e-6)),
                          cruise_vel)
        v_curv = np.minimum(v_curv, cruise_vel)

        # Forward pass (gentle acceleration)
        v_fwd = np.copy(v_curv)
        v_fwd[0] = min(v_curv[0], max(start_speed, 0.0))
        for i in range(1, n_samples):
            v_max = math.sqrt(v_fwd[i - 1] ** 2 + 2.0 * a_ramp * ds)
            v_fwd[i] = min(v_fwd[i], v_max)

        # Backward pass (gentle deceleration to 0)
        v_bwd = np.copy(v_fwd)
        v_bwd[-1] = 0.0
        for i in range(n_samples - 2, -1, -1):
            v_max = math.sqrt(v_bwd[i + 1] ** 2 + 2.0 * a_ramp * ds)
            v_bwd[i] = min(v_bwd[i], v_max)

        v_bwd = np.clip(v_bwd, 0.0, cruise_vel)

        # Gaussian smoothing to remove jerk spikes
        sigma_samples = max(int(self.vel_smooth_window / ds), 1)
        if sigma_samples >= 2 and n_samples > 5:
            from scipy.ndimage import gaussian_filter1d
            v_smooth = gaussian_filter1d(v_bwd, sigma=sigma_samples,
                                         mode='nearest')
            v_smooth = np.minimum(v_smooth, v_bwd)
            v_smooth[0] = v_bwd[0]
            v_smooth[-1] = 0.0
            v_smooth = np.clip(v_smooth, 0.0, cruise_vel)
            v_bwd = v_smooth

        return s_arr, v_bwd

    def _lookup_speed(self, s, prof_s, prof_v):
        return float(np.interp(s, prof_s, prof_v))

    # ===================================================================
    # PositionCommand timer (50 Hz)
    # ===================================================================
    def _publish_hold_cmd(self):
        """Publish zero-velocity hold at current position."""
        if self.robot_pos is None:
            return
        msg = PositionCommand()
        msg.header = Header(stamp=rospy.Time.now(), frame_id="world")
        msg.position.x = float(self.robot_pos[0])
        msg.position.y = float(self.robot_pos[1])
        msg.position.z = self.fixed_z
        msg.velocity.x = 0.0
        msg.velocity.y = 0.0
        msg.velocity.z = 0.0
        msg.acceleration.x = 0.0
        msg.acceleration.y = 0.0
        msg.acceleration.z = 0.0
        msg.yaw = self.last_yaw
        msg.yaw_dot = 0.0
        msg.kx = [0.0, 0.0, 0.0]
        msg.kv = [0.0, 0.0, 0.0]
        msg.trajectory_flag = msg.TRAJECTORY_STATUS_READY
        self.pub_cmd.publish(msg)

    def _cmd_timer_cb(self, event):
        # Stop-and-go: hold position while waiting for replan delay
        if self._replan_stop_until is not None:
            if rospy.Time.now() < self._replan_stop_until:
                self._publish_hold_cmd()
                return
            self._replan_stop_until = None

        with self.lock:
            if not self.traj_valid or self.robot_pos is None:
                return
            if self.cs_x is None or self.cs_y is None:
                return
            cs_x = self.cs_x
            cs_y = self.cs_y
            total_arc = self.total_arc
            prof_s = self.vel_profile_s
            prof_v = self.vel_profile_v

        # --- Closest point on B-spline ---
        n = self.n_search
        s_samples = np.linspace(0.0, total_arc, n + 1)
        x_samples = cs_x(s_samples)
        y_samples = cs_y(s_samples)
        ddx = x_samples - self.robot_pos[0]
        ddy = y_samples - self.robot_pos[1]
        dist_sq = ddx * ddx + ddy * ddy
        idx_closest = int(np.argmin(dist_sq))
        s_closest = s_samples[idx_closest]

        # --- Lookahead target ---
        s_target = min(s_closest + self.lookahead_dist, total_arc)

        px = float(cs_x(s_target))
        py = float(cs_y(s_target))

        # --- Tangent ---
        dpx_ds = float(cs_x(s_target, 1))
        dpy_ds = float(cs_y(s_target, 1))
        tangent_norm = math.sqrt(dpx_ds ** 2 + dpy_ds ** 2)
        if tangent_norm > 1e-6:
            tx = dpx_ds / tangent_norm
            ty = dpy_ds / tangent_norm
        else:
            tx, ty = 1.0, 0.0

        dist_to_end = total_arc - s_closest

        # --- End zone: local goal  →  hold position, v=0, a=0 ---
        if dist_to_end <= self.end_zone_arc:
            speed = 0.0
            a_tangential = 0.0
            px = float(self.robot_pos[0])
            py = float(self.robot_pos[1])
        else:
            speed = self._lookup_speed(s_closest, prof_s, prof_v)
            a_lim = self.max_acc * self.acc_ratio
            ds_fd = 0.05
            if s_closest + ds_fd <= total_arc:
                v_ahead = self._lookup_speed(s_closest + ds_fd, prof_s, prof_v)
                a_tangential = (v_ahead ** 2 - speed ** 2) / (2.0 * ds_fd + 1e-9)
                a_tangential = max(-a_lim, min(a_tangential, a_lim))
            else:
                a_tangential = -a_lim if speed > 0.01 else 0.0

        vx = tx * speed
        vy = ty * speed
        ax = tx * a_tangential
        ay = ty * a_tangential

        # --- Yaw ---
        if tangent_norm > 1e-6:
            yaw = math.atan2(ty, tx)
            self.last_yaw = yaw
        else:
            yaw = self.last_yaw

        # --- Yaw rate from curvature ---
        yaw_dot = 0.0
        ds_yaw = max(0.01, min(0.02 * total_arc, 0.1))
        if s_target + ds_yaw <= total_arc:
            dpx_next = float(cs_x(s_target + ds_yaw, 1))
            dpy_next = float(cs_y(s_target + ds_yaw, 1))
            n_next = math.sqrt(dpx_next ** 2 + dpy_next ** 2)
            if n_next > 1e-6 and tangent_norm > 1e-6:
                yaw_next = math.atan2(dpy_next, dpx_next)
                dyaw = yaw_next - yaw
                while dyaw > math.pi:
                    dyaw -= 2.0 * math.pi
                while dyaw < -math.pi:
                    dyaw += 2.0 * math.pi
                yaw_dot = (dyaw / ds_yaw) * speed

        # --- Publish PositionCommand ---
        msg = PositionCommand()
        msg.header = Header(stamp=rospy.Time.now(), frame_id="world")
        msg.position.x = px
        msg.position.y = py
        msg.position.z = self.fixed_z
        msg.velocity.x = vx
        msg.velocity.y = vy
        msg.velocity.z = 0.0
        msg.acceleration.x = ax
        msg.acceleration.y = ay
        msg.acceleration.z = 0.0
        msg.yaw = yaw
        msg.yaw_dot = yaw_dot
        msg.kx = [0.0, 0.0, 0.0]
        msg.kv = [0.0, 0.0, 0.0]
        msg.trajectory_flag = msg.TRAJECTORY_STATUS_READY
        self.pub_cmd.publish(msg)

    # ===================================================================
    # Replan check (2 Hz)
    # ===================================================================
    def _replan_cb(self, event):
        if not self.planning_active:
            return
        if self.robot_pos is None or self.goal_pos is None:
            return

        # Global goal reached?
        dx = self.robot_pos[0] - self.goal_pos[0]
        dy = self.robot_pos[1] - self.goal_pos[1]
        if math.sqrt(dx * dx + dy * dy) < self.goal_reach_thresh:
            rospy.loginfo("[Traditional] Goal reached!")
            self.planning_active = False
            with self.lock:
                self.traj_valid = False
            if self.replan_timer is not None:
                self.replan_timer.shutdown()
                self.replan_timer = None
            return

        # Still in stop-and-go waiting phase
        if self._replan_stop_until is not None:
            return

        need_replan = False

        # Approaching local goal end -> extend trajectory
        with self.lock:
            if self.traj_valid and self.cs_x is not None:
                cs_x = self.cs_x
                cs_y = self.cs_y
                total_arc = self.total_arc
                n = 100
                s_samp = np.linspace(0.0, total_arc, n + 1)
                x_samp = cs_x(s_samp)
                y_samp = cs_y(s_samp)
                ddx = x_samp - self.robot_pos[0]
                ddy = y_samp - self.robot_pos[1]
                dsq = ddx * ddx + ddy * ddy
                s_closest = s_samp[int(np.argmin(dsq))]
                dist_to_end = total_arc - s_closest
                if dist_to_end < self.replan_ahead:
                    need_replan = True

        # Current A* path collides with newly observed obstacles
        if self.current_path is not None and self._path_has_collision():
            rospy.loginfo("[Traditional] Path collision, replanning ...")
            need_replan = True

        if need_replan:
            if self.stop_and_go:
                with self.lock:
                    self.traj_valid = False
                delay = rospy.Duration(self.replan_delay)
                self._replan_stop_until = rospy.Time.now() + delay
                rospy.loginfo("[Traditional] Stop-and-go: holding %.2fs before replan",
                              self.replan_delay)
                rospy.Timer(delay, self._delayed_replan, oneshot=True)
            else:
                self._do_plan()

    def _delayed_replan(self, event):
        """Called after stop-and-go delay expires."""
        self._replan_stop_until = None
        self._do_plan()

    def _path_has_collision(self):
        """Check from closest point onward; grid is already inflated."""
        if self.current_path is None or self.occ_data is None:
            return False

        robot = self.robot_pos
        pts = np.array(self.current_path)
        dists = np.linalg.norm(pts - robot, axis=1)
        start_idx = max(int(np.argmin(dists)) - 1, 0)

        step = max(int(self.robot_radius / self.occ_res), 1)
        for i in range(start_idx, len(self.current_path), step):
            x, y = self.current_path[i]
            if self._is_occupied_world(x, y):
                        return True
        return False

    # ===================================================================
    # Visualization publishers
    # ===================================================================
    def _publish_vis_path(self, waypoints):
        path_msg = Path()
        path_msg.header = Header(stamp=rospy.Time.now(), frame_id="world")
        for x, y in waypoints:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = self.fixed_z
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_vis_path.publish(path_msg)

    def _publish_bspline_vis(self, cs_x, cs_y, total_arc):
        path_msg = Path()
        path_msg.header = Header(stamp=rospy.Time.now(), frame_id="odom")
        n_vis = max(int(total_arc / 0.02), 20)
        s_vals = np.linspace(0.0, total_arc, n_vis)
        x_vals = cs_x(s_vals)
        y_vals = cs_y(s_vals)
        for i in range(n_vis):
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(x_vals[i])
            ps.pose.position.y = float(y_vals[i])
            ps.pose.position.z = self.fixed_z
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_bspline_path.publish(path_msg)

    # ===================================================================
    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = TraditionalPlanner()
        node.spin()
    except rospy.ROSInterruptException:
        pass
