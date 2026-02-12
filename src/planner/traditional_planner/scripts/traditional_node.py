#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Traditional Path Planner - rospy node.

Traditional approach: A* frontend + polynomial smoothing backend.
Replans when path collides with new obstacles.
Stops when robot position is inside an obstacle.

Subscribes:
    /local_sensing/occupancy_grid  (nav_msgs/OccupancyGrid)
    /odom                          (nav_msgs/Odometry)
    /move_base_simple/goal         (geometry_msgs/PoseStamped)

Publishes:
    /planning/raw_path             (nav_msgs/Path)   -- for trajectory smoother
    /planning/traditional_path     (nav_msgs/Path)   -- visualization (full grid path)
"""

import math
import heapq
import numpy as np
import rospy

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Header


class TraditionalPlanner(object):

    def __init__(self):
        rospy.init_node("traditional_planner", anonymous=False)

        # ---- parameters ----
        self.robot_radius = rospy.get_param("~robot_radius", 0.3)
        self.replan_rate = rospy.get_param("~replan_rate", 2.0)
        self.goal_reach_thresh = rospy.get_param("~goal_reach_thresh", 0.5)
        self.path_downsample = rospy.get_param("~path_downsample", 5)
        self.obstacle_thresh = rospy.get_param("~obstacle_thresh", 50)

        # ---- state ----
        self.occ_data = None
        self.occ_origin = None
        self.occ_res = None
        self.occ_width = 0
        self.occ_height = 0
        self.robot_pos = None
        self.goal_pos = None
        self.current_path = None   # list of (x, y) world coords
        self.stopped = False
        self.planning_active = False

        # ---- ROS pub / sub ----
        self.pub_raw_path = rospy.Publisher("/planning/raw_path", Path, queue_size=1, latch=True)
        self.pub_vis_path = rospy.Publisher("/planning/traditional_path", Path, queue_size=1, latch=True)

        rospy.Subscriber("/local_sensing/occupancy_grid", OccupancyGrid, self._cb_occ, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self._cb_odom, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self._cb_goal, queue_size=1)

        # Replan timer
        self.replan_timer = None

        rospy.loginfo("[Traditional] Node initialized, waiting for goal ...")

    # ===================================================================
    # Callbacks
    # ===================================================================
    def _cb_occ(self, msg):
        w, h = msg.info.width, msg.info.height
        self.occ_data = np.array(msg.data, dtype=np.int8).reshape((h, w))
        self.occ_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.occ_res = msg.info.resolution
        self.occ_width = w
        self.occ_height = h

    def _cb_odom(self, msg):
        self.robot_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _cb_goal(self, msg):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        rospy.loginfo("[Traditional] Received goal (%.2f, %.2f)", gx, gy)
        self.goal_pos = (gx, gy)
        self.stopped = False
        self.planning_active = True
        self._do_plan()

        # Start periodic replan check
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

    def _is_occupied_cell(self, row, col):
        if not self._in_bounds(row, col):
            return True  # out of bounds = occupied (conservative)
        return self.occ_data[row, col] > self.obstacle_thresh

    def _is_occupied_world(self, x, y):
        if self.occ_data is None:
            return False
        row, col = self._world_to_grid(x, y)
        if not self._in_bounds(row, col):
            return False
        return self.occ_data[row, col] > self.obstacle_thresh

    def _get_inflated_grid(self):
        """Create an inflated occupancy grid considering robot radius."""
        if self.occ_data is None:
            return None
        inflate_cells = max(int(math.ceil(self.robot_radius / self.occ_res)), 1)
        occupied = (self.occ_data > self.obstacle_thresh).astype(np.uint8)

        # Simple box inflation using convolution
        from scipy.ndimage import binary_dilation
        struct = np.ones((2 * inflate_cells + 1, 2 * inflate_cells + 1), dtype=bool)
        inflated = binary_dilation(occupied, structure=struct).astype(np.uint8)
        return inflated

    # ===================================================================
    # A* algorithm
    # ===================================================================
    def _astar(self, start_rc, goal_rc):
        """A* on the inflated occupancy grid.
        start_rc, goal_rc: (row, col) tuples.
        Returns list of (row, col) from start to goal, or None if no path."""
        inflated = self._get_inflated_grid()
        if inflated is None:
            return None

        sr, sc = start_rc
        gr, gc = goal_rc

        # Clamp start/goal to bounds
        sr = max(0, min(sr, self.occ_height - 1))
        sc = max(0, min(sc, self.occ_width - 1))
        gr = max(0, min(gr, self.occ_height - 1))
        gc = max(0, min(gc, self.occ_width - 1))

        # If start or goal is in obstacle, find nearest free cell
        if inflated[sr, sc]:
            sr, sc = self._nearest_free(inflated, sr, sc)
            if sr is None:
                return None
        if inflated[gr, gc]:
            gr, gc = self._nearest_free(inflated, gr, gc)
            if gr is None:
                return None

        # Heuristic: Euclidean distance
        def h(r, c):
            return math.sqrt((r - gr) ** 2 + (c - gc) ** 2)

        # 8-connected neighbors
        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
        COSTS = [1.0, 1.0, 1.0, 1.0,
                 1.414, 1.414, 1.414, 1.414]

        open_set = []
        heapq.heappush(open_set, (h(sr, sc), 0.0, sr, sc))
        g_score = {(sr, sc): 0.0}
        came_from = {}
        closed = set()

        while open_set:
            f, g, r, c = heapq.heappop(open_set)
            if (r, c) in closed:
                continue
            closed.add((r, c))

            if r == gr and c == gc:
                # Reconstruct path
                path = [(r, c)]
                while (r, c) in came_from:
                    r, c = came_from[(r, c)]
                    path.append((r, c))
                path.reverse()
                return path

            for (dr, dc), cost in zip(DIRS, COSTS):
                nr, nc = r + dr, c + dc
                if not self._in_bounds(nr, nc):
                    continue
                if inflated[nr, nc]:
                    continue
                if (nr, nc) in closed:
                    continue
                ng = g + cost
                if ng < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = ng
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_set, (ng + h(nr, nc), ng, nr, nc))

        return None  # no path found

    def _nearest_free(self, inflated, row, col, max_radius=50):
        """BFS to find nearest free cell from (row, col)."""
        from collections import deque
        queue = deque()
        queue.append((row, col))
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
    # Path simplification (RDP)
    # ===================================================================
    def _simplify_path(self, path_world):
        """Downsample grid path to reduce waypoints.
        Uses uniform downsampling + always keeps start and end."""
        if len(path_world) <= 2:
            return path_world
        step = self.path_downsample
        simplified = [path_world[0]]
        for i in range(step, len(path_world) - 1, step):
            simplified.append(path_world[i])
        simplified.append(path_world[-1])
        return simplified

    # ===================================================================
    # Planning
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

        start_rc = self._world_to_grid(self.robot_pos[0], self.robot_pos[1])
        goal_rc = self._world_to_grid(self.goal_pos[0], self.goal_pos[1])

        rospy.loginfo("[Traditional] Planning from (%.2f,%.2f) to (%.2f,%.2f) ...",
                      self.robot_pos[0], self.robot_pos[1],
                      self.goal_pos[0], self.goal_pos[1])

        grid_path = self._astar(start_rc, goal_rc)
        if grid_path is None:
            rospy.logwarn("[Traditional] No path found!")
            self.current_path = None
            return False

        # Convert to world coordinates
        world_path = [self._grid_to_world(r, c) for r, c in grid_path]
        # Replace first point with exact robot position
        world_path[0] = self.robot_pos

        # Simplify for smoother
        simplified = self._simplify_path(world_path)

        self.current_path = world_path

        rospy.loginfo("[Traditional] Path found: %d grid cells -> %d waypoints",
                      len(grid_path), len(simplified))

        # Publish
        self._publish_raw_path(simplified)
        self._publish_vis_path(world_path)
        return True

    # ===================================================================
    # Replan check (timer callback)
    # ===================================================================
    def _replan_cb(self, event):
        if not self.planning_active:
            return
        if self.robot_pos is None or self.goal_pos is None:
            return

        # Check if goal reached
        dx = self.robot_pos[0] - self.goal_pos[0]
        dy = self.robot_pos[1] - self.goal_pos[1]
        if math.sqrt(dx * dx + dy * dy) < self.goal_reach_thresh:
            rospy.loginfo("[Traditional] Goal reached!")
            self.planning_active = False
            if self.replan_timer is not None:
                self.replan_timer.shutdown()
                self.replan_timer = None
            return

        # 1) Check if robot is in obstacle -> stop
        if self._is_occupied_world(self.robot_pos[0], self.robot_pos[1]):
            if not self.stopped:
                rospy.logwarn("[Traditional] Robot in obstacle! Stopping.")
                self.stopped = True
                self._publish_stop()
            return
        else:
            if self.stopped:
                rospy.loginfo("[Traditional] Robot left obstacle region, resuming.")
                self.stopped = False

        # 2) Check if current path collides with obstacles -> replan
        if self.current_path is not None and self._path_has_collision():
            rospy.loginfo("[Traditional] Path collision detected, replanning ...")
            self._do_plan()

    def _path_has_collision(self):
        """Check if current path crosses any obstacle."""
        if self.current_path is None or self.occ_data is None:
            return False

        # Only check from closest point onward (skip already traversed)
        robot = np.array(self.robot_pos)
        pts = np.array(self.current_path)
        dists = np.linalg.norm(pts - robot, axis=1)
        start_idx = max(int(np.argmin(dists)) - 1, 0)

        step = max(int(self.robot_radius / self.occ_res), 1)
        for i in range(start_idx, len(self.current_path), step):
            x, y = self.current_path[i]
            # Check with robot radius inflation
            for dx_off in (-self.robot_radius, 0, self.robot_radius):
                for dy_off in (-self.robot_radius, 0, self.robot_radius):
                    if self._is_occupied_world(x + dx_off, y + dy_off):
                        return True
        return False

    def _publish_stop(self):
        """Publish empty path to stop the trajectory smoother."""
        path_msg = Path()
        path_msg.header = Header(stamp=rospy.Time.now(), frame_id="world")
        self.pub_raw_path.publish(path_msg)

    # ===================================================================
    # Publishing
    # ===================================================================
    def _publish_raw_path(self, waypoints):
        path_msg = Path()
        path_msg.header = Header(stamp=rospy.Time.now(), frame_id="world")
        for x, y in waypoints:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 1.0
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_raw_path.publish(path_msg)

    def _publish_vis_path(self, waypoints):
        path_msg = Path()
        path_msg.header = Header(stamp=rospy.Time.now(), frame_id="world")
        for x, y in waypoints:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 1.0
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_vis_path.publish(path_msg)

    # ===================================================================
    # Spin
    # ===================================================================
    def spin(self):
        rospy.spin()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        node = TraditionalPlanner()
        node.spin()
    except rospy.ROSInterruptException:
        pass
