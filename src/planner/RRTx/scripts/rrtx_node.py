#!/usr/bin/env python
"""
RRTx dynamic path planner - rospy node.
Ported from RRTx_main.m (MATLAB, 2017).

Reference paper:
https://pdfs.semanticscholar.org/0cac/84962d0f1176c43b3319d379a6bf478d50fd.pdf

Subscribes:
    /local_sensing/occupancy_grid  (nav_msgs/OccupancyGrid)
    /odom                          (nav_msgs/Odometry)
    /move_base_simple/goal         (geometry_msgs/PoseStamped)

Publishes:
    /planning/pos_cmd   (geometry_msgs/PoseStamped)
    /planning/rrtx_path (nav_msgs/Path)
    /planning/rrtx_tree (visualization_msgs/Marker)
"""

import math
import heapq
import numpy as np
import rospy

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Header


# ---------------------------------------------------------------------------
# RRTx core data structure
# ---------------------------------------------------------------------------
class RRTxTree(object):
    """Stores graph data for the RRTx tree."""

    def __init__(self, max_nodes=6000):
        self.max_nodes = max_nodes
        self.pos = np.full((max_nodes, 2), np.nan)       # node positions
        self.parent = np.full(max_nodes, -1, dtype=int)   # parent index (-1 = none)
        self.cost = np.full(max_nodes, np.inf)             # cost-to-goal (g)
        self.rhs = np.full(max_nodes, np.inf)              # one-step lookahead
        self.neighbours = [[] for _ in range(max_nodes)]   # neighbour indices
        self.edge_wt = [[] for _ in range(max_nodes)]      # edge weights (parallel to neighbours)
        self.children = [set() for _ in range(max_nodes)]
        self.n = 0  # current node count

    def add_node(self, pos):
        idx = self.n
        self.pos[idx] = pos
        self.n += 1
        return idx


# ---------------------------------------------------------------------------
# Priority queue wrapper (min-heap on key)
# ---------------------------------------------------------------------------
class KeyQueue(object):
    def __init__(self):
        self._heap = []   # list of (key1, key2, node_idx)
        self._set = set()

    def push(self, node, key):
        entry = (key[0], key[1], node)
        heapq.heappush(self._heap, entry)
        self._set.add(node)

    def pop(self):
        while self._heap:
            k1, k2, node = heapq.heappop(self._heap)
            if node in self._set:
                self._set.discard(node)
                return node, (k1, k2)
        return None, None

    def top_key(self):
        while self._heap:
            if self._heap[0][2] in self._set:
                return (self._heap[0][0], self._heap[0][1])
            heapq.heappop(self._heap)
        return (np.inf, np.inf)

    def remove(self, node):
        self._set.discard(node)

    def empty(self):
        return len(self._set) == 0

    def contains(self, node):
        return node in self._set

    def update(self, node, key):
        """Remove then re-push (lazy deletion)."""
        self.remove(node)
        self.push(node, key)


# ---------------------------------------------------------------------------
# RRTx Planner Node
# ---------------------------------------------------------------------------
class RRTxNode(object):

    def __init__(self):
        rospy.init_node("rrtx_planner", anonymous=False)

        # ---- parameters ----
        self.epsilon = rospy.get_param("~epsilon", 1.0)            # steer distance (m)
        self.ball_radius = rospy.get_param("~ball_radius", 2.5)    # neighbour radius (m)
        self.sensing_radius = rospy.get_param("~sensing_radius", 5.0)
        self.max_samples = rospy.get_param("~max_samples", 8000)
        self.build_rate = rospy.get_param("~build_rate", 500)      # samples / sec during build
        self.move_rate = rospy.get_param("~move_rate", 2.0)        # waypoint publish rate (Hz)
        self.goal_reach_thresh = rospy.get_param("~goal_reach_thresh", 0.5)
        self.delta = rospy.get_param("~delta", 0.0)
        self.obstacle_cost = rospy.get_param("~obstacle_cost", 1e6)
        self.robot_radius = rospy.get_param("~robot_radius", 0.3)  # inflation (m)
        self.goal_bias = rospy.get_param("~goal_bias", 0.15)       # goal-biased sampling ratio
        self.online_samples = rospy.get_param("~online_samples", 30)  # new samples per move step

        # ---- state ----
        self.tree = RRTxTree(self.max_samples)
        self.queue = KeyQueue()
        self.occ_grid = None          # latest OccupancyGrid msg
        self.occ_data = None          # numpy 2-D array of the grid
        self.occ_origin = None        # (ox, oy)
        self.occ_res = None           # resolution
        self.robot_pos = None         # (x, y)
        self.goal_pos = None          # (x, y)
        self.goal_idx = -1
        self.start_idx = -1
        self.goal_reached = False
        self.tree_built = False
        self.obstacle_nodes = set()       # persistent: nodes confirmed in obstacles
        self.invalidated_edges = set()    # persistent: (min(a,b), max(a,b)) edges already penalized

        # ---- ROS pub / sub ----
        self.pub_path = rospy.Publisher("/planning/raw_path", Path, queue_size=1, latch=True)
        self.pub_tree = rospy.Publisher("/planning/rrtx_tree", Marker, queue_size=1, latch=True)

        rospy.Subscriber("/local_sensing/occupancy_grid", OccupancyGrid, self._cb_occ, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self._cb_odom, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self._cb_goal, queue_size=1)

        rospy.loginfo("[RRTx] Node initialized, waiting for goal ...")

    # ===================================================================
    # Callbacks
    # ===================================================================
    def _cb_occ(self, msg):
        """Cache the latest occupancy grid."""
        self.occ_grid = msg
        w, h = msg.info.width, msg.info.height
        self.occ_data = np.array(msg.data, dtype=np.int8).reshape((h, w))
        self.occ_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.occ_res = msg.info.resolution

    def _cb_odom(self, msg):
        self.robot_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _cb_goal(self, msg):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        rospy.loginfo("[RRTx] Received goal (%.2f, %.2f)", gx, gy)
        self.goal_pos = (gx, gy)
        self._start_planning()

    # ===================================================================
    # Occupancy helpers
    # ===================================================================
    def _is_occupied(self, x, y):
        """Check if world coordinate (x,y) is occupied in the cached grid.
        Unknown / out-of-range cells are treated as FREE (optimistic, for tree building)."""
        if self.occ_data is None:
            return False
        col = int((x - self.occ_origin[0]) / self.occ_res)
        row = int((y - self.occ_origin[1]) / self.occ_res)
        h, w = self.occ_data.shape
        if 0 <= row < h and 0 <= col < w:
            return self.occ_data[row, col] > 50  # >50 means occupied
        return False

    def _is_known_free(self, x, y):
        """Check if (x,y) is KNOWN and FREE in the occupancy grid.
        Returns False for unknown (-1), occupied (>50), or out-of-range cells."""
        if self.occ_data is None:
            return False
        col = int((x - self.occ_origin[0]) / self.occ_res)
        row = int((y - self.occ_origin[1]) / self.occ_res)
        h, w = self.occ_data.shape
        if 0 <= row < h and 0 <= col < w:
            v = self.occ_data[row, col]
            return 0 <= v <= 50  # known and free
        return False

    def _is_edge_free(self, p1, p2, step=None):
        """Check collision along edge by sampling (optimistic: unknown = free)."""
        if step is None:
            step = self.occ_res * 2 if self.occ_res else 0.2
        d = np.linalg.norm(np.array(p2) - np.array(p1))
        if d < 1e-6:
            return not self._is_occupied(p1[0], p1[1])
        n_steps = max(int(d / step), 1)
        for i in range(n_steps + 1):
            t = i / float(n_steps)
            px = p1[0] + t * (p2[0] - p1[0])
            py = p1[1] + t * (p2[1] - p1[1])
            if self._is_occupied(px, py):
                return False
        return True

    def _is_edge_known_free(self, p1, p2, step=None):
        """Strict check: every sampled point along the edge must be KNOWN and FREE.
        Used for direct-to-goal shortcut -- won't cut through unexplored areas."""
        if step is None:
            step = self.occ_res * 2 if self.occ_res else 0.2
        d = np.linalg.norm(np.array(p2) - np.array(p1))
        if d < 1e-6:
            return self._is_known_free(p1[0], p1[1])
        n_steps = max(int(d / step), 1)
        for i in range(n_steps + 1):
            t = i / float(n_steps)
            px = p1[0] + t * (p2[0] - p1[0])
            py = p1[1] + t * (p2[1] - p1[1])
            if not self._is_known_free(px, py):
                return False
        return True

    def _node_in_obstacle(self, idx):
        """Check if node idx sits inside an obstacle (with robot radius inflation)."""
        x, y = self.tree.pos[idx]
        r = self.robot_radius
        for dx in (-r, 0, r):
            for dy in (-r, 0, r):
                if self._is_occupied(x + dx, y + dy):
                    return True
        return False

    # ===================================================================
    # Key computation
    # ===================================================================
    def _get_key(self, node):
        return (min(self.tree.cost[node], self.tree.rhs[node]),
                self.tree.cost[node])

    # ===================================================================
    # RRTx core routines
    # ===================================================================
    def _update_rhs(self, node):
        tree = self.tree
        best_rhs = np.inf
        best_parent = -1
        for j, nb in enumerate(tree.neighbours[node]):
            if nb in self.obstacle_nodes:
                continue
            if tree.parent[nb] == node:
                continue  # avoid cycles
            c = tree.edge_wt[node][j] + tree.rhs[nb]
            if c < best_rhs:
                best_rhs = c
                best_parent = nb
        tree.rhs[node] = best_rhs
        if best_parent >= 0:
            old_parent = tree.parent[node]
            tree.parent[node] = best_parent
            tree.children[best_parent].add(node)
            if old_parent >= 0 and old_parent != best_parent:
                tree.children[old_parent].discard(node)

    def _rewire_neighbours(self, node):
        tree = self.tree
        diff = tree.cost[node] - tree.rhs[node]
        if diff <= self.delta and not np.isnan(diff):
            return
        for j, nb in enumerate(tree.neighbours[node]):
            if tree.parent[nb] == node:
                continue
            new_rhs = tree.edge_wt[node][j] + tree.rhs[node]
            if new_rhs < tree.rhs[nb]:
                old_parent = tree.parent[nb]
                tree.rhs[nb] = new_rhs
                tree.parent[nb] = node
                tree.children[node].add(nb)
                if old_parent >= 0 and old_parent != node:
                    tree.children[old_parent].discard(nb)
                d2 = tree.cost[nb] - tree.rhs[nb]
                if d2 > self.delta or np.isnan(d2):
                    self.queue.update(nb, self._get_key(nb))

    def _reduce_inconsistency(self):
        tree = self.tree
        while not self.queue.empty():
            node, _ = self.queue.pop()
            if node is None:
                break
            diff = tree.cost[node] - tree.rhs[node]
            if diff > self.delta or np.isnan(diff):
                self._update_rhs(node)
                self._rewire_neighbours(node)
            tree.cost[node] = tree.rhs[node]

    def _key_less(self, k1, k2):
        """Lexicographic comparison: k1 < k2."""
        return k1[0] < k2[0] or (k1[0] == k2[0] and k1[1] < k2[1])

    def _reduce_inconsistency_v2(self, curr):
        tree = self.tree
        iterations = 0
        max_iter = tree.n * 2  # safety cap to avoid infinite loop
        while not self.queue.empty() and iterations < max_iter:
            iterations += 1
            tk = self.queue.top_key()
            ck = self._get_key(curr)
            # Continue while: queue has better keys OR curr is inconsistent
            if not self._key_less(tk, ck) and \
               tree.rhs[curr] == tree.cost[curr] and tree.cost[curr] < np.inf:
                break
            node, _ = self.queue.pop()
            if node is None:
                break
            diff = tree.cost[node] - tree.rhs[node]
            if diff > self.delta or np.isnan(diff):
                self._update_rhs(node)
                self._rewire_neighbours(node)
            tree.cost[node] = tree.rhs[node]

    # ===================================================================
    # Sampling helpers
    # ===================================================================
    def _get_sampling_bounds(self):
        ox, oy = self.occ_origin
        h, w = self.occ_data.shape
        return (ox, ox + w * self.occ_res, oy, oy + h * self.occ_res)

    def _sample_point(self, bias_center=None, bias_radius=None):
        """Generate a random sample point, optionally biased toward a center."""
        x_min, x_max, y_min, y_max = self._get_sampling_bounds()

        if bias_center is not None and np.random.random() < self.goal_bias:
            # Sample within bias_radius of center (Gaussian)
            r = bias_radius if bias_radius else self.ball_radius * 3
            pt = np.array(bias_center) + np.random.randn(2) * r * 0.5
            pt[0] = np.clip(pt[0], x_min, x_max)
            pt[1] = np.clip(pt[1], y_min, y_max)
            return pt
        return np.array([np.random.uniform(x_min, x_max),
                         np.random.uniform(y_min, y_max)])

    def _try_add_sample(self, rand_pt):
        """Try to add a single sample to the tree. Returns new node index or -1."""
        tree = self.tree
        if tree.n >= tree.max_nodes:
            return -1
        if self._is_occupied(rand_pt[0], rand_pt[1]):
            return -1

        # Find nearest node
        valid = tree.pos[:tree.n]
        dists = np.linalg.norm(valid - rand_pt, axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_dist = dists[nearest_idx]

        # Steer
        if nearest_dist > self.epsilon:
            direction = (rand_pt - tree.pos[nearest_idx]) / nearest_dist
            rand_pt = tree.pos[nearest_idx] + direction * self.epsilon

        # Skip if too close to existing node (avoid redundancy)
        if nearest_dist < self.epsilon * 0.3:
            return -1

        # Collision check on edge to nearest
        if not self._is_edge_free(tree.pos[nearest_idx], rand_pt):
            return -1

        # Add node
        new_idx = tree.add_node(rand_pt)

        # Find neighbours within ball_radius
        dists2 = np.linalg.norm(tree.pos[:tree.n] - rand_pt, axis=1)
        nb_mask = (dists2 < self.ball_radius) & (dists2 > 1e-9)
        nb_indices = np.where(nb_mask)[0].tolist()
        nb_dists = dists2[nb_mask].tolist()

        # Filter neighbours by edge collision
        clean_nb = []
        clean_wt = []
        for nb_i, nb_d in zip(nb_indices, nb_dists):
            if self._is_edge_free(rand_pt, tree.pos[nb_i]):
                clean_nb.append(nb_i)
                clean_wt.append(nb_d)

        if not clean_nb:
            # Rollback: no valid neighbours
            tree.n -= 1
            tree.pos[tree.n] = np.nan
            return -1

        tree.neighbours[new_idx] = list(clean_nb)
        tree.edge_wt[new_idx] = list(clean_wt)

        # Symmetric: add new_idx as neighbour of each nb
        for nb_i, nb_d in zip(clean_nb, clean_wt):
            tree.neighbours[nb_i].append(new_idx)
            tree.edge_wt[nb_i].append(nb_d)

        # Best parent (min rhs + edge)
        best_cost = np.inf
        best_par = -1
        for j, nb_i in enumerate(clean_nb):
            c = clean_wt[j] + tree.rhs[nb_i]
            if c < best_cost:
                best_cost = c
                best_par = nb_i
        if best_par >= 0:
            tree.parent[new_idx] = best_par
            tree.rhs[new_idx] = best_cost
            tree.children[best_par].add(new_idx)

        # Rewire & reduce
        self._rewire_neighbours(new_idx)
        self._reduce_inconsistency()
        return new_idx

    # ===================================================================
    # Build phase
    # ===================================================================
    def _start_planning(self):
        if self.robot_pos is None:
            rospy.logwarn("[RRTx] No odom yet, cannot plan.")
            return
        if self.occ_data is None:
            rospy.logwarn("[RRTx] No occupancy grid yet, cannot plan.")
            return

        # Reset tree
        self.tree = RRTxTree(self.max_samples)
        self.queue = KeyQueue()
        self.goal_reached = False
        self.tree_built = False
        self.start_idx = -1
        self.obstacle_nodes = set()
        self.invalidated_edges = set()

        # Goal is the root of the tree (cost = 0)
        self.goal_idx = self.tree.add_node(np.array(self.goal_pos))
        self.tree.cost[self.goal_idx] = 0.0
        self.tree.rhs[self.goal_idx] = 0.0

        rospy.loginfo("[RRTx] Building tree from goal (%.2f,%.2f) toward robot (%.2f,%.2f) ...",
                      self.goal_pos[0], self.goal_pos[1],
                      self.robot_pos[0], self.robot_pos[1])

        rate = rospy.Rate(self.build_rate)
        goal_arr = np.array(self.goal_pos)
        start_arr = np.array(self.robot_pos)
        # Distance between start and goal, used to set goal bias radius
        sg_dist = np.linalg.norm(goal_arr - start_arr)
        bias_radius = max(sg_dist * 0.3, self.ball_radius * 3)

        attempt_start_every = 50
        i = 0
        build_limit = int(self.max_samples * 0.75)  # reserve 25% capacity for online sampling
        while i < build_limit and not rospy.is_shutdown():
            i += 1

            # Periodically try to sample the start position directly
            if not self.goal_reached and (i % attempt_start_every == 0):
                rand_pt = np.array(self.robot_pos)
            else:
                # Biased sampling: goal_bias% near goal, rest uniform
                rand_pt = self._sample_point(bias_center=goal_arr, bias_radius=bias_radius)

            new_idx = self._try_add_sample(rand_pt)
            if new_idx < 0:
                continue

            # Check if start connected
            if np.allclose(self.tree.pos[new_idx], self.robot_pos, atol=self.goal_reach_thresh) \
                    and not self.goal_reached:
                self.start_idx = new_idx
                self.goal_reached = True
                rospy.loginfo("[RRTx] Start connected at sample %d, cost=%.2f",
                              i, self.tree.rhs[new_idx])

            if i % 500 == 0:
                self._publish_tree_marker()
                rospy.loginfo("[RRTx] Building ... %d/%d nodes, tree size %d",
                              i, build_limit, self.tree.n)

            rate.sleep()

        self.tree_built = True

        if not self.goal_reached:
            dists = np.linalg.norm(self.tree.pos[:self.tree.n] - np.array(self.robot_pos), axis=1)
            self.start_idx = int(np.argmin(dists))
            if self.tree.rhs[self.start_idx] < np.inf:
                self.goal_reached = True
                rospy.loginfo("[RRTx] Using closest node %d (dist=%.2f) as start proxy, cost=%.2f",
                              self.start_idx, dists[self.start_idx], self.tree.rhs[self.start_idx])
            else:
                rospy.logwarn("[RRTx] Could not find a feasible path to goal!")
                return

        rospy.loginfo("[RRTx] Tree built with %d nodes. Starting execution.", self.tree.n)
        self._publish_tree_marker()
        self._execute_path()

    # ===================================================================
    # Execution phase (move along path, detect obstacles, replan)
    # ===================================================================
    def _online_sample(self, curr_node):
        """Add new samples around the current node and along the path ahead."""
        tree = self.tree
        if tree.n >= tree.max_nodes:
            return 0

        curr_pos = tree.pos[curr_node]
        goal_pos = tree.pos[self.goal_idx]
        added = 0
        for _ in range(self.online_samples):
            if tree.n >= tree.max_nodes:
                break
            r = np.random.random()
            if r < 0.4:
                # Sample near current position (for local detours)
                pt = curr_pos + np.random.randn(2) * self.sensing_radius * 0.5
            elif r < 0.7:
                # Sample along path ahead (between curr and goal)
                t = np.random.random()
                mid = curr_pos + t * (goal_pos - curr_pos)
                pt = mid + np.random.randn(2) * self.ball_radius
            else:
                # Sample near goal (densify goal region)
                pt = goal_pos + np.random.randn(2) * self.ball_radius * 2
            if self._try_add_sample(pt) >= 0:
                added += 1
        return added

    def _invalidate_path_ahead(self, curr_node):
        """Walk the entire parent chain from curr to goal.
        For every edge that NOW crosses an obstacle (based on latest occupancy grid),
        inflate its weight and trigger replanning.
        Returns True if any edge was invalidated."""
        tree = self.tree
        invalidated = False
        node = curr_node
        visited = set()

        while node != self.goal_idx and node >= 0 and node not in visited:
            visited.add(node)
            par = tree.parent[node]
            if par < 0:
                break
            edge_key = (min(node, par), max(node, par))
            if edge_key not in self.invalidated_edges:
                if not self._is_edge_free(tree.pos[node], tree.pos[par]):
                    self._mark_edge_invalid(node, par)
                    tree.cost[node] = np.inf
                    self.queue.update(node, self._get_key(node))
                    invalidated = True
            node = par

        if invalidated:
            self._reduce_inconsistency_v2(curr_node)
        return invalidated

    # ---- helper: advance curr along the parent chain using odom ----
    def _advance_curr(self, curr):
        """Advance curr toward goal as long as the robot is close to the next node."""
        if self.robot_pos is None:
            return curr
        robot = np.array(self.robot_pos)
        while curr != self.goal_idx:
            nxt = self.tree.parent[curr]
            if nxt < 0:
                break
            if np.linalg.norm(robot - self.tree.pos[nxt]) < self.goal_reach_thresh:
                curr = nxt
            else:
                break
        return curr

    def _execute_path(self):
        rate = rospy.Rate(self.move_rate)
        curr = self.start_idx
        max_replan_attempts = 5
        consecutive_failures = 0
        max_consecutive_failures = 3

        # Publish initial path so smoother can start moving
        self._publish_path(curr)

        while curr != self.goal_idx and curr >= 0 and not rospy.is_shutdown():
            # Advance curr based on actual robot position (odom)
            curr = self._advance_curr(curr)
            if curr == self.goal_idx:
                break

            # Fast finish: if direct line to goal is KNOWN free, publish direct path
            if self._is_edge_known_free(self.tree.pos[curr], self.tree.pos[self.goal_idx]):
                rospy.loginfo("[RRTx] Direct line to goal is clear, heading straight.")
                self._publish_path_direct(curr)
                # Wait for robot to actually reach goal
                while not rospy.is_shutdown():
                    if self.robot_pos is not None:
                        d = np.linalg.norm(np.array(self.robot_pos) - self.tree.pos[self.goal_idx])
                        if d < self.goal_reach_thresh:
                            break
                    rate.sleep()
                break

            # 0) Online sampling: densify tree around current area & path ahead
            self._online_sample(curr)

            # 1) Sense obstacles in local area (nodes + edges within sensing_radius)
            changed = self._sense_and_update(curr)

            # 2) Check ENTIRE remaining path to goal against latest occupancy grid
            path_changed = self._invalidate_path_ahead(curr)
            changed = changed or path_changed

            if changed:
                self._publish_tree_marker()

            # 3) Check next step validity; attempt local repair if broken
            nxt = self.tree.parent[curr]
            path_ok = False
            for attempt in range(max_replan_attempts):
                if nxt >= 0 and self.tree.rhs[curr] < np.inf \
                        and self._is_edge_free(self.tree.pos[curr], self.tree.pos[nxt]):
                    path_ok = True
                    break
                rospy.logwarn("[RRTx] Path invalid at node %d (attempt %d), repairing ...",
                             curr, attempt + 1)
                self._update_rhs(curr)
                self._reduce_inconsistency_v2(curr)
                nxt = self.tree.parent[curr]

            if path_ok:
                consecutive_failures = 0
                self._publish_path(curr)
            else:
                consecutive_failures += 1
                rospy.logwarn("[RRTx] Planning failed (%d/%d consecutive). Searching retreat ...",
                             consecutive_failures, max_consecutive_failures)

                if consecutive_failures >= max_consecutive_failures:
                    rospy.logerr("[RRTx] %d consecutive failures. No feasible path. Stopping.",
                                 max_consecutive_failures)
                    break

                retreat = self._find_retreat_node(curr)
                if retreat is not None and retreat != curr:
                    rospy.loginfo("[RRTx] Retreating to node %d (cost=%.2f)",
                                  retreat, self.tree.rhs[retreat])
                    curr = retreat
                    self._publish_path(curr)
                else:
                    rospy.logwarn("[RRTx] No retreat node found.")

            rate.sleep()

        if curr == self.goal_idx:
            rospy.loginfo("[RRTx] Goal reached!")
            self._publish_path(self.goal_idx)

    def _find_retreat_node(self, curr_node):
        """Find the nearest node that has a valid finite-cost path to goal
        AND is directly reachable (edge-free) from curr_node."""
        tree = self.tree
        curr_pos = tree.pos[curr_node]

        # Search all nodes, sorted by distance to current
        dists = np.linalg.norm(tree.pos[:tree.n] - curr_pos, axis=1)
        order = np.argsort(dists)

        for idx in order:
            idx = int(idx)
            if idx == curr_node:
                continue
            # Must have a valid path to goal
            if tree.rhs[idx] >= np.inf or tree.parent[idx] < 0:
                continue
            # Must be close enough to reach directly
            if dists[idx] > self.ball_radius * 2:
                break  # sorted by distance, no point searching farther
            # Must have collision-free direct edge
            if self._is_edge_free(curr_pos, tree.pos[idx]):
                return idx
        return None

    def _mark_edge_invalid(self, a, b):
        """Inflate edge weight between a and b, if not already done.
        Returns True if this is a newly invalidated edge."""
        edge_key = (min(a, b), max(a, b))
        if edge_key in self.invalidated_edges:
            return False
        self.invalidated_edges.add(edge_key)

        tree = self.tree
        try:
            j = tree.neighbours[a].index(b)
            tree.edge_wt[a][j] += self.obstacle_cost
        except ValueError:
            pass
        try:
            k = tree.neighbours[b].index(a)
            tree.edge_wt[b][k] += self.obstacle_cost
        except ValueError:
            pass
        return True

    def _sense_and_update(self, curr_node):
        """Detect obstacles (nodes AND edges) within sensing radius, replan.
        Returns True if any change was made."""
        if self.occ_data is None:
            return False

        tree = self.tree
        cx, cy = tree.pos[curr_node]
        sr2 = self.sensing_radius ** 2

        # Find nodes within sensing radius
        dists2 = np.sum((tree.pos[:tree.n] - np.array([cx, cy])) ** 2, axis=1)
        in_range = np.where(dists2 < sr2)[0]

        changed = False

        # --- Check 1: nodes sitting inside obstacles ---
        new_obstacle_nodes = set()
        for idx in in_range:
            if idx in self.obstacle_nodes:
                continue
            if self._node_in_obstacle(idx):
                new_obstacle_nodes.add(idx)

        # --- Handle new obstacle nodes ---
        for node in new_obstacle_nodes:
            self.obstacle_nodes.add(node)
            for j, nb in enumerate(tree.neighbours[node]):
                if self._mark_edge_invalid(node, nb):
                    changed = True
                # If nb's parent is this obstacle node, nb needs a new parent
                if tree.parent[nb] == node and nb not in self.obstacle_nodes:
                    tree.cost[nb] = np.inf
                    self.queue.update(nb, self._get_key(nb))

        # --- Check 2: edges that cross obstacles ---
        for idx in in_range:
            if idx in self.obstacle_nodes:
                continue
            for j, nb in enumerate(tree.neighbours[idx]):
                if nb in self.obstacle_nodes:
                    continue
                if nb > idx:  # check each edge once
                    edge_key = (idx, nb)
                    if edge_key in self.invalidated_edges:
                        continue
                    if not self._is_edge_free(tree.pos[idx], tree.pos[nb]):
                        self._mark_edge_invalid(idx, nb)
                        changed = True
                        # Queue affected nodes for replanning
                        if tree.parent[idx] == nb:
                            tree.cost[idx] = np.inf
                            self.queue.update(idx, self._get_key(idx))
                        if tree.parent[nb] == idx:
                            tree.cost[nb] = np.inf
                            self.queue.update(nb, self._get_key(nb))

        if not changed:
            return False

        rospy.loginfo("[RRTx] Discovered %d new obstacle nodes, total invalidated edges: %d",
                      len(new_obstacle_nodes), len(self.invalidated_edges))

        # Re-queue current node and reduce inconsistency
        self.queue.update(curr_node, self._get_key(curr_node))
        self._reduce_inconsistency_v2(curr_node)
        return True

    # ===================================================================
    # Publishing
    # ===================================================================
    def _publish_path_direct(self, from_node):
        """Publish a straight-line path from from_node to goal."""
        path_msg = Path()
        path_msg.header = Header(stamp=rospy.Time.now(), frame_id="world")
        for pos in [self.tree.pos[from_node], self.tree.pos[self.goal_idx]]:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = pos[0]
            ps.pose.position.y = pos[1]
            ps.pose.position.z = 1.0
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_path.publish(path_msg)

    def _publish_path(self, from_node):
        path_msg = Path()
        path_msg.header = Header(stamp=rospy.Time.now(), frame_id="world")
        node = from_node
        visited = set()
        while node >= 0 and node not in visited:
            visited.add(node)
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = self.tree.pos[node][0]
            ps.pose.position.y = self.tree.pos[node][1]
            ps.pose.position.z = 1.0
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
            node = self.tree.parent[node]
        self.pub_path.publish(path_msg)

    def _publish_tree_marker(self):
        marker = Marker()
        marker.header = Header(stamp=rospy.Time.now(), frame_id="world")
        marker.ns = "rrtx_tree"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.color = ColorRGBA(0.4, 0.4, 0.4, 0.6)
        marker.pose.orientation.w = 1.0

        tree = self.tree
        for i in range(tree.n):
            par = tree.parent[i]
            if par < 0:
                continue
            p1 = Point(x=tree.pos[i][0], y=tree.pos[i][1], z=1.0)
            p2 = Point(x=tree.pos[par][0], y=tree.pos[par][1], z=1.0)
            marker.points.append(p1)
            marker.points.append(p2)

        self.pub_tree.publish(marker)

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
        node = RRTxNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
