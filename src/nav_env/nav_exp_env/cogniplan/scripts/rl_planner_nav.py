#!/home/songhy/miniconda3/envs/forexnav/bin/python3
# -*- coding: utf-8 -*-
"""
CogniPlan Navigation ROS Node (Rewritten)

Faithfully uses the original CogniPlan library classes:
  - Graph_generator  for graph construction & incremental updates
  - Node             for per-node features (utility, direction_vector)
  - PolicyNet        for 100% RL greedy action selection (no goal heuristic)
  - Evaluator + post_process  for GAN map prediction
  - A*               for multi-waypoint path output

Data flow:
  /local_sensing/occupancy_grid  -->  256x256 belief
  /odom                          -->  robot_pos in 256x256
  /move_base_simple/goal         -->  target_pos in 256x256
       |
       v
  GAN predict  -->  Graph_generator  -->  get_observations  -->  PolicyNet
       |                                                            |
       v                                                            v
  predict_map                                               next_position
       |                                                            |
       +----->  A* path (robot -> next -> goal)  -->  /planning/raw_path
"""

import warnings
warnings.simplefilter("ignore", UserWarning)

import rospy
import numpy as np
import torch
import os
import sys
import time
import yaml
import copy
from PIL import Image
import torchvision.transforms as transforms
from skimage.measure import block_reduce

from std_msgs.msg import Float32, Header
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2

# ---------------------------------------------------------------------------
# Add CogniPlan source path so we can import the *original* library classes
# ---------------------------------------------------------------------------
import rospkg
_pkg_path = rospkg.RosPack().get_path('nav_exp_env')
COGNIPLAN_PATH = os.path.abspath(
    os.path.join(_pkg_path, '..', '..', 'planner', 'CogniPlan', 'CogniPlan'))
sys.path.insert(0, COGNIPLAN_PATH)

from planner.graph_generator import Graph_generator
from planner.node import Node
from planner.graph import Graph, a_star
from planner.model import PolicyNet
from mapinpaint.networks import Generator
from mapinpaint.evaluator import Evaluator

# ---------------------------------------------------------------------------
# Constants (matching training configuration in test_parameter.py)
# ---------------------------------------------------------------------------
CANVAS_SIZE = 256        # all planning in 256x256 pixel space
INPUT_DIM = 9
EMBEDDING_DIM = 128
K_SIZE = 20
N_GEN_SAMPLE = 4
SENSOR_RANGE = 40        # pixels (in 256x256 space)


# ===================================================================
# Main ROS Planner
# ===================================================================
class RLPlannerNav:

    def __init__(self):
        rospy.init_node('rl_planner_nav', anonymous=True)

        self.device = 'cpu'

        # ---- ROS parameters ----
        self.replan_freq = rospy.get_param('~replanning_frequency', 2.0)
        self.goal_reached_thr = rospy.get_param('~goal_reached_threshold', 0.5)
        self.publish_graph = rospy.get_param('~publish_graph', True)

        model_path = rospy.get_param(
            '~model_path',
            os.path.join(COGNIPLAN_PATH, 'checkpoints/cogniplan_nav_pred7'))
        generator_path = rospy.get_param(
            '~generator_path',
            os.path.join(COGNIPLAN_PATH, 'checkpoints/wgan_inpainting'))

        # ---- Map metadata (filled on first OccupancyGrid) ----
        self.map_origin_x = None
        self.map_origin_y = None
        self.map_width = None          # original pixel count
        self.map_height = None
        self.map_resolution = None     # m / pixel
        self.map_world_w = None        # metres
        self.map_world_h = None

        # ---- Planning state ----
        self.belief_256 = None         # (256,256) int  255/1/127
        self.robot_pos_world = None    # [x,y] metres
        self.robot_pos_256 = None      # [x,y] in 256-canvas
        self.goal_pos_world = None
        self.goal_pos_256 = None

        self.waiting_for_goal = True
        self.done = False
        self.visited_positions = []    # list of np.array in 256 space

        # ---- Step-and-wait execution state ----
        self.current_local_goal_256 = None   # RL-selected local goal
        self.executing = False               # True while moving to local goal
        self.local_goal_reach_thr = 10.0     # pixels in 256 space (~0.8m)

        # ---- GAN prediction ----
        self.pred_mean_belief = None
        self.pred_max_belief = None
        self.predictor = None

        # ---- Load models ----
        self._load_models(model_path, generator_path)

        # ---- ROS I/O ----
        rospy.Subscriber('/projected_map', OccupancyGrid,
                         self._cb_map, queue_size=1)
        rospy.Subscriber('/state_estimation', Odometry,
                         self._cb_odom, queue_size=1)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped,
                         self._cb_goal, queue_size=1)

        self.raw_path_pub = rospy.Publisher(
            '/planning/raw_path', Path, queue_size=1)
        self.runtime_pub = rospy.Publisher(
            '/runtime', Float32, queue_size=1)
        self.pred_map_pub = rospy.Publisher(
            '/predicted_map', OccupancyGrid, queue_size=1)

        if self.publish_graph:
            self.node_pub = rospy.Publisher(
                '/planner_nodes', PointCloud2, queue_size=1)
            self.frontier_pub = rospy.Publisher(
                '/frontiers', PointCloud2, queue_size=1)

        rospy.loginfo("CogniPlan Nav initialized (100%% RL, original library)")
        rospy.loginfo("  PolicyNet: %s", model_path)
        rospy.loginfo("  Generator: %s", generator_path)
        rospy.loginfo("  Waiting for map and goal ...")

        # Block until first map arrives
        while self.belief_256 is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # Planning timer
        rospy.Timer(rospy.Duration(1.0 / self.replan_freq), self._run)

    # ===============================================================
    # Model loading
    # ===============================================================
    def _load_models(self, model_path, generator_path):
        rospy.loginfo("Loading models ...")

        # PolicyNet
        self.policy_net = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(self.device)
        ckpt_file = os.path.join(model_path, 'checkpoint.pth')
        ckpt = torch.load(ckpt_file, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy_model'])
        self.policy_net.eval()
        rospy.loginfo("  PolicyNet loaded from %s", ckpt_file)

        # WGAN Generator -> Evaluator
        cfg_file = os.path.join(generator_path, 'config.yaml')
        with open(cfg_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        gen = Generator(config['netG'], False)
        gen_files = sorted(
            [fn for fn in os.listdir(generator_path)
             if fn.startswith('gen') and fn.endswith('.pt')])
        gen_path = os.path.join(generator_path, gen_files[0])
        gen.load_state_dict(torch.load(gen_path, map_location=self.device))
        self.predictor = Evaluator(config, gen, cuda=False, nsample=N_GEN_SAMPLE)
        rospy.loginfo("  Generator loaded from %s", gen_path)

    # ===============================================================
    # Coordinate helpers
    # ===============================================================
    def _world_to_256(self, world_xy):
        """World (metres) -> 256-canvas pixel [x, y]."""
        px = (world_xy[0] - self.map_origin_x) / self.map_world_w * CANVAS_SIZE
        py = (world_xy[1] - self.map_origin_y) / self.map_world_h * CANVAS_SIZE
        return np.array([px, py])

    def _p256_to_world(self, p256):
        """256-canvas pixel [x, y] -> world (metres)."""
        wx = p256[0] / CANVAS_SIZE * self.map_world_w + self.map_origin_x
        wy = p256[1] / CANVAS_SIZE * self.map_world_h + self.map_origin_y
        return np.array([wx, wy])

    # ===============================================================
    # ROS callbacks
    # ===============================================================
    def _cb_map(self, msg):
        w = msg.info.width
        h = msg.info.height
        self.map_width = w
        self.map_height = h
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.map_world_w = w * self.map_resolution
        self.map_world_h = h * self.map_resolution

        # ROS -> CogniPlan convention  (0->255 free, 100->1 occ, -1->127 unk)
        ros_map = np.array(msg.data, dtype=np.int8).reshape(h, w)
        belief = np.full((h, w), 127, dtype=np.uint8)
        belief[ros_map == 0] = 255
        belief[ros_map == 100] = 1
        # ros_map == -1 stays 127

        # Resize to 256x256
        belief_img = Image.fromarray(belief).resize(
            (CANVAS_SIZE, CANVAS_SIZE), Image.NEAREST)
        self.belief_256 = np.array(belief_img).astype(int)

    def _cb_odom(self, msg):
        if self.map_origin_x is None:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.robot_pos_world = np.array([x, y])
        self.robot_pos_256 = self._world_to_256(self.robot_pos_world)

    def _cb_goal(self, msg):
        if self.map_origin_x is None:
            return
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.goal_pos_world = np.array([x, y])
        self.goal_pos_256 = self._world_to_256(self.goal_pos_world)
        self.waiting_for_goal = False
        self.done = False
        self.visited_positions = []
        self.current_local_goal_256 = None
        self.executing = False
        rospy.loginfo("Goal received: world=(%.2f, %.2f)  px256=(%.1f, %.1f)",
                      x, y, self.goal_pos_256[0], self.goal_pos_256[1])

    # ===============================================================
    # Frontier detection  (matches Env.find_frontier exactly)
    # ===============================================================
    @staticmethod
    def _find_frontiers(belief):
        resolution = 2
        downsampled = block_reduce(
            belief.copy(), block_size=(resolution, resolution), func=np.min)
        y_len, x_len = downsampled.shape
        mapping = (downsampled == 127).astype(int)
        mapping = np.pad(mapping, ((1, 1), (1, 1)),
                         'constant', constant_values=0)
        fro_map = (mapping[2:][:, 1:x_len + 1]
                   + mapping[:y_len][:, 1:x_len + 1]
                   + mapping[1:y_len + 1][:, 2:]
                   + mapping[1:y_len + 1][:, :x_len]
                   + mapping[:y_len][:, 2:]
                   + mapping[2:][:, :x_len]
                   + mapping[2:][:, 2:]
                   + mapping[:y_len][:, :x_len])
        ind_free = np.where(downsampled.ravel(order='F') == 255)[0]
        ind_fron = np.intersect1d(
            np.where(fro_map.ravel(order='F') > 1)[0],
            np.where(fro_map.ravel(order='F') < 8)[0])
        ind_to = np.intersect1d(ind_free, ind_fron)
        x = np.linspace(0, x_len - 1, x_len)
        y = np.linspace(0, y_len - 1, y_len)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        f = points[ind_to].astype(int) * resolution
        return f

    # ===============================================================
    # GAN map prediction  (matches Env.pre_process_input + update_predict_map)
    # ===============================================================
    def _update_predict_map(self, belief_256):
        width_in, height_in, _ = self.predictor.config['image_shape']
        h_map, w_map = belief_256.shape

        # --- pre_process_input ---
        pad = h_map < width_in and w_map < height_in
        if pad:
            pad_top = (width_in - h_map) // 2
            pad_left = (height_in - w_map) // 2
            pad_bottom = width_in - h_map - pad_top
            pad_right = height_in - w_map - pad_left
            belief_for_gan = np.pad(
                belief_256.astype(np.uint8),
                ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
        else:
            belief_for_gan = belief_256.astype(np.uint8)

        mask_np = belief_for_gan.copy()
        mask_np[mask_np != 127] = 0
        mask_np[mask_np == 127] = 255

        x_raw_img = Image.fromarray(belief_256.astype(np.uint8)).convert('L')
        x_belief_img = Image.fromarray(belief_for_gan).convert('L')
        mask_img = Image.fromarray(mask_np.astype(np.uint8)).convert('1')
        if not pad:
            x_belief_img = transforms.Resize((width_in, height_in))(x_belief_img)
            mask_img = transforms.Resize((width_in, height_in))(mask_img)

        x_belief_t = transforms.ToTensor()(x_belief_img).unsqueeze(0).to(
            self.device).mul_(2).add_(-1)
        x_raw_t = transforms.ToTensor()(x_raw_img).unsqueeze(0).to(
            self.device).mul_(2).add_(-1)
        mask_t = transforms.ToTensor()(mask_img).unsqueeze(0).to(self.device)

        # --- update_predict_map  (with post_process) ---
        onehots = torch.tensor(
            [[0.333, 0.333, 0.333], [1, 0, 0], [0, 1, 0], [0, 0, 1],
             [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]
        ).unsqueeze(1).float().to(self.device)

        predictions = []
        for i in range(self.predictor.nsample):
            _, x_inpaint = self.predictor.eval_step(
                x_belief_t, mask_t, onehots[i], belief_256.shape)
            x_pp = self.predictor.post_process(
                x_inpaint, x_raw_t, kernel_size=5)
            x_pp = np.where(x_pp > 0, 255, 1)
            predictions.append(x_pp)

        self.pred_mean_belief = np.mean(predictions, axis=0)
        self.pred_max_belief = np.max(predictions, axis=0)

    # ===============================================================
    # Build observations  (matches TestWorker.get_observations)
    # ===============================================================
    def _get_observations(self, node_coords, graph_edges, node_utility,
                          indicator, direction_vector, pred_prob,
                          pred_signal, robot_position):
        # --- normalise ---
        nc = node_coords.copy() / 250.0
        nu = node_utility.copy() / 50.0
        n_nodes = nc.shape[0]

        nu_in = nu.reshape(n_nodes, 1)
        dv = direction_vector.copy().reshape(n_nodes, 3)
        dv[:, 2] /= 40.0
        ind = indicator.copy().reshape(n_nodes, 1)
        pp = pred_prob.copy().reshape(n_nodes, 1) / 255.0
        ps = pred_signal.copy().reshape(n_nodes, 1)

        node_inputs = np.concatenate(
            (nc, nu_in, ind, dv, pp, ps), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(
            self.device)

        node_padding_mask = None          # same as test_worker

        # --- current node ---
        cur_idx = int(np.argmin(
            np.linalg.norm(node_coords - robot_position, axis=1)))
        current_index = torch.tensor([cur_idx]).unsqueeze(0).unsqueeze(0).to(
            self.device)

        # --- build per-node edge list from graph_edges dict ---
        all_edges = []
        for i in range(n_nodes):
            key = str(i)
            if key in graph_edges:
                all_edges.append(list(map(int, graph_edges[key])))
            else:
                all_edges.append([i])     # self-loop fallback

        cur_edge = all_edges[cur_idx]
        real_neighbor = ps[cur_edge]      # (len, 1)  1=real  0=predicted

        # --- adjacency (edge_mask) ---
        size = n_nodes
        bias = np.ones((size, size))
        for i in range(size):
            for j in all_edges[i]:
                if j < size:
                    bias[i][j] = 0
        edge_mask = torch.from_numpy(bias).float().unsqueeze(0).to(self.device)

        # --- current node's neighbours, padded to K_SIZE ---
        edge = list(cur_edge)
        while len(edge) < K_SIZE:
            edge.append(0)
        edge = edge[:K_SIZE]
        edge_inputs = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(
            self.device)

        # --- edge padding mask ---
        edge_padding_mask = torch.zeros(
            (1, 1, K_SIZE), dtype=torch.int64, device=self.device)
        one = torch.ones_like(edge_padding_mask)
        edge_padding_mask = torch.where(edge_inputs == 0, one,
                                        edge_padding_mask)

        # mask predicted (non-real) neighbours
        for idx in range(min(len(real_neighbor), K_SIZE)):
            if real_neighbor[idx][0] == 0:
                edge_padding_mask[0, 0, idx] = 1

        return (node_inputs, edge_inputs, current_index,
                node_padding_mask, edge_padding_mask, edge_mask)

    # ===============================================================
    # Action selection  (matches TestWorker.select_node, greedy=True)
    # ===============================================================
    def _select_node(self, observations, node_coords):
        (node_inputs, edge_inputs, current_index,
         node_padding_mask, edge_padding_mask, edge_mask) = observations

        with torch.no_grad():
            logp = self.policy_net(
                node_inputs, edge_inputs, current_index,
                node_padding_mask, edge_padding_mask, edge_mask)

        # 100% RL, greedy argmax (no goal heuristic blending)
        action_index = torch.argmax(logp, dim=1).long()
        next_idx = edge_inputs[0, 0, action_index.item()].item()
        return node_coords[next_idx].copy()

    # ===============================================================
    # Main planning loop
    # ===============================================================
    def _run(self, event=None):
        t0 = time.time()

        if self.done or self.waiting_for_goal:
            return
        if (self.belief_256 is None or self.robot_pos_256 is None
                or self.goal_pos_256 is None):
            return

        # --- goal-reached check ---
        dist_to_goal = np.linalg.norm(
            self.robot_pos_world - self.goal_pos_world)
        if dist_to_goal < self.goal_reached_thr:
            rospy.loginfo("\033[92mGoal reached!\033[0m")
            self.done = True
            self.waiting_for_goal = True
            self.executing = False
            return

        # --- step-and-wait: skip planning while moving to local goal ---
        if self.executing and self.current_local_goal_256 is not None:
            robot_px = np.clip(self.robot_pos_256.copy().round(),
                               0, CANVAS_SIZE - 1).astype(int)
            dist_to_local = np.linalg.norm(
                robot_px.astype(float) - self.current_local_goal_256.astype(float))
            if dist_to_local > self.local_goal_reach_thr:
                return          # still moving, wait
            # reached local goal -> fall through to replan
            rospy.loginfo("Reached local goal, replanning ...")
            self.executing = False

        try:
            belief = self.belief_256.copy()
            robot_pos = np.clip(self.robot_pos_256.copy().round(),
                                0, CANVAS_SIZE - 1).astype(int)
            goal_pos = np.clip(self.goal_pos_256.copy().round(),
                               0, CANVAS_SIZE - 1).astype(int)

            # ----- 1. frontiers -----
            frontiers = self._find_frontiers(belief)
            if len(frontiers) == 0:
                frontiers = np.array([[0, 0]])  # avoid empty-array issues

            # ----- 2. GAN prediction -----
            self._update_predict_map(belief)

            # ----- 3. Graph construction (original Graph_generator) -----
            gg = Graph_generator(
                map_size=(CANVAS_SIZE, CANVAS_SIZE),
                sensor_range=SENSOR_RANGE,
                k_size=K_SIZE,
                target_position=goal_pos,
                plot=False)
            # Only pass current robot_pos (guaranteed to be in node_coords).
            # Historical positions may not be on the current grid and would
            # cause IndexError in generate_graph's indicator lookup.
            gg.route_node = [robot_pos.copy()]

            # generate_graph: belief as both ground_truth and robot_belief
            gg.generate_graph(robot_pos, belief, belief, frontiers)

            # predict graph on GAN-predicted map
            (node_coords, graph_edges, node_utility,
             indicator, direction_vector,
             pred_prob, pred_signal) = gg.update_predict_graph(
                robot_pos, self.pred_max_belief,
                self.pred_mean_belief, frontiers)

            if len(node_coords) < 2:
                rospy.logwarn("Not enough nodes for planning (%d)",
                              len(node_coords))
                return

            # ----- 4. Observations (TestWorker style) -----
            obs = self._get_observations(
                node_coords, graph_edges, node_utility, indicator,
                direction_vector, pred_prob, pred_signal, robot_pos)

            # ----- 5. 100% RL greedy action -----
            next_pos = self._select_node(obs, node_coords)

            # track visited
            self.visited_positions.append(robot_pos.copy())

            # ----- 6. A* from robot to next_pos (local goal) -----
            waypoints_256 = []
            try:
                dist_astar, route = gg.find_shortest_path(
                    robot_pos, next_pos,
                    gg.pred_node_coords, gg.pred_graph)
                if route is not None and len(route) > 0:
                    for nid in route:
                        waypoints_256.append(
                            gg.pred_node_coords[int(nid)].copy())
            except Exception:
                pass
            # fallback: direct [robot, next_pos]
            if len(waypoints_256) < 2:
                waypoints_256 = [robot_pos, next_pos]

            # mark execution state
            self.current_local_goal_256 = next_pos.copy()
            self.executing = True

            # ----- 7. Publish Path (world coords) -----
            path_msg = Path()
            path_msg.header = Header(stamp=rospy.Time.now(), frame_id="map")
            for wp in waypoints_256:
                w = self._p256_to_world(wp)
                ps = PoseStamped()
                ps.header = path_msg.header
                ps.pose.position.x = float(w[0])
                ps.pose.position.y = float(w[1])
                ps.pose.position.z = 1.0
                ps.pose.orientation.w = 1.0
                path_msg.poses.append(ps)
            self.raw_path_pub.publish(path_msg)

            # runtime
            self.runtime_pub.publish(Float32(data=time.time() - t0))

            # predicted map
            if self.pred_mean_belief is not None:
                self._publish_predicted_map()

            # visualisation
            if self.publish_graph:
                self._visualise(node_coords, frontiers, node_utility)

            nw = self._p256_to_world(next_pos)
            rospy.loginfo(
                "RL step -> local_goal: (%.2f, %.2f)  dist_goal: %.2fm  "
                "nodes: %d  path_wps: %d  dt: %.3fs",
                nw[0], nw[1], dist_to_goal,
                len(node_coords), len(waypoints_256),
                time.time() - t0)

        except Exception as e:
            rospy.logerr("Planning error: %s", e)
            import traceback
            traceback.print_exc()

    # ===============================================================
    # Predicted-map publisher
    # ===============================================================
    def _publish_predicted_map(self):
        pred = self.pred_mean_belief
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.info.width = CANVAS_SIZE
        msg.info.height = CANVAS_SIZE
        # map the 256-canvas back to world frame
        msg.info.resolution = self.map_world_w / CANVAS_SIZE
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.info.origin.orientation.w = 1.0

        ros_map = np.full_like(pred, -1, dtype=np.int8)
        ros_map[pred > 200] = 0      # free
        ros_map[pred < 50] = 100     # occupied
        msg.data = ros_map.flatten().tolist()
        self.pred_map_pub.publish(msg)

    # ===============================================================
    # Visualisation helpers
    # ===============================================================
    def _visualise(self, node_coords, frontiers, node_utility):
        stamp = rospy.Time.now()

        # --- graph nodes ---
        if len(node_coords) > 0:
            pts = []
            for i, c in enumerate(node_coords):
                w = self._p256_to_world(c)
                u = float(node_utility[i]) if i < len(node_utility) else 0.0
                pts.append((w[0], w[1], 0.5, u))
            header = Header(stamp=stamp, frame_id="map")
            fields = [
                PointField(name="x", offset=0,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name="intensity", offset=12,
                           datatype=PointField.FLOAT32, count=1)]
            self.node_pub.publish(
                point_cloud2.create_cloud(header, fields, pts))

        # --- frontiers ---
        if len(frontiers) > 0:
            fps = []
            for f in frontiers:
                w = self._p256_to_world(f)
                fps.append((w[0], w[1], 0.3))
            header = Header(stamp=stamp, frame_id="map")
            fields = [
                PointField(name="x", offset=0,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8,
                           datatype=PointField.FLOAT32, count=1)]
            self.frontier_pub.publish(
                point_cloud2.create_cloud(header, fields, fps))


# ===================================================================
def main():
    try:
        planner = RLPlannerNav()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
