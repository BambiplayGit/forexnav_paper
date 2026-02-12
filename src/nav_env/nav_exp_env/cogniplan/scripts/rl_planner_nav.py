#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CogniPlan Navigation ROS Node

This node implements the CogniPlan navigation planner for ROS.
It subscribes to occupancy grid map and odometry, and publishes waypoints
for the robot to follow.

Based on CogniPlan Nav (9-dim input with direction_vector to goal).
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
from sklearn.neighbors import NearestNeighbors
from skimage.measure import block_reduce

from std_msgs.msg import Float32, Header
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2

# Add CogniPlan source path (planner/mapinpaint) relative to this package
import rospkg
_pkg_path = rospkg.RosPack().get_path('nav_exp_env')
COGNIPLAN_NAV_PATH = os.path.abspath(os.path.join(_pkg_path, '..', '..', 'planner', 'CogniPlan', 'CogniPlan'))
sys.path.insert(0, COGNIPLAN_NAV_PATH)

from planner.model import PolicyNet
from mapinpaint.networks import Generator
from mapinpaint.evaluator import Evaluator


# ============== Parameters ==============
class NavParameter:
    # Map parameters (will be updated from ROS params)
    CELL_SIZE = 0.1  # meter (map resolution)
    NODE_RESOLUTION = 8  # pixels between nodes
    
    # Map values (ROS OccupancyGrid convention)
    FREE = 0
    OCCUPIED = 100
    UNKNOWN = -1
    
    # Sensor and utility
    SENSOR_RANGE = 80  # pixels
    UTILITY_RANGE = 40  # pixels
    MIN_UTILITY = 3
    
    # Network
    INPUT_DIM = 9
    EMBEDDING_DIM = 128
    K_SIZE = 20  # number of neighbors
    
    # Planning
    THR_TO_WAYPOINT = 1.0  # meter
    GOAL_REACHED_THR = 0.3  # meter
    
    # Prediction
    N_GEN_SAMPLE = 4


param = NavParameter()


# ============== Helper Classes ==============
class MapInfo:
    def __init__(self, map_array, origin_x, origin_y, cell_size):
        self.map = map_array
        self.map_origin_x = origin_x
        self.map_origin_y = origin_y
        self.cell_size = cell_size


class Node:
    def __init__(self, coords, frontiers, robot_belief, target_position, sensor_range=80):
        self.coords = coords
        self.sensor_range = sensor_range
        self.target_position = target_position
        self.observable_frontiers = []
        self.direction_vector = self.get_direction_vector()
        self.initialize_observable_frontiers(frontiers, robot_belief)
        self.utility = len(self.observable_frontiers)
        self.visited = False
        
    def get_direction_vector(self):
        """Get normalized direction vector to target"""
        dx = self.target_position[0] - self.coords[0]
        dy = self.target_position[1] - self.coords[1]
        mag = np.sqrt(dx**2 + dy**2)
        if mag > 0:
            dx, dy = dx / mag, dy / mag
        mag = min(mag, 40)  # clip
        return [dx, dy, mag]
    
    def initialize_observable_frontiers(self, frontiers, robot_belief):
        if len(frontiers) == 0:
            return
        dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
        frontiers_in_range = frontiers[dist_list < self.sensor_range - 10]
        for point in frontiers_in_range:
            if not self.check_collision(self.coords, point, robot_belief):
                self.observable_frontiers.append(point)
                
    def check_collision(self, start, end, robot_belief):
        """Bresenham line collision check"""
        x0, y0 = int(round(start[0])), int(round(start[1]))
        x1, y1 = int(round(end[0])), int(round(end[1]))
        
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2
        
        h, w = robot_belief.shape
        while 0 <= x < w and 0 <= y < h:
            if x == x1 and y == y1:
                break
            k = robot_belief[y, x]
            if k == 1 or k == 127:  # obstacle or unknown
                return True
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return False
    
    def set_visited(self):
        self.observable_frontiers = []
        self.utility = 0
        self.visited = True


class Graph:
    def __init__(self):
        self.edges = {}
        
    def add_node(self, node_id):
        if node_id not in self.edges:
            self.edges[node_id] = []
            
    def add_edge(self, from_id, to_id, weight=1.0):
        if from_id not in self.edges:
            self.edges[from_id] = []
        if to_id not in self.edges[from_id]:
            self.edges[from_id].append(to_id)
            
    def clear(self):
        self.edges = {}


# ============== Main Planner Class ==============
class RLPlannerNav:
    def __init__(self):
        rospy.init_node('rl_planner_nav', anonymous=True)
        
        self.device = 'cpu'
        
        # Get parameters
        param.CELL_SIZE = rospy.get_param('~map_resolution', 0.1)
        param.THR_TO_WAYPOINT = rospy.get_param('~waypoint_threshold', 1.0)
        param.GOAL_REACHED_THR = rospy.get_param('~goal_reached_threshold', 0.5)
        self.publish_graph = rospy.get_param('~publish_graph', True)
        self.replan_freq = rospy.get_param('~replanning_frequency', 2.0)
        
        # Model paths
        model_path = rospy.get_param('~model_path', 
            os.path.join(COGNIPLAN_NAV_PATH, 'checkpoints/cogniplan_nav_pred7'))
        generator_path = rospy.get_param('~generator_path',
            os.path.join(COGNIPLAN_NAV_PATH, 'checkpoints/wgan_inpainting'))
        
        # State
        self.map_info = None
        self.robot_belief = None  # Current known map (255=free, 1=obstacle, 127=unknown)
        self.robot_position = None  # [x, y] in pixels
        self.robot_position_world = None  # [x, y] in meters
        self.goal_position = None  # [x, y] in pixels
        self.goal_position_world = None  # [x, y] in meters
        
        self.waiting_for_goal = True
        self.done = False
        
        # Current target waypoint (for continuous movement)
        self.current_target_pixel = None
        self.current_target_world = None
        
        # Graph
        self.graph = Graph()
        self.node_coords = None
        self.nodes_list = []
        self.route_node = []
        
        # Prediction
        self.pred_mean_belief = None
        self.pred_max_belief = None
        self.predictor = None
        
        # Load models
        self.load_models(model_path, generator_path)
        
        # ROS subscribers
        rospy.Subscriber('/projected_map', OccupancyGrid, self.map_callback, queue_size=1)
        rospy.Subscriber('/state_estimation', Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        
        # ROS publishers
        self.raw_path_pub = rospy.Publisher('/planning/raw_path', Path, queue_size=1)
        self.runtime_pub = rospy.Publisher('/runtime', Float32, queue_size=1)
        self.pred_map_pub = rospy.Publisher('/predicted_map', OccupancyGrid, queue_size=1)
        
        if self.publish_graph:
            self.node_pub = rospy.Publisher('/planner_nodes', PointCloud2, queue_size=1)
            self.frontier_pub = rospy.Publisher('/frontiers', PointCloud2, queue_size=1)
        
        rospy.loginfo("RL Planner Nav initialized")
        rospy.loginfo("  Model path: {}".format(model_path))
        rospy.loginfo("  Waiting for map and goal...")
        
        # Wait for map
        while self.map_info is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        # Main planning timer
        rospy.Timer(rospy.Duration(1.0 / self.replan_freq), self.run)
        
    def load_models(self, model_path, generator_path):
        """Load PolicyNet and Generator"""
        rospy.loginfo("Loading models...")
        
        # Policy network
        self.policy_net = PolicyNet(param.INPUT_DIM, param.EMBEDDING_DIM).to(self.device)
        checkpoint_file = os.path.join(model_path, 'checkpoint.pth')
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_model'])
        self.policy_net.eval()
        rospy.loginfo("  PolicyNet loaded from {}".format(checkpoint_file))
        
        # Generator for map prediction
        config_file = os.path.join(generator_path, 'config.yaml')
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        
        generator = Generator(config['netG'], False)
        gen_files = [f for f in os.listdir(generator_path) if f.startswith('gen') and f.endswith('.pt')]
        if gen_files:
            gen_file = os.path.join(generator_path, gen_files[0])
            generator.load_state_dict(torch.load(gen_file, map_location=self.device))
            rospy.loginfo("  Generator loaded from {}".format(gen_file))
        
        self.predictor = Evaluator(config, generator, cuda=False, nsample=param.N_GEN_SAMPLE)
        
    def map_callback(self, msg):
        """Process OccupancyGrid message"""
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        
        # Convert to numpy array
        ros_map = np.array(msg.data, dtype=np.int8).reshape(height, width)
        
        # Convert ROS convention to our convention
        # ROS: 0=free, 100=occupied, -1=unknown
        # Ours: 255=free, 1=occupied, 127=unknown
        belief = np.full_like(ros_map, 127, dtype=np.uint8)
        belief[ros_map == 0] = 255  # free
        belief[ros_map == 100] = 1  # occupied
        belief[ros_map == -1] = 127  # unknown
        
        self.robot_belief = belief
        self.map_info = MapInfo(belief, origin_x, origin_y, resolution)
        
    def odom_callback(self, msg):
        """Process Odometry message"""
        if self.map_info is None:
            return
            
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        self.robot_position_world = np.array([x, y])
        
        # Convert to pixel coordinates
        px = (x - self.map_info.map_origin_x) / self.map_info.cell_size
        py = (y - self.map_info.map_origin_y) / self.map_info.cell_size
        self.robot_position = np.array([px, py])
        
    def goal_callback(self, msg):
        """Process goal message"""
        if self.map_info is None:
            return
            
        x = msg.pose.position.x
        y = msg.pose.position.y
        
        self.goal_position_world = np.array([x, y])
        
        # Convert to pixel coordinates
        px = (x - self.map_info.map_origin_x) / self.map_info.cell_size
        py = (y - self.map_info.map_origin_y) / self.map_info.cell_size
        self.goal_position = np.array([px, py])
        
        self.waiting_for_goal = False
        self.done = False
        self.route_node = []
        self.current_target_pixel = None
        self.current_target_world = None
        
        rospy.loginfo("Goal received: world=({:.2f}, {:.2f}), pixel=({:.1f}, {:.1f})".format(
            x, y, px, py))
        
    def world_to_pixel(self, world_pos):
        """Convert world coordinates to pixel coordinates"""
        px = (world_pos[0] - self.map_info.map_origin_x) / self.map_info.cell_size
        py = (world_pos[1] - self.map_info.map_origin_y) / self.map_info.cell_size
        return np.array([px, py])
    
    def pixel_to_world(self, pixel_pos):
        """Convert pixel coordinates to world coordinates"""
        wx = pixel_pos[0] * self.map_info.cell_size + self.map_info.map_origin_x
        wy = pixel_pos[1] * self.map_info.cell_size + self.map_info.map_origin_y
        return np.array([wx, wy])
    
    def find_frontiers(self):
        """Find frontier cells (free cells adjacent to unknown)"""
        if self.robot_belief is None:
            return np.array([])
            
        # Downsample for efficiency
        resolution = 2
        downsampled = block_reduce(self.robot_belief.copy(), 
                                   block_size=(resolution, resolution), func=np.min)
        
        h, w = downsampled.shape
        mapping = (downsampled == 127).astype(int)
        mapping = np.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)
        
        # Count unknown neighbors
        fro_map = (mapping[2:, 1:w+1] + mapping[:h, 1:w+1] + 
                   mapping[1:h+1, 2:] + mapping[1:h+1, :w] +
                   mapping[:h, 2:] + mapping[2:, :w] + 
                   mapping[2:, 2:] + mapping[:h, :w])
        
        ind_free = np.where(downsampled.ravel(order='F') == 255)[0]
        ind_fron = np.where((fro_map.ravel(order='F') > 1) & 
                            (fro_map.ravel(order='F') < 8))[0]
        ind_to = np.intersect1d(ind_free, ind_fron)
        
        x = np.linspace(0, w-1, w)
        y = np.linspace(0, h-1, h)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        
        frontiers = points[ind_to].astype(int) * resolution
        return frontiers
    
    def generate_uniform_points(self, shape):
        """Generate uniform grid points"""
        h, w = shape
        n_points = 30
        x = np.linspace(0, w-1, n_points).round().astype(int)
        y = np.linspace(0, h-1, n_points).round().astype(int)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points
    
    def generate_node_coords(self, belief, robot_pos, goal_pos):
        """Generate node coordinates from free space"""
        # Get free cells
        free_idx = np.where(belief == 255)
        free_cells = np.vstack([free_idx[1], free_idx[0]]).T  # [x, y]
        
        # Get uniform sample points
        uniform_pts = self.generate_uniform_points(belief.shape)
        
        # Filter to free cells
        free_set = set(map(tuple, free_cells))
        node_coords = []
        for pt in uniform_pts:
            if tuple(pt) in free_set:
                node_coords.append(pt)
        
        node_coords = np.array(node_coords) if node_coords else np.array([]).reshape(0, 2)
        
        # Add robot and goal positions
        if len(node_coords) > 0:
            node_coords = np.vstack([robot_pos.reshape(1, 2), 
                                     goal_pos.reshape(1, 2),
                                     node_coords])
        else:
            node_coords = np.vstack([robot_pos.reshape(1, 2), 
                                     goal_pos.reshape(1, 2)])
        
        # Remove duplicates
        node_coords = np.unique(node_coords, axis=0)
        return node_coords
    
    def check_collision_line(self, start, end, belief):
        """Check if line collides with obstacle"""
        x0, y0 = int(round(start[0])), int(round(start[1]))
        x1, y1 = int(round(end[0])), int(round(end[1]))
        
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2
        
        h, w = belief.shape
        while 0 <= x < w and 0 <= y < h:
            if x == x1 and y == y1:
                break
            k = belief[y, x]
            if k == 1 or k == 127:
                return True
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return False
    
    def build_graph(self, node_coords, belief):
        """Build navigation graph with KNN neighbors"""
        self.graph.clear()
        
        if len(node_coords) < 2:
            return
        
        k = min(param.K_SIZE, len(node_coords))
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(node_coords)
        distances, indices = knn.kneighbors(node_coords)
        
        for i, p in enumerate(node_coords):
            for j, neighbor_idx in enumerate(indices[i]):
                neighbor = node_coords[neighbor_idx]
                if not self.check_collision_line(p, neighbor, belief):
                    self.graph.add_node(str(i))
                    self.graph.add_edge(str(i), str(neighbor_idx), distances[i, j])
    
    def update_predict_map(self):
        """Use GAN to predict unknown regions"""
        if self.robot_belief is None or self.predictor is None:
            return
        
        belief = self.robot_belief
        h, w = belief.shape  # h=rows, w=cols
        
        # GAN expects square input, use 256x256
        target_size = 256
        
        # Resize belief to 256x256 for GAN
        belief_img = Image.fromarray(belief).convert('L')
        belief_resized = belief_img.resize((target_size, target_size), Image.NEAREST)
        belief_np = np.array(belief_resized)
        
        # Create mask (unknown regions)
        mask = (belief_np == 127).astype(np.float32) * 255
        mask_img = Image.fromarray(mask.astype(np.uint8)).convert('1')
        
        # Convert to tensor
        x_belief = transforms.ToTensor()(belief_resized).unsqueeze(0).to(self.device)
        x_belief = x_belief.mul_(2).add_(-1)
        mask_t = transforms.ToTensor()(mask_img).unsqueeze(0).to(self.device)
        
        # Generate predictions
        onehots = torch.tensor([[0.333, 0.333, 0.333], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).unsqueeze(1).float()
        predictions = []
        
        for i in range(min(param.N_GEN_SAMPLE, len(onehots))):
            _, x_inpaint = self.predictor.eval_step(x_belief, mask_t, onehots[i], (target_size, target_size))
            # x_inpaint is tensor, convert to numpy
            x_inpaint_np = x_inpaint.squeeze().cpu().numpy()
            x_inpaint_np = np.where(x_inpaint_np > -0.3, 255, 1).astype(np.uint8)
            predictions.append(x_inpaint_np)
        
        # Average predictions
        pred_mean_256 = np.mean(predictions, axis=0)
        pred_max_256 = np.max(predictions, axis=0)
        
        # Resize back to original size
        pred_mean_img = Image.fromarray(pred_mean_256.astype(np.uint8)).resize((w, h), Image.NEAREST)
        pred_max_img = Image.fromarray(pred_max_256.astype(np.uint8)).resize((w, h), Image.NEAREST)
        
        self.pred_mean_belief = np.array(pred_mean_img).astype(np.float32)
        self.pred_max_belief = np.array(pred_max_img).astype(np.float32)
        
        # Preserve known regions from original belief
        known_mask = belief != 127
        self.pred_mean_belief[known_mask] = belief[known_mask]
        self.pred_max_belief[known_mask] = belief[known_mask]
    
    def get_observations(self, node_coords, frontiers, belief, robot_pos, goal_pos):
        """
        Get 9-dimensional node observations for PolicyNet
        
        Features (9 dims):
        - coords (2): normalized coordinates
        - utility (1): number of observable frontiers
        - indicator (1): whether node is visited
        - direction_vector (3): [dx, dy, dist] to goal
        - pred_prob (1): prediction confidence
        - pred_signal (1): whether node is in real (observed) region
        """
        n_nodes = len(node_coords)
        
        # Build nodes
        self.nodes_list = []
        utilities = []
        direction_vectors = []
        indicators = []
        pred_probs = []
        pred_signals = []
        
        # Check which nodes are in observed (real) region
        for coords in node_coords:
            x, y = int(round(coords[0])), int(round(coords[1]))
            
            # Create node
            node = Node(coords, frontiers, belief, goal_pos, param.SENSOR_RANGE)
            self.nodes_list.append(node)
            
            utilities.append(node.utility)
            direction_vectors.append(node.direction_vector)
            
            # Check if visited
            is_visited = 1 if (coords[0], coords[1]) in [(r[0], r[1]) for r in self.route_node] else 0
            indicators.append(is_visited)
            
            # Prediction probability (from pred_mean_belief)
            if self.pred_mean_belief is not None and 0 <= y < self.pred_mean_belief.shape[0] and 0 <= x < self.pred_mean_belief.shape[1]:
                pred_prob = self.pred_mean_belief[y, x]
            else:
                pred_prob = 127
            pred_probs.append(pred_prob)
            
            # Real or predicted node
            if 0 <= y < belief.shape[0] and 0 <= x < belief.shape[1]:
                is_real = 1 if belief[y, x] == 255 else 0
            else:
                is_real = 0
            pred_signals.append(is_real)
        
        # Normalize and stack
        norm_coords = node_coords / 250.0
        norm_utilities = np.array(utilities).reshape(-1, 1) / 50.0
        indicators = np.array(indicators).reshape(-1, 1)
        direction_vectors = np.array(direction_vectors)
        direction_vectors[:, 2] /= 40.0  # normalize distance
        pred_probs = np.array(pred_probs).reshape(-1, 1) / 255.0
        pred_signals = np.array(pred_signals).reshape(-1, 1)
        
        # Concatenate: [coords(2), utility(1), indicator(1), direction(3), pred_prob(1), pred_signal(1)]
        node_inputs = np.concatenate([
            norm_coords,
            norm_utilities,
            indicators,
            direction_vectors,
            pred_probs,
            pred_signals
        ], axis=1)
        
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
        
        # Find current node index
        dists = np.linalg.norm(node_coords - robot_pos, axis=1)
        current_idx = np.argmin(dists)
        current_index = torch.tensor([[current_idx]]).to(self.device)
        
        # Build edge inputs and mask
        graph_edges = list(self.graph.edges.values())
        
        # Get neighbors of current node
        if str(current_idx) in self.graph.edges:
            neighbors = [int(n) for n in self.graph.edges[str(current_idx)]]
        else:
            neighbors = []
        
        # Pad neighbors
        while len(neighbors) < param.K_SIZE:
            neighbors.append(0)
        neighbors = neighbors[:param.K_SIZE]
        
        edge_inputs = torch.tensor(neighbors).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Edge padding mask
        edge_padding_mask = torch.zeros((1, 1, param.K_SIZE), dtype=torch.int64).to(self.device)
        for idx, n in enumerate(neighbors):
            if n == 0 and idx > 0:  # padded
                edge_padding_mask[0, 0, idx] = 1
            # Mask predicted (non-real) neighbors
            if idx < len(pred_signals) and pred_signals[neighbors[idx]][0] == 0:
                edge_padding_mask[0, 0, idx] = 1
        
        # Edge mask (adjacency matrix)
        edge_mask = np.ones((n_nodes, n_nodes))
        for i in range(n_nodes):
            if str(i) in self.graph.edges:
                for j_str in self.graph.edges[str(i)]:
                    edge_mask[i, int(j_str)] = 0
        edge_mask = torch.FloatTensor(edge_mask).unsqueeze(0).to(self.device)
        
        return node_inputs, edge_inputs, current_index, None, edge_padding_mask, edge_mask
    
    def select_waypoint(self, observations, node_coords, goal_pos):
        """Use PolicyNet to select next waypoint"""
        node_inputs, edge_inputs, current_index, _, edge_padding_mask, edge_mask = observations
        
        with torch.no_grad():
            logp = self.policy_net(node_inputs, edge_inputs, current_index, 
                                   None, edge_padding_mask, edge_mask)
        
        # Get neighbor indices
        neighbors = edge_inputs[0, 0].numpy()
        neighbor_coords = node_coords[neighbors]
        
        # Goal attraction score
        distances_to_goal = np.linalg.norm(neighbor_coords - goal_pos, axis=1)
        max_dist = np.max(distances_to_goal) + 1e-6
        goal_score = (max_dist - distances_to_goal) / max_dist
        
        # RL score
        rl_prob = logp.exp().squeeze().cpu().numpy()
        rl_score = rl_prob / (np.max(rl_prob) + 1e-6)
        
        # Combined score (RL weight 0.1, goal direction weight 0.9)
        combined_score = 0.1 * rl_score + 0.9 * goal_score
        
        # Mask invalid neighbors
        mask = edge_padding_mask[0, 0].numpy()
        combined_score[mask == 1] = -np.inf
        
        best_idx = np.argmax(combined_score)
        next_node_idx = neighbors[best_idx]
        next_position = node_coords[next_node_idx]
        
        return next_position
    
    def run(self, event=None):
        """Main planning loop"""
        t_start = time.time()
        
        if self.done or self.waiting_for_goal:
            return
        
        if self.map_info is None or self.robot_position is None or self.goal_position is None:
            return
        
        # Check if goal reached
        dist_to_goal = np.linalg.norm(self.robot_position_world - self.goal_position_world)
        if dist_to_goal < param.GOAL_REACHED_THR:
            rospy.loginfo("\033[92mGoal reached!\033[0m")
            self.done = True
            self.waiting_for_goal = True
            return
        
        try:
            # Check if we need to replan (no target or reached current target)
            need_replan = False
            if self.current_target_world is None:
                need_replan = True
            else:
                dist_to_target = np.linalg.norm(self.robot_position_world - self.current_target_world)
                if dist_to_target < param.THR_TO_WAYPOINT:
                    need_replan = True
            
            if need_replan:
                # Update prediction
                self.update_predict_map()
                
                # Use predicted map for graph building
                belief_for_graph = self.pred_max_belief if self.pred_max_belief is not None else self.robot_belief
                
                # Find frontiers
                frontiers = self.find_frontiers()
                
                # Generate nodes
                self.node_coords = self.generate_node_coords(
                    belief_for_graph, self.robot_position, self.goal_position)
                
                if len(self.node_coords) < 2:
                    rospy.logwarn("Not enough nodes for planning")
                    return
                
                # Build graph
                self.build_graph(self.node_coords, belief_for_graph)
                
                # Get observations
                observations = self.get_observations(
                    self.node_coords, frontiers, self.robot_belief,
                    self.robot_position, self.goal_position)
                
                # Select waypoint
                self.current_target_pixel = self.select_waypoint(observations, self.node_coords, self.goal_position)
                self.current_target_world = self.pixel_to_world(self.current_target_pixel)
                
                # Update route
                self.route_node.append(self.robot_position.copy())
                
                # Publish visualization
                if self.publish_graph:
                    self.visualize(frontiers)
            
            # Publish raw path [robot_pos, target] for trajectory smoother
            path_msg = Path()
            path_msg.header = Header(stamp=rospy.Time.now(), frame_id="map")
            for px, py in [(self.robot_position_world[0], self.robot_position_world[1]),
                           (self.current_target_world[0], self.current_target_world[1])]:
                ps = PoseStamped()
                ps.header = path_msg.header
                ps.pose.position.x = px
                ps.pose.position.y = py
                ps.pose.position.z = 1.0
                ps.pose.orientation.w = 1.0
                path_msg.poses.append(ps)
            self.raw_path_pub.publish(path_msg)
            
            # Publish runtime
            runtime_msg = Float32()
            runtime_msg.data = time.time() - t_start
            self.runtime_pub.publish(runtime_msg)
            
            # Publish predicted map
            if self.pred_mean_belief is not None:
                self.publish_predicted_map()
            
            rospy.loginfo("Target: ({:.2f}, {:.2f}), dist_to_goal: {:.2f}m".format(
                self.current_target_world[0], self.current_target_world[1], dist_to_goal))
            
        except Exception as e:
            rospy.logerr("Planning error: {}".format(e))
            import traceback
            traceback.print_exc()
    
    def publish_predicted_map(self):
        """Publish predicted map for visualization"""
        pred = self.pred_mean_belief
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.info.width = pred.shape[1]
        msg.info.height = pred.shape[0]
        msg.info.resolution = self.map_info.cell_size
        msg.info.origin.position.x = self.map_info.map_origin_x
        msg.info.origin.position.y = self.map_info.map_origin_y
        msg.info.origin.orientation.w = 1.0
        
        # Convert to ROS format
        ros_map = np.full_like(pred, -1, dtype=np.int8)
        ros_map[pred > 200] = 0  # free
        ros_map[pred < 50] = 100  # occupied
        msg.data = ros_map.flatten().tolist()
        
        self.pred_map_pub.publish(msg)
    
    def visualize(self, frontiers):
        """Publish visualization markers"""
        stamp = rospy.Time.now()
        
        # Nodes
        if self.node_coords is not None and len(self.node_coords) > 0:
            nodes = []
            for i, coord in enumerate(self.node_coords):
                world_pos = self.pixel_to_world(coord)
                utility = self.nodes_list[i].utility if i < len(self.nodes_list) else 0
                nodes.append((world_pos[0], world_pos[1], 0.5, float(utility)))
            
            header = Header(stamp=stamp, frame_id="map")
            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1)
            ]
            pc = point_cloud2.create_cloud(header, fields, nodes)
            self.node_pub.publish(pc)
        
        # Frontiers
        if len(frontiers) > 0:
            frontier_pts = []
            for f in frontiers:
                world_pos = self.pixel_to_world(f)
                frontier_pts.append((world_pos[0], world_pos[1], 0.3))
            
            header = Header(stamp=stamp, frame_id="map")
            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            pc = point_cloud2.create_cloud(header, fields, frontier_pts)
            self.frontier_pub.publish(pc)


def main():
    try:
        planner = RLPlannerNav()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
