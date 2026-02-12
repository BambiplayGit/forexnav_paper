#!/usr/bin/env python3
import os
import rospy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
import torchvision.transforms as transforms
import sys
from collections import deque
import yaml
import csv
import datetime
import math

# [NEW] Imports for Visualization
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Pose, PoseStamped, Point
from sensor_msgs.msg import Image as RosImage
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

from networks import Generator
from tools import get_config

# ==========================================
# 1. KF Mean (Adaptive Prior Fusion) - [The Core Method]
# ==========================================
class AdaptivePriorFusion:
    def __init__(self, num_priors, device, beta=0.2, temperature=0.3, window_size=5, min_pixel_thresh=20):
        self.num_priors = num_priors
        self.device = device
        self.beta = beta
        self.temperature = temperature
        self.min_pixel_thresh = min_pixel_thresh

        # Variance vector for 16 anchors (initialized to 0)
        self.variances = torch.zeros(num_priors, device=device) 
        
        # Reference frame for internal variance update
        self.ref_belief_raw = None
        self.ref_preds = None
        self.unknown_thresh = 0.1
        
        # Smoothing buffer
        self.window_size = window_size
        self.history_buffer = deque(maxlen=window_size)

    def step(self, current_belief_raw, current_preds_list):
        """
        Main KF Step:
        1. Try to update variances based on previous frame (prediction vs reality).
        2. Calculate weights based on variances (Softmax).
        3. Fuse predictions.
        4. Temporal smoothing.
        """
        # 1. Update Variances (Process Model Update)
        if self.ref_belief_raw is not None:
            updated = self._try_update_variances(current_belief_raw)
        else:
            updated = False

        if updated or self.ref_belief_raw is None:
            self.ref_belief_raw = current_belief_raw.detach().clone()
            self.ref_preds = current_preds_list.detach().clone()

        # 2. Calculate Weights (Measurement Model)
        # Lower variance -> Higher weight
        weights = F.softmax(-self.variances / self.temperature, dim=0)

        # 3. Fuse (Weighted Sum)
        if current_preds_list.dim() == 4: # [16, 1, H, W]
            w_expanded = weights.view(-1, 1, 1, 1)
        else: # [16, H, W]
            w_expanded = weights.view(-1, 1, 1)
            
        current_fused = (current_preds_list * w_expanded).sum(dim=0)

        # 4. Temporal Smoothing
        self.history_buffer.append(current_fused.detach())
        smoothed_output = torch.stack(list(self.history_buffer)).mean(dim=0)

        return smoothed_output, weights

    def _try_update_variances(self, current_belief):
        """Internal logic: Update variance sigma^2 based on prediction error on newly discovered areas."""
        belief_flat = current_belief.view(-1)
        ref_belief_flat = self.ref_belief_raw.view(-1)
        
        # Mask: Where was unknown BEFORE, but is known NOW?
        mask_ref_unknown = torch.abs(ref_belief_flat) < self.unknown_thresh
        mask_curr_known = torch.abs(belief_flat) >= self.unknown_thresh
        mask_eval = mask_ref_unknown & mask_curr_known 

        pixel_count = mask_eval.sum()
        if pixel_count < self.min_pixel_thresh:
            return False 

        target = belief_flat[mask_eval]

        # Update variance for each anchor
        for i in range(self.num_priors):
            pred_ref_flat = self.ref_preds[i].view(-1)
            if pred_ref_flat.shape[0] != mask_eval.shape[0]: continue

            pred_val = pred_ref_flat[mask_eval]
            raw_mse = torch.mean((pred_val - target) ** 2)
            scaled_error = raw_mse * 10.0 

            # Recursive Filter Update
            self.variances[i] = (1 - self.beta) * self.variances[i] + self.beta * scaled_error

        return True

# ==========================================
# 2. Benchmark Evaluator (Simplified: KF Only)
# ==========================================
class BenchmarkEvaluator:
    def __init__(self, anchors, device, config_dict):
        self.device = device
        self.num_anchors = len(anchors)
        
        self.min_pixel_thresh = config_dict.get('min_pixel_thresh', 50)
        self.unknown_thresh = config_dict.get('unknown_thresh', 0.1)
        
        self.benchmark_delay = 4.0  
        
        # [ONLY KF Logic]
        self.kf_model = AdaptivePriorFusion(self.num_anchors, device)
        
        self.history_queue = deque()
        
        # Only track KF history
        self.mse_history = {
            'ours_kf': []
        }
            
        self.viz_cache = {
            "all_preds_realtime": None, 
            "ours_kf_curr": 0.0, 
            "ours_kf_cum": 0.0,     
            "info_text": "Waiting for Goal..."
        }

        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_dir = os.path.join(script_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.csv_path = os.path.join(csv_dir, f"kf_mse_log_{timestamp}.csv")
        try:
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['Frame', 'MSE_KF', 'Avg_MSE_KF'])
            rospy.loginfo(f"Logging KF data to: {self.csv_path}")
        except Exception as e:
            rospy.logerr(f"Failed to open CSV log file: {e}")
            self.csv_file = None

    def reset_history(self):
        self.history_queue.clear()
        self.mse_history['ours_kf'] = []
        self.viz_cache["info_text"] = "Goal Received. Buffer filling..."
        if "delayed_snapshot" in self.viz_cache:
            del self.viz_cache["delayed_snapshot"]
        rospy.loginfo("Benchmark Reset (KF Only).")

    def step(self, current_belief_raw, all_preds_goal, current_time_sec, record=False):
        if current_belief_raw.dim() == 4:
            current_belief_raw = current_belief_raw.squeeze(0)

        # Run KF
        kf_output, weights = self.kf_model.step(current_belief_raw, all_preds_goal)
        
        self.viz_cache["all_preds_realtime"] = all_preds_goal.detach().cpu()
        
        if record:
            snapshot = {
                'time': current_time_sec,
                'belief': current_belief_raw.detach().clone(), 
                'ours_kf': kf_output.detach().clone(),     
                'evaluated': False
            }
            self.history_queue.append(snapshot)
            self.viz_cache["info_text"] = "Recording..."

            self._process_history(current_belief_raw, current_time_sec)
            self._update_stats()
        else:
            self.viz_cache["info_text"] = "Idle / Reached Goal"
        
        return kf_output

    def _process_history(self, current_belief, current_time):
        if not self.history_queue: return
        mask_curr_known = torch.abs(current_belief) >= self.unknown_thresh
        
        while self.history_queue:
            head = self.history_queue[0]
            time_diff = current_time - head['time']
            
            if time_diff >= self.benchmark_delay:
                self.viz_cache["delayed_snapshot"] = head

                if not head['evaluated']:
                    self._do_hindsight_evaluation(head, current_belief, mask_curr_known)
                    head['evaluated'] = True
                    self.history_queue.popleft() 
                else:
                    self.history_queue.popleft()
            else:
                break

    def _do_hindsight_evaluation(self, snapshot, current_belief, mask_curr_known):
        mask_ref_unknown = torch.abs(snapshot['belief']) < self.unknown_thresh
        mask_score = mask_ref_unknown & mask_curr_known
        
        if mask_score.sum() < self.min_pixel_thresh: return False 

        target_score = current_belief[mask_score]
        
        def calc_mse(pred_vals, target_vals, fp_weight=2.5):
            diff = pred_vals - target_vals
            mask_fp = (target_vals < -0.4) & (diff > 0.8)
            weighted_diff = diff.clone()
            weighted_diff[mask_fp] *= fp_weight 
            return torch.mean(weighted_diff ** 2).item() * 10.0

        # Only Calc KF MSE
        mse_kf = calc_mse(snapshot['ours_kf'][mask_score], target_score)     
        
        self.mse_history['ours_kf'].append(mse_kf)
        self.viz_cache["ours_kf_curr"] = mse_kf
        if self.csv_file:
            frame_idx = len(self.mse_history['ours_kf'])
            avg_kf = sum(self.mse_history['ours_kf']) / len(self.mse_history['ours_kf'])
            self.csv_writer.writerow([frame_idx, f"{mse_kf:.4f}", f"{avg_kf:.4f}"])
            self.csv_file.flush()
        return True

    def _update_stats(self):
        lst = self.mse_history['ours_kf']
        self.viz_cache["ours_kf_cum"] = sum(lst)/len(lst) if lst else 0.0

    def __del__(self):
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()

# ==========================================
# 3. Evaluator (Standard Helper)
# ==========================================
class Evaluator:
    def __init__(self, config, netG, use_cuda, nsample=1):
        self.config = config
        self.use_cuda = use_cuda
        self.nsample = nsample
        self.netG = netG
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.netG.to(self.device)

    @torch.no_grad()
    def eval_batch(self, x, mask, anchors_tensor, img_raw_size):
        self.netG.eval()
        num_anchors = anchors_tensor.size(0)
        x_expanded = x.expand(num_anchors, -1, -1, -1)
        mask_expanded = mask.expand(num_anchors, -1, -1, -1)
        x_out = self.netG(x_expanded, mask_expanded, anchors_tensor)
        inpainted_batch = x_out * mask_expanded + x_expanded * (1. - mask_expanded)
        
        width, height = x.size(2), x.size(3)
        crop = img_raw_size[0] < width and img_raw_size[1] < height
        
        if crop:
            i_left = (width - img_raw_size[0]) // 2
            i_top = (height - img_raw_size[1]) // 2 
            i_right = i_left + img_raw_size[0]
            i_bottom = i_top + img_raw_size[1]
            inpainted_batch = inpainted_batch[:, :, i_left:i_right, i_top:i_bottom]
        else:
            inpainted_batch = F.interpolate(inpainted_batch, size=(img_raw_size[1], img_raw_size[0]), mode='bilinear', align_corners=False)
        return inpainted_batch 

    @staticmethod
    def post_process_probability(inpaint_tensor, x_raw_tensor):
        inpaint_np = inpaint_tensor.squeeze().cpu().numpy()
        x_np = x_raw_tensor.squeeze().cpu().numpy()
        if x_np.shape != inpaint_np.shape:
             x_np = cv2.resize(x_np, (inpaint_np.shape[1], inpaint_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        is_known_mask = np.abs(x_np) > 0.1 
        final_img = np.where(is_known_mask, x_np, inpaint_np)
        return final_img
 
def preprocess_robot_belief_input(belief_np, input_shape, device):
    width_in, height_in = input_shape[0], input_shape[1]
    width_map, height_map = belief_np.shape
    pad = width_map < width_in and height_map < height_in
    if pad:
        pad_left = (width_in - width_map) // 2
        pad_top = (height_in - height_map) // 2
        pad_right = width_in - width_map - pad_left
        pad_bottom = height_in - height_map - pad_top
        belief = np.pad(belief_np, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
    else:
        belief = belief_np
    mask = belief.copy()
    mask[mask != 127] = 0
    mask[mask == 127] = 255 
    x_belief = Image.fromarray(belief.astype(np.uint8)).convert('L')
    mask = Image.fromarray(mask.astype(np.uint8)).convert('1')
    x_raw = Image.fromarray(belief_np.astype(np.uint8)).convert('L')
    if not pad:
        x_belief = transforms.Resize((width_in, height_in))(x_belief)
        mask = transforms.Resize((width_in, height_in))(mask)
    x_belief = transforms.ToTensor()(x_belief).unsqueeze(0).to(device).mul(2).add_(-1)
    mask = transforms.ToTensor()(mask).unsqueeze(0).to(device)
    x_raw = transforms.ToTensor()(x_raw).unsqueeze(0).to(device).mul(2).add_(-1)
    return x_belief, mask, x_raw

# ==========================================
# 4. Inpaint Node
# ==========================================
class InpaintNode:
    def __init__(self):
        rospy.init_node("inpaint_kf_only_node")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(script_dir, "predict_config.yaml") 
        if not os.path.exists(yaml_path):
            node_cfg = {}
        else:
            with open(yaml_path, 'r') as f:
                full_config = yaml.safe_load(f)
                node_cfg = full_config.get('inpaint_node', {})

        topics = node_cfg.get('topics', {})
        self.topic_map_in = topics.get("input_map", "/local_sensing/occupancy_grid")
        self.topic_goal_in = topics.get("input_goal", "/move_base_simple/goal")
        self.topic_odom_in = topics.get("input_odom", "/state_ukf/odom")
        
        self.topic_img_out = topics.get("output_image", "/inpainted/image")
        self.topic_dash_out = topics.get("output_dashboard", "/inpainted/dashboard")
        self.topic_map_out = topics.get("output_map", "/inpainted/map")
        
        # [NEW] Marker Topic
        self.topic_marker_out = topics.get("output_markers", "/inpainted/markers_3d")

        model_cfg = node_cfg.get('model', {})
        run_path = model_cfg.get("checkpoint_path", "../checkpoints/wgan_inpainting")
        
        self.fusion_config = {
            'min_pixel_thresh': 50,
            'unknown_thresh': 0.15,
        }

        # Net Setup
        config_path = os.path.join(run_path, "config.yaml")
        try:
            checkpoint_file = next((f for f in os.listdir(run_path) if f.startswith('gen') and f.endswith('.pt')), None)
            checkpoint_path = os.path.join(run_path, checkpoint_file)
            config = get_config(config_path)
        except Exception:
            rospy.logerr(f"Checkpoints not found in {run_path}")
            sys.exit(1)

        if config['cuda'] and torch.cuda.is_available():
            self.device = torch.device('cuda')
            cudnn.benchmark = True
            use_cuda_flag = True
        else:
            self.device = torch.device('cpu')
            use_cuda_flag = False

        self.netG = Generator(config['netG'], use_cuda_flag)
        self.netG.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.netG.to(self.device)
        self.netG.eval()

        self.evaluator = Evaluator(config, self.netG, use_cuda_flag)
        self.config = config
        
        # 16 Anchors
        self.anchors = [
            [0.333, 0.333, 0.333], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6],
            [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
            [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
            [0.4, 0.4, 0.2], [0.4, 0.2, 0.4], [0.2, 0.4, 0.4]        
        ]
        self.anchors_tensor = torch.tensor(self.anchors).float().to(self.device)

        # KF Evaluator (Only KF)
        self.bench_evaluator = BenchmarkEvaluator(self.anchors, self.device, self.fusion_config)

        # State Control
        self.is_navigating = False
        self.current_goal_pose = None
        self.goal_tolerance = 5.0 

        self.pub = rospy.Publisher(self.topic_img_out, RosImage, queue_size=1)
        self.dashboard_pub = rospy.Publisher(self.topic_dash_out, RosImage, queue_size=1)
        self.map_pub = rospy.Publisher(self.topic_map_out, OccupancyGrid, queue_size=1)  
        
        # [NEW] Publisher for Voxel Markers
        self.marker_pub = rospy.Publisher(self.topic_marker_out, Marker, queue_size=1)
        
        self.map_info_ready = False
        self.map_resolution = 0.05
        self.map_origin_pose = Pose()
        self.map_sub = rospy.Subscriber(self.topic_map_in, OccupancyGrid, self.map_callback, queue_size=1, buff_size=2**24)
        
        self.goal_received = False
        self.goal_sub = rospy.Subscriber(self.topic_goal_in, PoseStamped, self.goal_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.topic_odom_in, Odometry, self.odom_callback, queue_size=1)

        rospy.loginfo(f"InpaintNode (KF Only) Init Complete.")

    def imgmsg_to_cv2(self, img_msg):
        dtype = np.dtype("uint8")
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        shape = (img_msg.height, img_msg.width) if img_msg.encoding == 'mono8' else (img_msg.height, img_msg.width, 3)
        image_opencv = np.ndarray(shape=shape, dtype=dtype, buffer=img_msg.data)
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            image_opencv = image_opencv.byteswap().newbyteorder()
        return image_opencv.copy()

    def cv2_to_imgmsg(self, cv_image, encoding='mono8'):
        msg = RosImage()
        msg.height, msg.width = cv_image.shape[0], cv_image.shape[1]
        msg.encoding = encoding
        msg.is_bigendian = (sys.byteorder == 'big')
        msg.step = msg.width * (3 if encoding == 'bgr8' else 1)
        msg.data = cv_image.tobytes()
        return msg

    @staticmethod
    def occupancy_grid_to_cv2(grid_msg):
        """OccupancyGrid to grayscale: 100->0, 0->255, -1->127."""
        w, h = grid_msg.info.width, grid_msg.info.height
        arr = np.array(grid_msg.data, dtype=np.int8).reshape((h, w))
        out = np.where(arr == 100, 0, np.where(arr == 0, 255, 127)).astype(np.uint8)
        return out

    def map_callback(self, map_msg):
        self.map_frame = map_msg.header.frame_id
        self.map_resolution = map_msg.info.resolution
        self.map_origin_pose = map_msg.info.origin
        self.map_info_ready = True
        try:
            cv_img_clean = self.occupancy_grid_to_cv2(map_msg)
        except Exception:
            return
        now = rospy.get_time()
        if not hasattr(self, '_last_t'):
            self._last_t = 0
        if now - self._last_t < 0.5:
            return
        self._last_t = now
        self._run_pipeline(cv_img_clean, map_msg.header)

    def goal_callback(self, msg):
            if not self.map_info_ready: return
            self.current_goal_pose = msg.pose
            
            if not self.goal_received:
                self.bench_evaluator.reset_history()
                self.goal_received = True
                rospy.loginfo("First Goal received. Starting KF.")
            else:
                rospy.loginfo("New Goal updated.")
            
            self.is_navigating = True

    def odom_callback(self, msg):
        if not self.is_navigating or not self.current_goal_pose: return
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        gx = self.current_goal_pose.position.x
        gy = self.current_goal_pose.position.y
        dist = math.sqrt((px - gx)**2 + (py - gy)**2)
        if dist < self.goal_tolerance:
            self.is_navigating = False
            rospy.loginfo(f"Goal reached. Stopping MSE recording.")

    def inject_goal_region_to_input(self, cv_img):
            if not self.goal_received or not self.map_info_ready: return
            res = self.map_resolution
            H, W = cv_img.shape[:2]
            cx, cy = W // 2, H // 2
            gx = int(self.current_goal_pose.position.x / res)
            gy = int(self.current_goal_pose.position.y / res)
            radius = int(1.0 / res) 
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    row, col = cy - (gx + dx), cx - (gy + dy)
                    if 0 <= row < H and 0 <= col < W: 
                        pixel_val = cv_img[row, col]
                        if abs(int(pixel_val) - 127) < 5: 
                            cv_img[row, col] = 255

    def image_to_occupancy_grid(self, cv_image, img_header):
        grid = OccupancyGrid()
        grid.header.stamp = img_header.stamp
        grid.header.frame_id = self.map_frame or img_header.frame_id
        # Input is from OccupancyGrid, pipeline output is already in grid frame; no rotate/flip
        h, w = cv_image.shape[:2]
        grid.info.width, grid.info.height = w, h
        grid.info.resolution = self.map_resolution
        grid.info.origin = self.map_origin_pose
        prob = (255 - cv_image.astype(np.float32)) / 255.0 * 100.0
        grid.data = np.clip(prob, 0, 100).astype(np.int8).flatten().tolist()
        return grid

    # ==================================================
    # [UPDATED] Publish Voxel Markers (Unknown Areas Only)
    # ==================================================
    def publish_voxel_markers(self, cv_image, raw_image, img_header):
        """
        Convert the inpainted image to a CubeList Marker.
        Constraint: Only visualize predicted obstacles in areas that were originally UNKNOWN.
        """
        if not self.map_info_ready:
            return

        # 1. Align Images to Map Frame (Result & Raw)
        # Process Final Result
        rotated = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        aligned = np.flipud(rotated)
        
        # Process Raw Input (to identify unknown regions)
        raw_rotated = cv2.rotate(raw_image, cv2.ROTATE_90_CLOCKWISE)
        raw_aligned = np.flipud(raw_rotated)
        
        # 2. Downsample to target resolution (1m x 1m)
        target_xy = 1.0  
        target_z  = 3.0  
        
        scale_factor = target_xy / self.map_resolution
        if scale_factor < 1.0: 
            scale_factor = 1.0
        
        h, w = aligned.shape
        new_w = int(w / scale_factor)
        new_h = int(h / scale_factor)
        
        if new_w <= 0 or new_h <= 0:
            return

        # Downsample Result: Use INTER_AREA to average occupancy probability
        downsampled = cv2.resize(aligned, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Downsample Raw: Use INTER_NEAREST to preserve the "127" (Unknown) category tag
        raw_downsampled = cv2.resize(raw_aligned, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 3. Setup Marker
        marker = Marker()
        marker.header.frame_id = self.map_frame or img_header.frame_id
        marker.header.stamp = img_header.stamp
        marker.ns = "inpainted_voxels_unknown_only"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = target_xy
        marker.scale.y = target_xy
        marker.scale.z = target_z

        # 4. Filter Logic (The Core Change)
        # Condition A: The result says it is Occupied (Darker than 245)
        is_occupied = downsampled < 245
        
        # Condition B: The Raw Input says it was Unknown (Value approx 127)
        # We allow a small margin (+/- 5) to account for noise in the raw image
        is_unknown = np.abs(raw_downsampled.astype(np.int16) - 127) < 5

        # Final Mask: Must be Occupied AND originally Unknown
        final_mask = is_occupied & is_unknown
        
        rows, cols = np.where(final_mask)
        
        if len(rows) == 0:
            self.marker_pub.publish(marker)
            return

        pixel_values = downsampled[rows, cols]

        # 5. Calculate Coordinates
        ox = self.map_origin_pose.position.x
        oy = self.map_origin_pose.position.y
        
        x_coords = ox + (cols * target_xy) + (target_xy / 2.0)
        y_coords = oy + (rows * target_xy) + (target_xy / 2.0)
        z_coord = self.map_origin_pose.position.z + (target_z / 2.0)

        marker.points = [Point(x=x, y=y, z=z_coord) for x, y in zip(x_coords, y_coords)]

        # 6. Calculate Colors
        base_alpha = 1.0 - (pixel_values.astype(np.float32) / 255.0)
        alphas = base_alpha ** 3
        alphas = np.clip(alphas, 0, 0.4)
        
        # Purple Color: R=0.6, G=0.0, B=0.8
        marker.colors = [ColorRGBA(r=0.6, g=0.0, b=0.8, a=float(a)) for a in alphas]

        self.marker_pub.publish(marker)

    # ==================================================
    # Dashboard: KF Only
    # ==================================================
    def draw_mse_chart(self, mse_history, width, height):
        chart = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        def get_smooth(lst, win=5):
            if not lst: return []
            if len(lst) < win: return lst
            ret = np.cumsum(lst, dtype=float)
            ret[win:] = ret[win:] - ret[:-win]
            return (ret[win - 1:] / win).tolist()

        ours_kf_data = get_smooth(mse_history['ours_kf'])
        if not ours_kf_data: return chart

        # Scaling
        global_min, global_max = min(ours_kf_data), max(ours_kf_data)
        margin = (global_max - global_min) * 0.15 if (global_max - global_min) > 1e-3 else 1.0
        min_val, max_val = max(0, global_min - margin), global_max + margin
        N = len(ours_kf_data)

        pad_l, pad_r, pad_t, pad_b = 45, 15, 20, 30
        plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
        step_x = plot_w / (N - 1) if N > 1 else 0
        
        def val_to_y(v):
            denom = (max_val - min_val) if (max_val - min_val) > 1e-5 else 1.0
            ratio = (v - min_val) / denom
            return int((height - pad_b) - ratio * plot_h)
            
        def draw_line(data, color, thick=1):
            if len(data) < 2: return
            pts = []
            for i, v in enumerate(data):
                px = int(pad_l + i * step_x)
                py = np.clip(val_to_y(v), pad_t, height - pad_b)
                pts.append((px, py))
            cv2.polylines(chart, [np.array(pts)], False, color, thick, lineType=cv2.LINE_AA)

        # Axes
        cv2.line(chart, (pad_l, pad_t), (pad_l, height - pad_b), (0,0,0), 2)
        cv2.line(chart, (pad_l, height - pad_b), (width - pad_r, height - pad_b), (0,0,0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Y-Ticks
        for i in range(5):
            ratio = i / 4.0
            val = min_val + ratio * (max_val - min_val)
            y_pos = int((height - pad_b) - ratio * plot_h)
            cv2.line(chart, (pad_l - 4, y_pos), (pad_l, y_pos), (0,0,0), 1)
            cv2.putText(chart, f"{val:.1f}", (2, y_pos + 4), font, 0.4, (0,0,0), 1)
            if i > 0: cv2.line(chart, (pad_l, y_pos), (width - pad_r, y_pos), (220,220,220), 1)

        # KF Line (Red/Blue)
        draw_line(ours_kf_data, (200, 0, 0), 2)     
        return chart

    def create_dashboard_image(self, input_cv_img, evaluator):
        IMG_SZ = 200 # Slightly larger since we have fewer images
        cache = evaluator.viz_cache
        preds_tensor = cache.get("all_preds_realtime", None)
        
        if preds_tensor is None:
            return np.zeros((600, 400, 3), dtype=np.uint8)

        # Get KF Output
        snap_last = evaluator.history_queue[-1]['ours_kf'] if evaluator.history_queue else None
        if snap_last is not None:
            kf_tensor = snap_last
        else:
             # Fallback if queue empty (using current weights)
            device = evaluator.kf_model.variances.device
            kf_vars = evaluator.kf_model.variances
            kf_temp = evaluator.kf_model.temperature
            kf_weights = F.softmax(-kf_vars / kf_temp, dim=0).to('cpu')
            
            if preds_tensor.dim() == 4: kw = kf_weights.view(-1, 1, 1, 1)
            else: kw = kf_weights.view(-1, 1, 1)
            kf_tensor = (preds_tensor * kw).sum(dim=0)

        def process_img(img_src, label, sz=(IMG_SZ, IMG_SZ), color=(128,128,128)):
            if img_src is None: return np.zeros((sz[1], sz[0], 3), dtype=np.uint8)
            if isinstance(img_src, torch.Tensor): arr = img_src.squeeze().detach().cpu().numpy()
            else: arr = img_src
            
            if arr.dtype != np.uint8:
                arr = np.clip((arr + 1) / 2 * 255, 0, 255).astype(np.uint8)
            
            resized = cv2.resize(arr, sz, interpolation=cv2.INTER_NEAREST)
            bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR) if len(resized.shape) == 2 else resized
            
            header = np.zeros((24, sz[0], 3), dtype=np.uint8)
            cv2.rectangle(header, (0, 0), (sz[0], 24), color, -1)
            cv2.putText(header, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
            return np.vstack([header, bgr])

        # Row 1: Input & KF
        img_input = process_img(input_cv_img, "Input", color=(150,150,150))
        img_kf    = process_img(kf_tensor, "Ours(KF) Fused", color=(255, 0, 0))            
        row1 = np.hstack([img_input, img_kf])

        # Row 2: Delayed Check
        delayed_snap = cache.get("delayed_snapshot", None)
        if delayed_snap is None:
            blank = np.zeros((IMG_SZ, IMG_SZ, 3), dtype=np.uint8)
            row2 = np.hstack([process_img(blank, "Filling Buffer...", color=(50,50,50)), 
                              process_img(blank, "Filling Buffer...", color=(50,50,50))])
        else:
            prev_raw_t = delayed_snap['belief']
            prev_raw_np = (prev_raw_t.squeeze().detach().cpu().numpy() + 1) / 2 * 255
            prev_raw_np = np.clip(prev_raw_np, 0, 255).astype(np.uint8)
            
            # Mask logic
            curr_h, curr_w = input_cv_img.shape[:2]
            prev_t_cpu = delayed_snap['belief'].squeeze().detach().cpu()
            mask_prev_unknown = torch.abs(prev_t_cpu) < 0.1 
            mask_prev_unknown_np = cv2.resize(mask_prev_unknown.numpy().astype(np.uint8), (curr_w, curr_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            mask_curr_known = np.abs(input_cv_img.astype(float) - 127) > 5
            mask_diff = mask_prev_unknown_np & mask_curr_known

            def get_overlaid_pred(pred_t, base_img, mask_d, lbl, color):
                if isinstance(pred_t, torch.Tensor): arr = pred_t.squeeze().detach().cpu().numpy()
                else: arr = pred_t
                if arr.dtype != np.uint8: arr = np.clip((arr + 1) / 2 * 255, 0, 255).astype(np.uint8)
                h, w = arr.shape
                base_resized = cv2.resize(base_img, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_resized = cv2.resize(mask_d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                final_img = base_resized.copy()
                roi = (mask_resized > 0)
                final_img[roi] = arr[roi]
                return process_img(final_img, lbl, color=color)

            img_prev_raw = process_img(prev_raw_np, "Prev Raw (Delayed)", color=(100,100,100))
            snap_kf = delayed_snap['ours_kf']
            img_kf_new = get_overlaid_pred(snap_kf, prev_raw_np, mask_diff, "KF Overlay Check", (255, 0, 0))
            row2 = np.hstack([img_prev_raw, img_kf_new])

        # Table & Chart
        W_FULL = row1.shape[1]
        table_img = np.zeros((50, W_FULL, 3), dtype=np.uint8)
        kf_curr = cache.get("ours_kf_curr", 0.0)
        kf_cum  = cache.get("ours_kf_cum", 0.0)
        info = cache.get("info_text", "")
        
        cv2.putText(table_img, f"Status: {info}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(table_img, f"KF MSE (Curr/Avg): {kf_curr:.2f} / {kf_cum:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        chart_img = self.draw_mse_chart(evaluator.mse_history, W_FULL, 150)
        
        return np.vstack([row1, row2, table_img, chart_img])

    def _run_pipeline(self, cv_img_clean, header):
        cv_img_goal = cv_img_clean.copy()
        self.inject_goal_region_to_input(cv_img_goal)
        x_goal, mask_goal, _ = preprocess_robot_belief_input(cv_img_goal, self.config['image_shape'], self.device)
        _, _, x_raw_clean = preprocess_robot_belief_input(cv_img_clean, self.config['image_shape'], self.device)
        all_preds_goal = self.evaluator.eval_batch(x_goal, mask_goal, self.anchors_tensor, cv_img_goal.shape[::-1])
        current_time = header.stamp.to_sec()
        nav_map_out = self.bench_evaluator.step(
            x_raw_clean, all_preds_goal, current_time, record=self.is_navigating
        )
        try:
            dashboard_img = self.create_dashboard_image(cv_img_clean, self.bench_evaluator)
            dash_msg = self.cv2_to_imgmsg(dashboard_img, encoding='bgr8')
            dash_msg.header = header
            self.dashboard_pub.publish(dash_msg)
        except Exception as e:
            rospy.logerr(f"Dash Error: {e}")
        processed_img_np = self.evaluator.post_process_probability(nav_map_out, x_raw_clean)
        np_result = (np.clip((processed_img_np + 1) / 2, 0, 1) * 255).astype(np.uint8)
        out_msg = self.cv2_to_imgmsg(np_result, encoding='mono8')
        out_msg.header = header
        self.pub.publish(out_msg)
        self.map_pub.publish(self.image_to_occupancy_grid(np_result, header))
        self.publish_voxel_markers(np_result, cv_img_clean, header)
if __name__ == "__main__":
    node = InpaintNode()
    rospy.spin()