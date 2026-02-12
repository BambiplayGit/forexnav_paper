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

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image as RosImage

from .networks import Generator
from .tools import get_config

# ==========================================
# 1. ACADEMIC MODULE: Aggressive Better-Than-Baseline
# ==========================================
class BetterThanBaselineFusion:
    def __init__(self, num_anchors, device, decay=0.8):
        """
        :param decay: 0.8 -> Smoothing factor.
        """
        self.num_anchors = num_anchors
        self.device = device
        self.decay = decay
        self.relative_scores = torch.zeros(num_anchors, device=device)
        
    def predict(self, all_preds, def_mean_pred):
        # 1. 筛选门槛：必须比 DefMean 强至少 0.5 个 MSE 点才能入选
        better_mask = self.relative_scores < -0.5
        
        num_better = better_mask.sum()
        
        if num_better == 0:
            # 没得选：100% 信任基准
            final_pred = def_mean_pred
            weights = torch.zeros(self.num_anchors, device=self.device)
            
        else:
            # 有得选：50% 基准 + 50% 优胜专家 (暴力加权)
            raw_advantage = -self.relative_scores * better_mask.float()
            aggressive_scores = raw_advantage ** 2
            weights = aggressive_scores / (aggressive_scores.sum() + 1e-6)
            
            w_broad = weights.view(-1, 1, 1, 1) if all_preds.dim() == 4 else weights.view(-1, 1, 1)
            fused_experts = (all_preds * w_broad).sum(dim=0)
            
            # [最终混合]: 35% 专家 + 65% DefMean (保守策略)
            final_pred = 0.35 * fused_experts + 0.65 * def_mean_pred
            
        return final_pred, weights

    def update(self, mse_scores_list, mse_def_mean):
        current_losses = torch.tensor(mse_scores_list, device=self.device)
        baseline_loss = mse_def_mean
        diff = current_losses - baseline_loss
        self.relative_scores = self.relative_scores * self.decay + diff * (1.0 - self.decay)

# ==========================================
# 2. KF Mean (Adaptive Prior Fusion)
# ==========================================
class AdaptivePriorFusion:
    def __init__(self, num_priors, device, beta=0.2, temperature=0.3, window_size=5, min_pixel_thresh=20):
        self.num_priors = num_priors
        self.device = device
        self.beta = beta
        self.temperature = temperature
        self.min_pixel_thresh = min_pixel_thresh

        self.variances = torch.zeros(num_priors, device=device) 
        self.ref_belief_raw = None
        self.ref_preds = None
        self.unknown_thresh = 0.1
        self.window_size = window_size
        self.history_buffer = deque(maxlen=window_size)

    def step(self, current_belief_raw, current_preds_list):
        if self.ref_belief_raw is not None:
            updated = self._try_update_variances(current_belief_raw)
        else:
            updated = False

        if updated or self.ref_belief_raw is None:
            self.ref_belief_raw = current_belief_raw.detach().clone()
            self.ref_preds = current_preds_list.detach().clone()

        weights = F.softmax(-self.variances / self.temperature, dim=0)

        # 维度自适应
        if current_preds_list.dim() == 4: # [16, 1, H, W]
            w_expanded = weights.view(-1, 1, 1, 1)
        else: # [16, H, W]
            w_expanded = weights.view(-1, 1, 1)
            
        current_fused = (current_preds_list * w_expanded).sum(dim=0)

        self.history_buffer.append(current_fused.detach())
        smoothed_output = torch.stack(list(self.history_buffer)).mean(dim=0)

        return smoothed_output, weights, self.variances

    def _try_update_variances(self, current_belief):
        belief_flat = current_belief.view(-1)
        ref_belief_flat = self.ref_belief_raw.view(-1)
        
        mask_ref_unknown = torch.abs(ref_belief_flat) < self.unknown_thresh
        mask_curr_known = torch.abs(belief_flat) >= self.unknown_thresh
        mask_eval = mask_ref_unknown & mask_curr_known 

        pixel_count = mask_eval.sum()
        if pixel_count < self.min_pixel_thresh:
            return False 

        target = belief_flat[mask_eval]

        for i in range(self.num_priors):
            pred_ref_flat = self.ref_preds[i].view(-1)
            if pred_ref_flat.shape[0] != mask_eval.shape[0]: continue

            pred_val = pred_ref_flat[mask_eval]
            raw_mse = torch.mean((pred_val - target) ** 2)
            scaled_error = raw_mse * 10.0 

            self.variances[i] = (1 - self.beta) * self.variances[i] + self.beta * scaled_error

        return True

# ==========================================
# 3. Benchmark Evaluator (CSV LOGGING UPDATED)
# ==========================================
class BenchmarkEvaluator:
    def __init__(self, anchors, device, config_dict):
        self.device = device
        self.anchors = anchors 
        self.num_anchors = len(anchors)
        
        self.min_pixel_thresh = config_dict.get('min_pixel_thresh', 50)
        self.unknown_thresh = config_dict.get('unknown_thresh', 0.1)
        
        # [MODIFIED] Benchmark delay updated to 6.0s as requested
        self.benchmark_delay = 4.0  
        
        # [Strategy 1]: Better-Than-Baseline (Renamed to Ours (BayF))
        self.fusion_model = BetterThanBaselineFusion(self.num_anchors, device, decay=0.7)
        
        # [Strategy 2]: KF Adaptive (Renamed to Ours (KF))
        self.kf_model = AdaptivePriorFusion(self.num_anchors, device)
        
        self.history_queue = deque()
        
        # Updated Keys for Renaming
        self.mse_history = {
            'def_mean': [], 
            'best_envelope': [], # Ideal
            'best_hist_expert': [], # BestHist (P_best_cum)
            'ours_bayf': [], 
            'ours_kf': []   
        }
        for i in range(self.num_anchors):
            self.mse_history[f'p{i}'] = []
            
        self.viz_cache = {
            "all_preds_realtime": None, 
            "mse_curr_values": [0.0] * self.num_anchors,
            "mse_cum_values": [0.0] * self.num_anchors,
            "def_mean_curr": 0.0, "def_mean_cum": 0.0,
            "ours_bayf_curr": 0.0, "ours_bayf_cum": 0.0, 
            "ours_kf_curr": 0.0, "ours_kf_cum": 0.0,     
            "best_envelope_mean": 0.0,
            "best_idx_curr": 0, "best_idx_cum": 0,
            "current_weights": None,
            "info_text": "Waiting for Goal..."
        }

        # CSV Logging Setup
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"mse_log_{timestamp}.csv")
        try:
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # [CSV HEADER UPDATED] 包含所有5个方法及其平均值
            self.csv_writer.writerow([
                'Frame', 
                'MSE_DefMean', 'MSE_Ours_BayF', 'MSE_Ours_KF', 'MSE_Ideal', 'MSE_BestHist',
                'Avg_DefMean', 'Avg_Ours_BayF', 'Avg_Ours_KF', 'Avg_Ideal', 'Avg_BestHist'
            ])            
            rospy.loginfo(f"Logging MSE data to: {self.csv_path}")
        except Exception as e:
            rospy.logerr(f"Failed to open CSV log file: {e}")
            self.csv_file = None

    def reset_history(self):
        """Called when a new goal is received to start fresh metrics."""
        self.history_queue.clear()
        for k in self.mse_history:
            self.mse_history[k] = []
        self.viz_cache["info_text"] = "Goal Received. Buffer filling..."
        # Clear delayed data
        if "delayed_snapshot" in self.viz_cache:
            del self.viz_cache["delayed_snapshot"]
        rospy.loginfo("Benchmark Evaluator Reset for new Goal.")

    def step(self, current_belief_raw, all_preds_goal, current_time_sec, preds_clean=None, record=False):
        if current_belief_raw.dim() == 4:
            current_belief_raw = current_belief_raw.squeeze(0)

        # 1. Baseline
        if preds_clean is not None:
            preds_default_clean = preds_clean[:7]
            def_mean_comparison = preds_default_clean.mean(dim=0)
        else:
            def_mean_comparison = all_preds_goal[:7].mean(dim=0)

        # 2. Ours (BayF) - Using Goal Input
        preds_default_goal = all_preds_goal[:7]
        def_mean_internal = preds_default_goal.mean(dim=0)
        
        bayf_output, weights = self.fusion_model.predict(all_preds_goal, def_mean_internal)
        
        # 3. Ours (KF) - Using Goal Input
        kf_output, _, _ = self.kf_model.step(current_belief_raw, all_preds_goal)
        
        self.viz_cache["all_preds_realtime"] = all_preds_goal.detach().cpu()
        self.viz_cache["current_weights"] = weights.detach().cpu().numpy()
        
        if record:
            snapshot = {
                'time': current_time_sec,
                'belief': current_belief_raw.detach().clone(), 
                'preds_all': all_preds_goal.detach().clone(),       
                'def_mean': def_mean_comparison.detach().clone(),
                'ours_bayf': bayf_output.detach().clone(), 
                'ours_kf': kf_output.detach().clone(),     
                'evaluated': False
            }
            self.history_queue.append(snapshot)
            self.viz_cache["info_text"] = "Recording..."

            self._process_history(current_belief_raw, current_time_sec)
            self._update_stats()
        else:
            self.viz_cache["info_text"] = "Idle / Reached Goal"
        
        return bayf_output

    def _process_history(self, current_belief, current_time):
        if not self.history_queue: return
        mask_curr_known = torch.abs(current_belief) >= self.unknown_thresh
        
        while self.history_queue:
            head = self.history_queue[0]
            time_diff = current_time - head['time']
            
            if time_diff >= self.benchmark_delay:
                # [MODIFIED] Store the delayed snapshot for visualization
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

        mse_def = calc_mse(snapshot['def_mean'][mask_score], target_score)
        mse_bayf = calc_mse(snapshot['ours_bayf'][mask_score], target_score) 
        mse_kf = calc_mse(snapshot['ours_kf'][mask_score], target_score)     
        
        mse_list = []
        for i in range(self.num_anchors):
            val_p = snapshot['preds_all'][i][mask_score]
            mse_list.append(calc_mse(val_p, target_score))
        
        # 1. Ideal (Best Envelope): 当前帧里所有Experts中最小的那个误差
        best_mse_in_frame = min(mse_list)

        # 2. BestHist: 获取"累计表现最好的那个Expert"在"当前帧"的误差
        best_idx_cum = self.viz_cache.get("best_idx_cumulative", 0) # 如果还没统计，默认0
        mse_best_hist_expert = mse_list[best_idx_cum]
        
        # Feedback Loop
        self.fusion_model.update(mse_list, mse_def)
        
        # Store
        self.mse_history['def_mean'].append(mse_def)
        self.mse_history['ours_bayf'].append(mse_bayf)
        self.mse_history['ours_kf'].append(mse_kf)
        self.mse_history['best_envelope'].append(best_mse_in_frame)     # Ideal
        self.mse_history['best_hist_expert'].append(mse_best_hist_expert) # BestHist

        for i in range(self.num_anchors):
            self.mse_history[f'p{i}'].append(mse_list[i])
            
        # Update Cache
        self.viz_cache["mse_curr_values"] = mse_list
        self.viz_cache["def_mean_mse_curr"] = mse_def
        self.viz_cache["ours_bayf_curr"] = mse_bayf 
        self.viz_cache["ours_kf_curr"] = mse_kf     
        self.viz_cache["best_idx_curr"] = int(np.argmin(mse_list))
        
        if self.csv_file:
            frame_idx = len(self.mse_history['def_mean'])
            
            # --- 新增计算所有指标的平均值 ---
            def get_avg(k): 
                if not self.mse_history[k]: return 0.0
                return sum(self.mse_history[k]) / len(self.mse_history[k])
            
            avg_def = get_avg('def_mean')
            avg_bayf = get_avg('ours_bayf')
            avg_kf = get_avg('ours_kf')
            avg_ideal = get_avg('best_envelope')     # Avg Ideal
            avg_best_hist = get_avg('best_hist_expert') # Avg BestHist
            # ------------------

            # [CSV WRITE UPDATED] 写入11列数据
            self.csv_writer.writerow([
                frame_idx, 
                f"{mse_def:.4f}", f"{mse_bayf:.4f}", f"{mse_kf:.4f}", f"{best_mse_in_frame:.4f}", f"{mse_best_hist_expert:.4f}",
                f"{avg_def:.4f}", f"{avg_bayf:.4f}", f"{avg_kf:.4f}", f"{avg_ideal:.4f}", f"{avg_best_hist:.4f}"
            ])
            self.csv_file.flush()

        return True

    def _update_stats(self):
        def get_avg(lst): return sum(lst)/len(lst) if lst else 0.0
        
        vals = []
        for i in range(self.num_anchors):
            vals.append(get_avg(self.mse_history[f'p{i}']))
        
        self.viz_cache["mse_cum_values"] = vals
        if vals: self.viz_cache["best_idx_cumulative"] = int(np.argmin(vals))
            
        self.viz_cache["def_mean_mse_cum"] = get_avg(self.mse_history['def_mean'])
        self.viz_cache["ours_bayf_cum"] = get_avg(self.mse_history['ours_bayf']) 
        self.viz_cache["ours_kf_cum"] = get_avg(self.mse_history['ours_kf'])     
        self.viz_cache["best_envelope_mean"] = get_avg(self.mse_history['best_envelope'])
    
    def __del__(self):
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()

# ==========================================
# 4. Evaluator (Standard)
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
# 5. Inpaint Node
# ==========================================
class InpaintNode:
    def __init__(self):
        rospy.init_node("inpaint_benchmark_node")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(script_dir, "predict_config.yaml") 
        if not os.path.exists(yaml_path):
            node_cfg = {}
        else:
            with open(yaml_path, 'r') as f:
                full_config = yaml.safe_load(f)
                node_cfg = full_config.get('inpaint_node', {})

        topics = node_cfg.get('topics', {})
        self.topic_img_in = topics.get("input_image", "/sdf_map/2d/image")
        self.topic_map_in = topics.get("input_map", "/sdf_map/2d")
        self.topic_goal_in = topics.get("input_goal", "/move_base_simple/goal")
        # Sub to Odom for stop condition
        self.topic_odom_in = topics.get("input_odom", "/state_ukf/odom")
        
        self.topic_img_out = topics.get("output_image", "/inpainted/image")
        self.topic_dash_out = topics.get("output_dashboard", "/inpainted/dashboard")
        self.topic_map_out = topics.get("output_map", "/inpainted/map")

        model_cfg = node_cfg.get('model', {})
        run_path = model_cfg.get("checkpoint_path", "./checkpoints/wgan_inpainting")
        
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

        self.bench_evaluator = BenchmarkEvaluator(self.anchors, self.device, self.fusion_config)

        # State Control
        self.is_navigating = False
        self.current_goal_pose = None
        self.goal_tolerance = 5.0 # meters, stop MSE when closer than this

        self.sub = rospy.Subscriber(self.topic_img_in, RosImage, self.image_callback, queue_size=1, buff_size=2**24)
        self.pub = rospy.Publisher(self.topic_img_out, RosImage, queue_size=1)
        self.dashboard_pub = rospy.Publisher(self.topic_dash_out, RosImage, queue_size=1)
        self.map_pub = rospy.Publisher(self.topic_map_out, OccupancyGrid, queue_size=1)  
        
        self.map_info_ready = False
        self.map_resolution = 0.05
        self.map_origin_pose = Pose()
        self.map_sub = rospy.Subscriber(self.topic_map_in, OccupancyGrid, self.map_info_callback, queue_size=1)
        
        self.goal_received = False
        self.goal_sub = rospy.Subscriber(self.topic_goal_in, PoseStamped, self.goal_callback, queue_size=1)
        
        # Odom Sub
        self.odom_sub = rospy.Subscriber(self.topic_odom_in, Odometry, self.odom_callback, queue_size=1)

        rospy.loginfo(f"InpaintNode Init Complete.")

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

    def map_info_callback(self, map_msg):
        self.map_frame = map_msg.header.frame_id
        self.map_resolution = map_msg.info.resolution
        self.map_origin_pose = map_msg.info.origin
        self.map_info_ready = True

    def goal_callback(self, msg):
            if not self.map_info_ready: return
            self.current_goal_pose = msg.pose
            
            # === 修改开始 ===
            # 逻辑：只有当 self.goal_received 为 False (即第一次) 时才重置 Benchmark
            if not self.goal_received:
                self.bench_evaluator.reset_history()
                self.goal_received = True
                rospy.loginfo("First Goal received. STARTING MSE calculation (reset history).")
            else:
                rospy.loginfo("New Goal updated. Continuing MSE calculation (NO reset).")
            # === 修改结束 ===
            
            # 无论是否重置，只要收到新目标，就确保存储/录制状态开启
            self.is_navigating = True

    def odom_callback(self, msg):
        """Check distance to goal to STOP MSE recording."""
        if not self.is_navigating or not self.current_goal_pose:
            return
            
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        gx = self.current_goal_pose.position.x
        gy = self.current_goal_pose.position.y
        
        dist = math.sqrt((px - gx)**2 + (py - gy)**2)
        
        if dist < self.goal_tolerance:
            self.is_navigating = False
            rospy.loginfo(f"Robot reached goal (dist={dist:.2f}m). STOPPING MSE calculation.")

    def inject_goal_region_to_input(self, cv_img):
            if not self.goal_received or not self.map_info_ready: return
            res = self.map_resolution
            H, W = cv_img.shape[:2]
            cx, cy = W // 2, H // 2
            gx = int(self.current_goal_pose.position.x / res)
            gy = int(self.current_goal_pose.position.y / res)
            radius = int(0.5 / res) # 0.5米的半径
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    row, col = cy - (gx + dx), cx - (gy + dy)
                    
                    # 越界检查
                    if 0 <= row < H and 0 <= col < W: 
                        # === [修改核心] ===
                        # 获取当前像素值
                        pixel_val = cv_img[row, col]
                        
                        # 逻辑：只有当它是“灰色/未知”区域时，才应用 Goal 的强制涂白。
                        # 127 是未知。考虑到浮点转换误差，设定一个由 127 左右的范围。
                        # 如果它是 0 (障碍) 或者 255 (已知空地)，都不要动它。
                        if abs(int(pixel_val) - 127) < 5: 
                            cv_img[row, col] = 255

    def image_to_occupancy_grid(self, cv_image, img_header):
        grid = OccupancyGrid()
        grid.header.stamp = img_header.stamp
        grid.header.frame_id = self.map_frame or img_header.frame_id
        rotated = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        aligned = np.flipud(rotated)
        h, w = aligned.shape[:2]
        grid.info.width, grid.info.height = w, h
        grid.info.resolution = self.map_resolution
        grid.info.origin = self.map_origin_pose
        prob = (255 - aligned.astype(np.float32)) / 255.0 * 100.0
        grid.data = np.clip(prob, 0, 100).astype(np.int8).flatten().tolist()
        return grid

    # ==================================================
    # Dashboard: White Background Chart with Axes
    # ==================================================
    def draw_mse_chart(self, mse_history, evaluator_cache, width, height):
        # 1. Initialize WHITE Background (255)
        chart = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        def get_smooth(lst, win=5):
            if not lst: return []
            if len(lst) < win: return lst
            ret = np.cumsum(lst, dtype=float)
            ret[win:] = ret[win:] - ret[:-win]
            return (ret[win - 1:] / win).tolist()

        def_data = get_smooth(mse_history['def_mean'])
        ours_bayf_data = get_smooth(mse_history['ours_bayf']) # Updated
        pink_data = get_smooth(mse_history['best_envelope']) 
        ours_kf_data = get_smooth(mse_history['ours_kf'])     # Updated
        
        if not def_data: return chart

        all_vals = def_data + ours_bayf_data + pink_data + ours_kf_data
        
        # [MODIFIED] Better scaling to prevent truncation
        global_min = min(all_vals)
        global_max = max(all_vals)
        val_range = global_max - global_min
        
        # Add 10-20% margin
        margin = val_range * 0.15 if val_range > 1e-3 else 1.0
        min_val = max(0, global_min - margin)
        max_val = global_max + margin
        
        N = len(def_data)

        # 2. Define Margins for Axes
        pad_l, pad_r, pad_t, pad_b = 45, 15, 20, 30 # Slightly increased margins
        plot_w = width - pad_l - pad_r
        plot_h = height - pad_t - pad_b
        
        step_x = plot_w / (N - 1) if N > 1 else 0
        
        # Coordinate mapping
        def val_to_y(v):
            # Safe divide
            denom = (max_val - min_val) if (max_val - min_val) > 1e-5 else 1.0
            ratio = (v - min_val) / denom
            # Y flips in image coords
            return int((height - pad_b) - ratio * plot_h)
            
        def draw_line(data, color, thick=1):
            if len(data) < 2: return
            pts = []
            for i, v in enumerate(data):
                px = int(pad_l + i * step_x)
                py = val_to_y(v)
                # Clip to plot area just in case
                py = np.clip(py, pad_t, height - pad_b)
                pts.append((px, py))
            cv2.polylines(chart, [np.array(pts)], False, color, thick, lineType=cv2.LINE_AA)

        # 3. Draw Axes (Black)
        # Y-Axis
        cv2.line(chart, (pad_l, pad_t), (pad_l, height - pad_b), (0,0,0), 2)
        # X-Axis
        cv2.line(chart, (pad_l, height - pad_b), (width - pad_r, height - pad_b), (0,0,0), 2)

        # 4. Draw Ticks & Labels (Black Text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Y-Ticks (5 steps)
        for i in range(5):
            ratio = i / 4.0
            val = min_val + ratio * (max_val - min_val)
            y_pos = int((height - pad_b) - ratio * plot_h)
            
            # Tick line
            cv2.line(chart, (pad_l - 4, y_pos), (pad_l, y_pos), (0,0,0), 1)
            # Text (Right aligned roughly)
            txt = f"{val:.1f}"
            cv2.putText(chart, txt, (2, y_pos + 4), font, 0.4, (0,0,0), 1)
            
            # Optional: Grid line (Light Grey)
            if i > 0:
                cv2.line(chart, (pad_l, y_pos), (width - pad_r, y_pos), (220,220,220), 1)

        # X-Ticks (5 steps)
        for i in range(5):
            ratio = i / 4.0
            idx = int(ratio * (N - 1)) if N > 1 else 0
            x_pos = int(pad_l + ratio * plot_w)
            
            # Tick line
            cv2.line(chart, (x_pos, height - pad_b), (x_pos, height - pad_b + 4), (0,0,0), 1)
            # Text
            cv2.putText(chart, str(idx), (x_pos - 5, height - 5), font, 0.4, (0,0,0), 1)

        # 5. Background P0-P15 (Pastel/Light Colors)
        base_colors = [
            (180, 200, 200), (200, 180, 200), (200, 200, 180), (180, 180, 200),
            (210, 190, 190), (190, 210, 190), (190, 190, 210), (210, 210, 190),
            (200, 220, 220), (220, 200, 220), (220, 220, 200), (200, 200, 220),
            (230, 210, 210), (210, 230, 210), (210, 210, 230), (230, 230, 210)
        ]
        best_cum_idx = evaluator_cache.get("best_idx_cumulative", 0)
        
        for i in range(16):
            key = f'p{i}'
            if key not in mse_history: continue
            raw = mse_history[key]
            smooth = get_smooth(raw)
            if i == best_cum_idx:
                draw_line(smooth, (0, 150, 0), 2) 
            else:
                draw_line(smooth, (200, 200, 200), 1)

        # 6. Main Lines
        # DefMean (Cyan -> Dark Cyan/Teal)
        draw_line(def_data, (200, 180, 0), 2)  
        # Ideal (Pink -> Magenta)
        draw_line(pink_data, (200, 0, 200), 2) 
        # Ours(BayF) (Yellow -> Orange/Gold)
        draw_line(ours_bayf_data, (0, 165, 255), 2) 
        # Ours(KF) (Blue -> Dark Blue)
        draw_line(ours_kf_data, (200, 0, 0), 2)     
        
        return chart

    def create_dashboard_image(self, input_cv_img, evaluator):
                IMG_SZ = 160
                cache = evaluator.viz_cache
                preds_tensor = cache.get("all_preds_realtime", None)
                
                if preds_tensor is None:
                    return np.zeros((600, 800, 3), dtype=np.uint8)

                # 确保预测张量在CPU (用于绘图)
                device = preds_tensor.device 

                # ==========================================
                # 1. 准备核心图像 (Ours, Def, Best, KF)
                # ==========================================
                
                # --- (A) DefMean (基准) ---
                def_mean_tensor = preds_tensor[:7].mean(dim=0)

                # --- (B) Ours (修复闪烁问题) ---
                weights_ours = cache.get("current_weights", None)
                
                # 逻辑必须与 BetterThanBaselineFusion.predict 保持完全一致
                if weights_ours is None or np.sum(weights_ours) < 1e-6:
                    ours_tensor = def_mean_tensor
                else:
                    w_torch = torch.tensor(weights_ours, device=device)
                    if preds_tensor.dim() == 4: 
                        w_broad = w_torch.view(-1, 1, 1, 1)
                    else: 
                        w_broad = w_torch.view(-1, 1, 1)
                    
                    fused_experts = (preds_tensor * w_broad).sum(dim=0)
                    ours_tensor = 0.35 * fused_experts + 0.65 * def_mean_tensor

                # --- (C) BestHist (历史最佳) ---
                best_idx_cum = cache.get("best_idx_cumulative", 0)
                best_hist_tensor = preds_tensor[best_idx_cum]

                # --- (D) KF (修复设备报错) ---
                kf_vars = evaluator.kf_model.variances.to(device) # 强制转到 CPU
                kf_temp = evaluator.kf_model.temperature
                kf_weights = F.softmax(-kf_vars / kf_temp, dim=0)
                
                if preds_tensor.dim() == 4:
                    kw_expanded = kf_weights.view(-1, 1, 1, 1)
                else:
                    kw_expanded = kf_weights.view(-1, 1, 1)
                kf_tensor = (preds_tensor * kw_expanded).sum(dim=0)

                # ==========================================
                # Helper: Process Img [修改部分]
                # ==========================================
                def process_img(img_src, label, sz=(IMG_SZ, IMG_SZ), color=(128,128,128)):
                    if img_src is None: 
                        return np.zeros((sz[1], sz[0], 3), dtype=np.uint8)
                    
                    # 1. 处理数据格式
                    if isinstance(img_src, torch.Tensor): 
                        arr = img_src.squeeze().detach().cpu().numpy()
                    else: 
                        arr = img_src
                    
                    if arr.dtype != np.uint8:
                        arr = np.clip((arr + 1) / 2 * 255, 0, 255).astype(np.uint8)

                    # 2. 调整图像大小
                    resized = cv2.resize(arr, sz, interpolation=cv2.INTER_NEAREST)
                    if len(resized.shape) == 2:
                        bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                    else:
                        bgr = resized

                    # 3. 创建独立的标签栏 (Header)
                    label_height = 24  # 稍微增加高度以容纳文字
                    header = np.zeros((label_height, sz[0], 3), dtype=np.uint8)
                    
                    # 填充标签栏背景色
                    cv2.rectangle(header, (0, 0), (sz[0], label_height), color, -1)
                    
                    # 写字 (在 Header 上)
                    cv2.putText(header, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

                    # 4. 垂直拼接：Header 在上，Map 在下
                    # 使用 vstack 物理隔离，绝对不会遮挡 Map
                    labeled_img = np.vstack([header, bgr])
                    
                    # (可选) 如果你需要给整体加个极细的黑框，可以在这里加，但不要遮挡内容
                    # cv2.rectangle(labeled_img, (0, 0), (labeled_img.shape[1]-1, labeled_img.shape[0]-1), (0,0,0), 1)

                    return labeled_img

                # ==========================================
                # 2. ROW 1: Current Frame Full Maps
                # ==========================================
                img_input = process_img(input_cv_img, "Input", color=(150,150,150))
                img_ours  = process_img(ours_tensor, "Ours(BayF)", color=(0, 255, 255))      
                img_def   = process_img(def_mean_tensor, "DefMean", color=(255, 255, 0)) 
                img_best  = process_img(best_hist_tensor, f"BestHist(P{best_idx_cum})", color=(0, 255, 0)) 
                img_kf    = process_img(kf_tensor, "Ours(KF)", color=(255, 0, 0))            

                top_section = np.hstack([img_input, img_ours, img_def, img_best, img_kf])

                # ==========================================
                # [NEW] ROW 2: DELAYED Input & Overlay Prediction on Newly Discovered Area
                # ==========================================
                delayed_snap = cache.get("delayed_snapshot", None)
                
                # --- 修改后的帮助函数：将预测叠加在老地图上 ---
                def get_overlaid_pred(pred_t, base_img, mask_diff, lbl, color):
                    # 1. 转换预测结果为 Numpy (0-255)
                    if isinstance(pred_t, torch.Tensor):
                        arr = pred_t.squeeze().detach().cpu().numpy()
                    else:
                        arr = pred_t
                    if arr.dtype != np.uint8:
                        arr = np.clip((arr + 1) / 2 * 255, 0, 255).astype(np.uint8)
                    
                    # 2. 统一尺寸 (以预测图 arr 为准)
                    h, w = arr.shape
                    # 将底图(老地图)缩放到同样大小
                    base_resized = cv2.resize(base_img, (w, h), interpolation=cv2.INTER_NEAREST)
                    # 将Mask缩放到同样大小
                    mask_resized = cv2.resize(mask_diff.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # 3. 图像融合 (Overlay)
                    # 初始图像 = 老地图
                    final_img = base_resized.copy()
                    
                    # 核心逻辑：仅在 mask_diff 为 True (新发现区域) 的位置，
                    # 用 arr (预测值) 覆盖 base_resized (老地图)
                    # 其余位置保持老地图的样子
                    roi_indices = (mask_resized > 0)
                    final_img[roi_indices] = arr[roi_indices]
                    
                    return process_img(final_img, lbl, color=color)

                if delayed_snap is None:
                    # Buffer not full: Empty black placeholders
                    blank = np.zeros((IMG_SZ, IMG_SZ, 3), dtype=np.uint8)
                    # 使用 process_img 保持格式一致
                    blank_labeled = process_img(blank, "Filling Buffer (6s)...", color=(50,50,50))
                    row2_imgs = [blank_labeled] * 5
                else:
                    # 1. 获取6秒前的原始输入 (作为底图 Background)
                    prev_raw_t = delayed_snap['belief'] # Tensor [-1, 1]
                    prev_raw_np = (prev_raw_t.squeeze().detach().cpu().numpy() + 1) / 2 * 255
                    prev_raw_np = np.clip(prev_raw_np, 0, 255).astype(np.uint8)
                    
                    # 2. 计算当前帧的尺寸以便对齐 Mask
                    curr_h, curr_w = input_cv_img.shape[:2]

                    # 3. 计算 Mask: 
                    # 定义：当前是"已知"(Known) 且 6秒前是"未知"(Unknown) 的区域
                    # 注意：Tensor中0.0代表灰色未知区域 (归一化后)
                    prev_t_cpu = delayed_snap['belief'].squeeze().detach().cpu()
                    mask_prev_unknown = torch.abs(prev_t_cpu) < 0.1  # 6秒前是灰色的
                    
                    # 将 mask 调整到当前输入大小以便做逻辑运算
                    mask_prev_unknown_np = cv2.resize(mask_prev_unknown.numpy().astype(np.uint8), (curr_w, curr_h), interpolation=cv2.INTER_NEAREST).astype(bool)

                    # 当前输入中不等于127的地方视为已知 (127是灰色)
                    mask_curr_known = np.abs(input_cv_img.astype(float) - 127) > 5
                    
                    # 最终的新发现区域 Mask
                    mask_diff = mask_prev_unknown_np & mask_curr_known
                    
                    # 4. 第一列：显示纯净的 6秒前原始输入 (参考基准)
                    img_prev_raw = process_img(prev_raw_np, "Prev Raw (6s ago)", color=(100,100,100))
                    
                    # 5. 后四列：调用新的 get_overlaid_pred
                    # 传入参数：(预测图, 底图prev_raw_np, Mask, 标签, 颜色)
                    
                    snap_ours = delayed_snap['ours_bayf']
                    snap_def = delayed_snap['def_mean']
                    snap_kf = delayed_snap['ours_kf']
                    snap_best = delayed_snap['preds_all'][best_idx_cum]
                    
                    img_ours_new = get_overlaid_pred(snap_ours, prev_raw_np, mask_diff, "BayF Overlay", (0, 255, 255))
                    img_def_new = get_overlaid_pred(snap_def, prev_raw_np, mask_diff, "DefMean Overlay", (255, 255, 0))
                    img_best_new = get_overlaid_pred(snap_best, prev_raw_np, mask_diff, "BestHist Overlay", (0, 255, 0))
                    img_kf_new = get_overlaid_pred(snap_kf, prev_raw_np, mask_diff, "KF Overlay", (255, 0, 0))
                    
                    row2_imgs = [img_prev_raw, img_ours_new, img_def_new, img_best_new, img_kf_new]

                mid_section = np.hstack(row2_imgs)

                # ==========================================
                # 3. 状态栏 (修复 Ideal 缺失)
                # ==========================================
                W_FULL = top_section.shape[1]
                H_TABLE = 60
                table_img = np.zeros((H_TABLE, W_FULL, 3), dtype=np.uint8)
                
                # 数据获取
                ours_bayf_curr = cache.get("ours_bayf_curr", 0.0) # Updated
                ours_bayf_cum  = cache.get("ours_bayf_cum", 0.0)  # Updated
                def_curr  = cache.get("def_mean_mse_curr", 0.0)
                def_cum   = cache.get("def_mean_mse_cum", 0.0)
                ours_kf_curr   = cache.get("ours_kf_curr", 0.0)   # Updated
                ours_kf_cum    = cache.get("ours_kf_cum", 0.0)    # Updated
                
                mse_curr  = cache.get("mse_curr_values", [0.0]*16)
                mse_cum   = cache.get("mse_cum_values", [0.0]*16)
                
                # Best Hist 数据
                best_hist_curr = mse_curr[best_idx_cum] if best_idx_cum < len(mse_curr) else 0.0
                best_hist_cum  = mse_cum[best_idx_cum] if best_idx_cum < len(mse_cum) else 0.0

                # Ideal (Best Envelope) 数据 [之前遗漏的]
                # best_idx_curr 是当前帧最好的那个 expert 的索引
                best_idx_curr = cache.get("best_idx_curr", 0)
                ideal_curr = mse_curr[best_idx_curr] if best_idx_curr < len(mse_curr) else 0.0
                ideal_cum  = cache.get("best_envelope_mean", 0.0)

                info_text = cache.get("info_text", "")
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Line 1: Status
                cv2.putText(table_img, f"State: {info_text}", (10, 20), font, 0.5, (200,200,200), 1)

                # Line 2: Metrics (5列布局: Ours, Def, BestHist, Ideal, KF)
                y_txt = 45
                col_w = W_FULL // 5  # 分成5份以容纳 Ideal
                
                # 1. Ours(BayF) (Yellow)
                cv2.putText(table_img, f"Ours(BayF): {ours_bayf_curr:.1f}/{ours_bayf_cum:.1f}", (5, y_txt), font, 0.45, (0, 255, 255), 1)
                # 2. DefMean (Cyan)
                cv2.putText(table_img, f"Def: {def_curr:.1f}/{def_cum:.1f}", (5 + col_w, y_txt), font, 0.45, (255, 255, 0), 1)
                # 3. BestHist (Green)
                cv2.putText(table_img, f"B.Hist: {best_hist_curr:.1f}/{best_hist_cum:.1f}", (5 + col_w*2, y_txt), font, 0.45, (0, 255, 0), 1)
                # 4. Ideal (Pink) [已找回]
                cv2.putText(table_img, f"Ideal: {ideal_curr:.1f}/{ideal_cum:.1f}", (5 + col_w*3, y_txt), font, 0.45, (255, 0, 255), 1)
                # 5. Ours(KF) (Blue)
                cv2.putText(table_img, f"Ours(KF): {ours_kf_curr:.1f}/{ours_kf_cum:.1f}", (5 + col_w*4, y_txt), font, 0.45, (255, 0, 0), 1)

                # ==========================================
                # 4. Chart
                # ==========================================
                chart_h = 180
                chart_img = self.draw_mse_chart(evaluator.mse_history, evaluator.viz_cache, W_FULL, chart_h)

                final = np.vstack([top_section, mid_section, table_img, chart_img])
                return final
    def image_callback(self, msg):
            try:
                cv_img_clean = self.imgmsg_to_cv2(msg)
            except Exception: return

            # 1. 准备两份数据
            # A. 注入Goal的图 (仅用于生成 Ours 预测结果)
            cv_img_goal = cv_img_clean.copy()
            self.inject_goal_region_to_input(cv_img_goal)
            
            # B. 纯净的原图 (用于显示、作为 Ground Truth、以及生成 Baseline 预测)
            # 预处理...
            x_goal, mask_goal, x_raw_goal = preprocess_robot_belief_input(cv_img_goal, self.config['image_shape'], self.device)
            x_clean, mask_clean, x_raw_clean = preprocess_robot_belief_input(cv_img_clean, self.config['image_shape'], self.device)

            # 2. 网络推理
            # Ours(BayF) 和 KF 需要用到基于 Goal 优化的预测结果
            all_preds_goal = self.evaluator.eval_batch(x_goal, mask_goal, self.anchors_tensor, cv_img_goal.shape[::-1])
            # Baseline 使用纯净图的预测结果
            all_preds_clean = self.evaluator.eval_batch(x_clean, mask_clean, self.anchors_tensor, cv_img_clean.shape[::-1])
            
            current_time = msg.header.stamp.to_sec()
            
            # 3. Benchmark Step (修改点：传入 x_raw_clean)
            # 我们希望 MSE 评估的是“预测值 vs 真实传感器原图”，而不是“预测值 vs 注入Goal的图”
            # 同时，这保证了存入历史队列的 'belief' 是纯净的，下次显示 Previous Raw 时也是纯净的
            nav_map_out = self.bench_evaluator.step(
                x_raw_clean,       # <--- [修改] 传入纯净的 Ground Truth
                all_preds_goal,    # 预测结果依然使用优化过的 all_preds_goal
                current_time, 
                preds_clean=all_preds_clean, 
                record=self.is_navigating
            )

            try:
                # 4. Dashboard (修改点：传入 cv_img_clean)
                # 确保 Dashboard 显示的是不带 Goal 白点的原始输入
                dashboard_img = self.create_dashboard_image(cv_img_clean, self.bench_evaluator) # <--- [修改]
                dash_msg = self.cv2_to_imgmsg(dashboard_img, encoding='bgr8')
                dash_msg.header = msg.header
                self.dashboard_pub.publish(dash_msg)
            except Exception as e:
                rospy.logerr(f"Dash Error: {e}")

            # 5. 发布最终用于导航的地图
            processed_img_np = self.evaluator.post_process_probability(nav_map_out, x_raw_clean) # <--- [建议] 这里也可以改用 clean 保证底图纯净
            np_result = (np.clip((processed_img_np + 1) / 2, 0, 1) * 255).astype(np.uint8)

            out_msg = self.cv2_to_imgmsg(np_result, encoding='mono8')
            out_msg.header = msg.header
            self.pub.publish(out_msg)
            if self.map_info_ready:
                self.map_pub.publish(self.image_to_occupancy_grid(np_result, msg.header))

if __name__ == "__main__":
    node = InpaintNode()
    rospy.spin()