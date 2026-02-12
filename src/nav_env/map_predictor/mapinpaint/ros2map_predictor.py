#!/usr/bin/env python3
import os
import rclpy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
import torchvision.transforms as transforms
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from cv_bridge import CvBridge
from .networks import Generator
from .tools import get_config

class Evaluator:
    def __init__(self, config, netG, cuda, nsample=1):
        self.config = config
        self.use_cuda = cuda
        self.nsample = nsample
        self.netG = netG
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.netG.to(self.device)

    @torch.no_grad()
    def eval_step(self, x, mask, onehot, img_raw_size, ground_truth=None, calc_metrics=False):
        self.netG.eval()
        x_out = self.netG(x, mask, onehot)
        inpainted_result = x_out * mask + x * (1. - mask)

        width, height = x.size(2), x.size(3)
        crop = img_raw_size[0] < width and img_raw_size[1] < height
        if crop:
            i_left = (width - img_raw_size[0]) // 2
            i_top = (height - img_raw_size[1]) // 2
            i_right = i_left + img_raw_size[0]
            i_bottom = i_top + img_raw_size[1]
            inpainted_result = inpainted_result[:, :, i_left:i_right, i_top:i_bottom]
        else:
            inpainted_result = F.interpolate(inpainted_result, size=(img_raw_size[1], img_raw_size[0]), mode='bilinear', align_corners=False)

        return {}, inpainted_result

    @staticmethod
    def post_process(inpaint, x, kernel_size=5, return_tensor=False):
        unique_values, counts = torch.unique(x, return_counts=True)
        k = min(3, counts.size(0))
        topk_indices = torch.topk(counts, k=k).indices
        topk_values = unique_values[topk_indices]
        obs_v, free_v = topk_values.min(), topk_values.max()

        inpaint = torch.where(inpaint > -0.3, free_v, obs_v)
        binary_img = inpaint.cpu().numpy()[0, 0]
        obs_v = obs_v.item()
        free_v = free_v.item()

        mask = np.zeros_like(binary_img, dtype=np.uint8)
        mask[binary_img == free_v] = 255
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
        morph_clean_img = np.where(opening == 255, free_v, obs_v).astype(binary_img.dtype)
        x_array = x.cpu().numpy()[0, 0]
        morph_clean_img = np.where((x_array == obs_v) | (x_array == free_v), x_array, morph_clean_img)
        if return_tensor:
            morph_clean_img = torch.from_numpy(morph_clean_img).unsqueeze(0).unsqueeze(0).float().to(inpaint.device)
        return morph_clean_img


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


class InpaintNode(Node):
    def __init__(self):
        super().__init__('inpaint_node')

        run_path = self.get_parameter("checkpoint_path").get_parameter_value().string_value
        config_path = os.path.join(run_path, "config.yaml")
        checkpoint_file = next(f for f in os.listdir(run_path) if f.startswith('gen') and f.endswith('.pt'))
        checkpoint_path = os.path.join(run_path, checkpoint_file)

        config = get_config(config_path)
        cuda = config['cuda']
        if cuda:
            cudnn.benchmark = True

        self.device = torch.device('cuda' if cuda else 'cpu')
        self.netG = Generator(config['netG'], cuda)
        self.netG.load_state_dict(torch.load(checkpoint_path))
        self.netG.eval()

        self.evaluator = Evaluator(config, self.netG, cuda)
        self.config = config
        self.bridge = CvBridge()

        self.sub = self.create_subscription(RosImage, "/sdf_map/2d/image", self.image_callback, 10)
        self.pub = self.create_publisher(RosImage, "/inpainted/image", 10)

        self.map_info_ready = False
        self.map_frame = None
        self.map_resolution = None
        self.map_origin_pose = Pose()
        self.map_size = None
        self.map_sub = self.create_subscription(OccupancyGrid, "/sdf_map/2d", self.map_info_callback, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, "/inpainted/map", 10)

        self.goal_pose = None
        self.goal_received = False
        self.goal_sub = self.create_subscription(PoseStamped, "/move_base_simple/goal", self.goal_callback, 10)

        self.get_logger().info("InpaintNode initialized and listening to /sdf_map/2d/image")

    def imgmsg_to_cv2(self, img_msg):
        return self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')

    def map_info_callback(self, map_msg: OccupancyGrid):
        self.map_frame = map_msg.header.frame_id
        self.map_resolution = map_msg.info.resolution
        self.map_origin_pose = map_msg.info.origin
        self.map_size = (map_msg.info.width, map_msg.info.height)
        self.map_info_ready = True

    def goal_callback(self, msg: PoseStamped):
        if not self.map_info_ready:
            self.get_logger().warn("Map info not ready. Goal ignored.")
            return

        self.goal_pose = msg.pose
        self.goal_received = True
        self.get_logger().info(f"Received goal at ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

    def image_callback(self, msg: RosImage):
        try:
            cv_img = self.imgmsg_to_cv2(msg).copy()
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        if self.goal_received and self.map_info_ready:
            self.inject_goal_region_to_input(cv_img)

        x, mask, x_raw = preprocess_robot_belief_input(cv_img, self.config['image_shape'], self.device)

        onehots = torch.tensor([
            [0.333, 0.333, 0.333],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ]).unsqueeze(1).float().to(self.device)

        all_results = []
        for i in range(onehots.size(0)):
            onehot = onehots[i]
            _, inpainted = self.evaluator.eval_step(x, mask, onehot, img_raw_size=cv_img.shape[::-1])
            all_results.append(inpainted)

        avg_inpainted_result = torch.stack(all_results, dim=0).mean(dim=0)

        np_result = avg_inpainted_result.squeeze().detach().cpu().numpy()
        np_result = np.clip((np_result + 1) / 2, 0, 1)  # scale from [-1,1] to [0,1]
        np_result = (np_result * 255).astype(np.uint8)

        try:
            out_msg = self.bridge.cv2_to_imgmsg(np_result, encoding='mono8')
            out_msg.header = msg.header
            self.pub.publish(out_msg)
            grid_msg = self.image_to_occupancy_grid(np_result, msg.header)
            self.map_pub.publish(grid_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to publish inpainted image: {e}")

    def inject_goal_region_to_input(self, cv_img):
        if not self.goal_received or not self.map_info_ready:
            return

        goal_x = self.goal_pose.position.x
        goal_y = self.goal_pose.position.y

        res = self.map_resolution
        H, W = cv_img.shape[:2]
        cx = W // 2
        cy = H // 2

        gx = int(goal_x / res)
        gy = int(goal_y / res)

        radius_cells = int(0.5 / res)

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                gx_ = gx + dx
                gy_ = gy + dy

                row = cy - gx_
                col = cx - gy_

                if 0 <= row < H and 0 <= col < W:
                    cv_img[row, col] = 255  # white, indicating free space


def main(args=None):
    rclpy.init(args=args)
    node = InpaintNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
