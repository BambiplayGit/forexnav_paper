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
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image as RosImage
# from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped

from networks import Generator
from tools import get_config


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


class InpaintNode:
    def __init__(self):
        rospy.init_node("inpaint_node")

        run_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'wgan_inpainting')
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
        # self.bridge = CvBridge()

        self.sub = rospy.Subscriber("/sdf_map/2d/image", RosImage, self.image_callback, queue_size=1, buff_size=2**24)
        self.pub = rospy.Publisher("/inpainted/image", RosImage, queue_size=1)

        # ✅ 新增：从已存在的 grid map 读取元信息（分辨率 / 原点 / 大小 / frame）
        src_map_topic = rospy.get_param("/sdf_map/2d", "/sdf_map/2d")  # 如有不同请用参数改
        self.map_info_ready = False
        self.map_frame = None
        self.map_resolution = None
        self.map_origin_pose = Pose()
        self.map_size = None  # (width, height)
        self.map_sub = rospy.Subscriber(src_map_topic, OccupancyGrid, self.map_info_callback, queue_size=1)
        self.map_pub = rospy.Publisher("/inpainted/map", OccupancyGrid, queue_size=1)  

        self.goal_pose = None
        self.goal_received = False
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)

        rospy.loginfo("InpaintNode initialized and listening to /sdf_map/2d/image")

    def imgmsg_to_cv2(self, img_msg):
        import sys
        import numpy as np

        # 支持的编码：mono8, bgr8
        if img_msg.encoding == "mono8":
            channels = 1
        elif img_msg.encoding == "bgr8":
            channels = 3
        else:
            raise ValueError(f"[imgmsg_to_cv2] Unsupported image encoding: {img_msg.encoding}")

        dtype = np.dtype("uint8")
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')

        if channels == 1:
            shape = (img_msg.height, img_msg.width)
        else:  # channels == 3
            shape = (img_msg.height, img_msg.width, 3)

        expected_bytes = np.prod(shape)
        if len(img_msg.data) < expected_bytes:
            raise ValueError(f"[imgmsg_to_cv2] Buffer too small: expected {expected_bytes} bytes, got {len(img_msg.data)}")

        image_opencv = np.ndarray(shape=shape, dtype=dtype, buffer=img_msg.data)

        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            image_opencv = image_opencv.byteswap().newbyteorder()

        return image_opencv
    def map_info_callback(self, map_msg: OccupancyGrid):
        self.map_frame = map_msg.header.frame_id
        self.map_resolution = map_msg.info.resolution
        self.map_origin_pose = map_msg.info.origin
        self.map_size = (map_msg.info.width, map_msg.info.height)
        self.map_info_ready = True
    def goal_callback(self, msg: PoseStamped):
        if not self.map_info_ready:
            rospy.logwarn("Map info not ready. Goal ignored.")
            return

        self.goal_pose = msg.pose
        self.goal_received = True
        rospy.loginfo(f"Received goal at ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
    def cv2_to_imgmsg(self, cv_image, encoding='mono8'):
        """
        Convert a numpy ndarray (OpenCV image) to a ROS Image message.
        Currently supports only mono8.
        """
        if encoding != 'mono8':
            raise ValueError(f"[cv2_to_imgmsg] Only 'mono8' encoding is supported. Got: {encoding}")

        if not isinstance(cv_image, np.ndarray):
            raise TypeError("[cv2_to_imgmsg] Input must be a numpy.ndarray")

        if cv_image.dtype != np.uint8:
            raise TypeError(f"[cv2_to_imgmsg] Image dtype must be uint8, got {cv_image.dtype}")

        if len(cv_image.shape) != 2:
            raise ValueError(f"[cv2_to_imgmsg] mono8 image must be 2D (H, W), got shape: {cv_image.shape}")

        msg = RosImage()
        msg.height = cv_image.shape[0]
        msg.width = cv_image.shape[1]
        msg.encoding = encoding
        msg.is_bigendian = (sys.byteorder == 'big')
        msg.step = msg.width  # mono8: 1 byte per pixel
        msg.data = cv_image.tobytes()

        return msg
    def image_to_occupancy_grid(self, cv_image, img_header):
        """
        将预测图变换为 OccupancyGrid：
        - 顺时针旋转 90 度
        - 上下翻转（图像原点左上 → ROS 地图原点左下）
        - 使用用户提供的 origin / resolution（必须确保你是对的）
        """
        grid = OccupancyGrid()
        grid.header.stamp = img_header.stamp
        grid.header.frame_id = self.map_frame or img_header.frame_id

        # 🚨 先旋转90度：顺时针 (H,W) -> (W,H)
        rotated = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)

        # 🚨 再上下翻转，使得左上角对齐 ROS 左下角
        aligned = np.flipud(rotated)

        h, w = aligned.shape[:2]
        grid.info.width = w
        grid.info.height = h
        grid.info.resolution = self.map_resolution  # ✅ 你输入网络之前就知道的值
        grid.info.origin = self.map_origin_pose     # ✅ 你输入图在原始地图中的左下角世界坐标

        # ✅ 0-255 转占据概率（0: free, 100: occupied）
        prob = (255 - aligned.astype(np.float32)) / 255.0 * 100.0
        prob = np.clip(prob, 0, 100)

        grid.data = prob.astype(np.int8).flatten().tolist()
        return grid
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

                row = cy - gx_   # x 轴向上 -> row 向下
                col = cx - gy_   # y 轴向左 -> col 向右

                if 0 <= row < H and 0 <= col < W:
                    cv_img[row, col] = 255  # 白色区域，表示 free
    def image_callback(self, msg):
        try:
            import cv_bridge
            # print(cv_bridge.__file__)
            cv_img = self.imgmsg_to_cv2(msg).copy()
        except Exception as e:
            rospy.logerr("Failed to convert image: %s", e)
            return
        # ✅ 在 cv_img 上根据 goal_pose 注入白色 255 区域
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
        # inpainted_processed = self.evaluator.post_process(avg_inpainted_result, x_raw, return_tensor=True)

        np_result = avg_inpainted_result.squeeze().detach().cpu().numpy()
        np_result = np.clip((np_result + 1) / 2, 0, 1)  # scale from [-1,1] to [0,1]
        np_result = (np_result * 255).astype(np.uint8)

        try:
            out_msg = self.cv2_to_imgmsg(np_result, encoding='mono8')
            out_msg.header = msg.header
            self.pub.publish(out_msg)
            grid_msg = self.image_to_occupancy_grid(np_result, msg.header)
            self.map_pub.publish(grid_msg)

        except Exception as e:
            rospy.logerr("Failed to publish inpainted image: %s", e)


if __name__ == "__main__":
    node = InpaintNode()
    rospy.spin()
