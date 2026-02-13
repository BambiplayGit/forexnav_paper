import os
from argparse import ArgumentParser
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
import torchvision.transforms as transforms

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


def to_numpy_image(t):
    img = t.squeeze().detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def main():
    run_path = './checkpoints/wgan_inpainting'
    config_path = f'{run_path}/config.yaml'
    checkpoint_path = os.path.join(run_path, [f for f in os.listdir(run_path) if f.startswith('gen') and f.endswith('.pt')][0])
    
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path)
    parser.add_argument('--img', type=str, required=True, help="path to input robot_belief image")
    args = parser.parse_args()

    config = get_config(args.config)
    cuda = config['cuda']
    if cuda:
        cudnn.benchmark = True

    device = torch.device('cuda' if cuda else 'cpu')
    netG = Generator(config['netG'], cuda)
    netG.load_state_dict(torch.load(checkpoint_path))
    netG.eval()
    print("Loaded model from", checkpoint_path)

    evaluator = Evaluator(config, netG, cuda)

    # 加载图像（robot_belief）
    raw_img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
    assert raw_img is not None, "图像加载失败"
    raw_img = raw_img.astype(np.uint8)

    x, mask, x_raw = preprocess_robot_belief_input(raw_img, config['image_shape'], device)

    # 多个 onehot 做 ensemble
    onehots = torch.tensor([
        [0.333, 0.333, 0.333],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],
    ]).unsqueeze(1).float().to(device)

    all_results = []
    for i in range(onehots.size(0)):
        onehot = onehots[i]  # [1, 3]
        _, inpainted = evaluator.eval_step(x, mask, onehot, img_raw_size=raw_img.shape[::-1])
        all_results.append(inpainted)

    avg_inpainted_result = torch.stack(all_results, dim=0).mean(dim=0)
    inpainted_processed = evaluator.post_process(avg_inpainted_result, x_raw, return_tensor=True)

    # 显示
    img_input = to_numpy_image(x)
    img_inpainted = to_numpy_image(avg_inpainted_result)
    img_postprocessed = to_numpy_image(inpainted_processed)

    cv2.imshow("Input (x)", img_input)
    cv2.imshow("Inpainted", img_inpainted)
    cv2.imshow("Post-processed", img_postprocessed)
    print("Press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
