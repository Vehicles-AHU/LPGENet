import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from ultralytics import YOLO

# ==========================================
# 工具函数
# ==========================================
def min_max_norm(tensor):
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max - t_min < 1e-6: return tensor
    return (tensor - t_min) / (t_max - t_min)

class DualModelVisualizer:
    def __init__(self, baseline_path, best_path, target_layer_idx=25):
        # 显式指定设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"当前使用设备: {self.device}")
        
        print(f"正在加载 Baseline 模型: {baseline_path}")
        self.model_base = YOLO(baseline_path)
        # 强制将底层模型移至 GPU/CPU
        self.model_base.to(self.device) 
        
        print(f"正在加载 Best 模型: {best_path}")
        self.model_best = YOLO(best_path)
        # 强制将底层模型移至 GPU/CPU
        self.model_best.to(self.device)
        
        self.target_layer_idx = target_layer_idx
        self.features = {"base": None, "best": None}
        
        # 注册 Hooks
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(name):
            def hook_fn(module, input, output):
                self.features[name] = output
            return hook_fn

        self.model_base.model.model[self.target_layer_idx].register_forward_hook(get_hook("base"))
        self.model_best.model.model[self.target_layer_idx].register_forward_hook(get_hook("best"))

    def get_heatmap(self, model_type, input_tensor, img_size, ref_norm):
        """通用特征提取与差异计算逻辑"""
        model = self.model_base if model_type == "base" else self.model_best
        model.model.eval()
        with torch.no_grad():
            _ = model.model(input_tensor)
        
        feat = self.features[model_type]
        # 压缩通道并上采样
        feat_map = torch.mean(feat, dim=1, keepdim=True)
        feat_map = F.interpolate(feat_map, size=(img_size, img_size), mode='bilinear', align_corners=False)
        feat_norm = min_max_norm(feat_map)
        
        # 计算差异热力图
        diff = torch.abs(feat_norm - ref_norm)
        diff_np = np.clip(diff.squeeze().cpu().numpy() * 1.2, 0, 1)
        heatmap = cv2.applyColorMap((diff_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return heatmap

    def process_and_save(self, rgb_p, ir_p, save_path, img_size=640):
        # 1. 图像准备
        img_rgb = cv2.imread(rgb_p)
        img_ir = cv2.imread(ir_p)
        if img_rgb is None or img_ir is None: return
        
        img_rgb_in = cv2.resize(img_rgb, (img_size, img_size))
        img_ir_in = cv2.resize(img_ir, (img_size, img_size))
        # 准备用于显示的彩色IR
        img_ir_display = cv2.cvtColor(img_ir_in, cv2.COLOR_GRAY2BGR) if len(img_ir_in.shape)==2 else img_ir_in

        # 转 Tensor [1, 6, H, W]
        t_rgb = torch.from_numpy(img_rgb_in).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        t_ir = torch.from_numpy(img_ir_in).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        input_6ch = torch.cat([t_rgb, t_ir], dim=1)

        # 归一化基准
        norm_rgb = min_max_norm(torch.mean(t_rgb, dim=1, keepdim=True))
        norm_ir = min_max_norm(torch.mean(t_ir, dim=1, keepdim=True))

        # 2. 生成四张叠加图
        # 第一行：叠加在 RGB 上
        hm_base_rgb = self.get_heatmap("base", input_6ch, img_size, norm_rgb)
        hm_best_rgb = self.get_heatmap("best", input_6ch, img_size, norm_rgb)
        over_base_rgb = cv2.addWeighted(img_rgb_in, 0.6, hm_base_rgb, 0.4, 0)
        over_best_rgb = cv2.addWeighted(img_rgb_in, 0.6, hm_best_rgb, 0.4, 0)

        # 第二行：叠加在 IR 上
        hm_base_ir = self.get_heatmap("base", input_6ch, img_size, norm_ir)
        hm_best_ir = self.get_heatmap("best", input_6ch, img_size, norm_ir)
        over_base_ir = cv2.addWeighted(img_ir_display, 0.6, hm_base_ir, 0.4, 0)
        over_best_ir = cv2.addWeighted(img_ir_display, 0.6, hm_best_ir, 0.4, 0)

        # 3. 2x3 布局绘图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 第一行
        axes[0, 0].imshow(cv2.cvtColor(img_rgb_in, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original RGB", fontsize=14)
        axes[0, 1].imshow(cv2.cvtColor(over_base_rgb, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Baseline @ RGB", fontsize=14)
        axes[0, 2].imshow(cv2.cvtColor(over_best_rgb, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Best @ RGB", fontsize=14)

        # 第二行
        axes[1, 0].imshow(cv2.cvtColor(img_ir_display, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Original IR", fontsize=14)
        axes[1, 1].imshow(cv2.cvtColor(over_base_ir, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Baseline @ IR", fontsize=14)
        axes[1, 2].imshow(cv2.cvtColor(over_best_ir, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("Best @ IR", fontsize=14)

        for ax in axes.ravel(): ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)

# ==========================================
# 执行主逻辑
# ==========================================
if __name__ == "__main__":
    baseline_pt = r'runs/best_pt/dronevehicle/baseline.pt' # 替换为你的baseline路径
    best_pt = r'runs/best_pt/dronevehicle/dsam-pccl.pt'
    rgb_dir = r'/data/datasets/DroneVehicle1003/images/val'
    ir_dir = r'/data/datasets/DroneVehicle1003/images_infrared/val'
    save_dir = r'runs/comparison_results'

    if not os.path.exists(save_dir): os.makedirs(save_dir)

    viz = DualModelVisualizer(baseline_pt, best_pt)
    
    rgb_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg', '.png'))]
    
    for filename in tqdm(rgb_files, desc="Comparing Models"):
        rgb_path = os.path.join(rgb_dir, filename)
        ir_path = os.path.join(ir_dir, filename)
        
        if os.path.exists(ir_path):
            save_path = os.path.join(save_dir, f"cmp_{filename}")
            viz.process_and_save(rgb_path, ir_path, save_path)