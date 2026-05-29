import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from ultralytics import YOLO


def min_max_norm(tensor):
    """归一化 tensor 到 0-1 之间"""
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max - t_min < 1e-6: return tensor
    return (tensor - t_min) / (t_max - t_min)

class FEMHeatmapGenerator:
    def __init__(self, model_path, target_layer_idx=25):
        # 加载模型
        print(f"Loading model from {model_path}...")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print("如果加载失败，请确保你的环境 ultralytics 代码中已经注册了 DualIn, FEM 等模块")
            raise e
            
        self.device = self.model.device
        self.target_layer_idx = target_layer_idx
        self.fem_features = None
        self.hook_handle = None
        
        # 注册 Hook
        self._register_hook()

    def _register_hook(self):
        # 获取底层 nn.Module
        net = self.model.model
        # 按照索引获取层
        target_layer = net.model[self.target_layer_idx]
        print(f"Hooking into layer {self.target_layer_idx}: {type(target_layer)}")
        
        def hook_fn(module, input, output):
            self.fem_features = output
            
        self.hook_handle = target_layer.register_forward_hook(hook_fn)

    def run(self, rgb_path, ir_path, img_size=640):
        # 1. 读取和预处理图像
        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path)
        
        if img_rgb is None or img_ir is None:
            raise FileNotFoundError("无法找到图片，请检查路径")

        # Resize
        img_rgb_in = cv2.resize(img_rgb, (img_size, img_size))
        img_ir_in = cv2.resize(img_ir, (img_size, img_size))

        # 转 Tensor (BGR -> RGB, Normalize)
        # 注意：这里保持 opencv 的 BGR 读入，通常 yolov8 训练时内部会处理，
        # 但为了计算差异，我们需要明确的 3通道数据。
        t_rgb = torch.from_numpy(img_rgb_in).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        t_ir = torch.from_numpy(img_ir_in).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        
        # 拼接为 6 通道输入 [1, 6, 640, 640]
        input_tensor = torch.cat([t_rgb, t_ir], dim=1)

        # 2. 模型推理 (Forward)
        self.model.model.eval()
        with torch.no_grad():
            # 直接调用底层 model，绕过 predict 的预处理
            _ = self.model.model(input_tensor)

        # 3. 处理 FEM 特征
        if self.fem_features is None:
            raise RuntimeError("Hook 未捕获到特征，请检查层索引是否正确。")
        
        # 压缩通道 (Mean) -> [1, 1, H, W]
        feat_map = torch.mean(self.fem_features, dim=1, keepdim=True)
        # 上采样回原图大小
        feat_map = F.interpolate(feat_map, size=(img_size, img_size), mode='bilinear', align_corners=False)
        # 归一化特征
        feat_norm = min_max_norm(feat_map)

        # 4. 处理 RGB 基准 (转灰度强度图)
        # RGB 通道均值作为亮度信息
        rgb_intensity = torch.mean(t_rgb, dim=1, keepdim=True)
        rgb_norm = min_max_norm(rgb_intensity)

        # 5. 计算差异 (FEM 输出 - 原始 RGB)
        # 差异越大，说明 FEM 融合了 IR 或频域信息后，产生了与原始 RGB 不同的关注点
        diff = torch.abs(feat_norm - rgb_norm)
        
        # 6. 生成可视化图
        diff_np = diff.squeeze().cpu().numpy()
        diff_uint8 = (diff_np * 255).astype(np.uint8)
        
        # 热力图
        heatmap = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_JET)
        
        # 叠加图
        overlay = cv2.addWeighted(img_rgb_in, 0.5, heatmap, 0.5, 0)
        
        return img_rgb_in, img_ir_in, heatmap, overlay

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 配置路径
    model_pt = r'runs/best_pt/dronevehicle/dsam-pccl.pt'
    rgb_dir = r'/data/datasets/DroneVehicle1003/images/val/00130.jpg'
    ir_dir = r'/data/datasets/DroneVehicle1003/images_infrared/val/00130.jpg'
    
    # 检查文件是否存在 (调试用)
    import os
    if not os.path.exists(model_pt):
        print(f"Warning: 模型文件 {model_pt} 不存在，无法运行。")
    
    # 实例化并运行
    try:
        # 这里的 target_layer_idx=25 对应 YAML 中的 FEM 模块
        viz = FEMHeatmapGenerator(model_pt, target_layer_idx=25)
        
        orig_rgb, orig_ir, heatmap, overlay = viz.run(rgb_dir, ir_dir)
        
        # 保存结果
        cv2.imwrite("result_heatmap.jpg", heatmap)
        cv2.imwrite("result_overlay.jpg", overlay)
        print("处理完成！结果已保存为 result_heatmap.jpg 和 result_overlay.jpg")
        
        # 如果是在有界面的环境，可以显示
        # cv2.imshow("RGB", orig_rgb)
        # cv2.imshow("Heatmap", heatmap)
        # cv2.imshow("Overlay", overlay)
        # cv2.waitKey(0)
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()