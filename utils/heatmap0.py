# 多模态旋转边框热力图 - 双模型比较版本
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys
import numpy as np
from tqdm import trange

np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywhr2xyxyxyxy, non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        # OBB格式: [x, y, w, h, angle, class_scores...]
        logits_ = result[:, 5:]  # 前5个是旋转框参数(x, y, w, h, angle)，后面是类别分数
        boxes_ = result[:, :5]   # 旋转框参数
        
        # 按置信度排序
        sorted_scores, indices = torch.sort(logits_.max(1)[0], descending=True)
        
        # 返回排序后的结果
        sorted_logits = torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]]
        sorted_boxes = torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]
        
        # 将旋转框转换为四个角点格式用于可视化
        obb_boxes = xywhr2xyxyxyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()
        
        return sorted_logits, sorted_boxes, obb_boxes

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()


class yolov8_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                # OBB有5个参数: x, y, w, h, angle
                for j in range(5):
                    result.append(pre_post_boxes[i, j])
        return sum(result)


class DualModelYOLOv8Heatmap:
    def __init__(self, baseline_weight, best_weight, device, method, layer, backward_type, conf_threshold, ratio, renormalize):
        self.device = torch.device(device)
        
        # 加载两个模型
        self.baseline_model = self.load_model(baseline_weight)
        self.best_model = self.load_model(best_weight)
        
        # 设置目标层
        target_layers_baseline = [self.baseline_model.model[l] for l in layer]
        target_layers_best = [self.best_model.model[l] for l in layer]
        
        # 创建Grad-CAM方法
        self.baseline_method = eval(method)(self.baseline_model, target_layers_baseline)
        self.best_method = eval(method)(self.best_model, target_layers_best)
        
        # 设置activations和gradients
        self.baseline_method.activations_and_grads = ActivationsAndGradients(self.baseline_model, target_layers_baseline, None)
        self.best_method.activations_and_grads = ActivationsAndGradients(self.best_model, target_layers_best, None)
        
        # 创建目标
        self.target = yolov8_target(backward_type, conf_threshold, ratio)
        
        self.conf_threshold = conf_threshold
        self.ratio = ratio
        self.renormalize = renormalize

    def load_model(self, weight):
        """加载模型"""
        model = attempt_load_weights(weight, self.device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        return model

    def post_process(self, result):
        # 使用OBB的非极大值抑制
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    def normalize_ir_image(self, img):
        """专门处理IR图像的归一化，解决显示全黑的问题"""
        # 如果图像是全黑的（所有像素值为0），直接返回
        if img.max() == 0:
            return img
            
        # 对于16位图像，转换为8位
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        
        # 如果图像仍然几乎全黑，尝试对比度拉伸
        if img.max() - img.min() < 50:  # 动态范围很小
            # 对比度拉伸
            if img.max() > img.min():
                img_normalized = (img - img.min()) * 255 / (img.max() - img.min())
                img_normalized = img_normalized.astype(np.uint8)
            else:
                img_normalized = img
            return img_normalized
        
        return img

    def load_and_preprocess_ir_image(self, img_path):
        """专门处理IR图像，解决显示全黑的问题"""
        # 尝试以不同方式读取图像
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # 尝试用普通方式读取
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to read IR image at path: {img_path}")
        
        # 处理单通道IR图像
        if len(img.shape) == 2:  # 单通道
            img = self.normalize_ir_image(img)
            # 转换为伪彩色以便可视化
            img_colored = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            return img_colored
        elif len(img.shape) == 3:  # 3通道
            # 如果是3通道但显示全黑，可能每个通道都需要单独处理
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            # 检查每个通道的动态范围
            for i in range(3):
                channel = img[:, :, i]
                if channel.max() - channel.min() < 10:  # 通道动态范围很小
                    if channel.max() > channel.min():
                        img[:, :, i] = (channel - channel.min()) * 255 / (channel.max() - channel.min())
                    else:
                        img[:, :, i] = channel
            
            return img
        else:
            raise ValueError(f"Unexpected IR image shape: {img.shape}")

    def load_and_preprocess_rgb_image(self, img_path):
        """处理RGB图像"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read RGB image at path: {img_path}")
        return img

    def generate_heatmap_for_model(self, model_method, img_rgb, img_ir, modality_type):
        """为指定模型生成热力图"""
        # 预处理图像
        img_rgb_processed = letterbox(img_rgb)[0]
        img_rgb_processed = cv2.cvtColor(img_rgb_processed, cv2.COLOR_BGR2RGB)
        img_rgb_processed = np.float32(img_rgb_processed) / 255.0

        img_ir_processed = letterbox(img_ir)[0]
        img_ir_processed = cv2.cvtColor(img_ir_processed, cv2.COLOR_BGR2RGB)
        img_ir_processed = np.float32(img_ir_processed) / 255.0

        # 根据模态类型调整输入
        if modality_type == 'rgb':
            # RGB模态：使用RGB图像，IR通道设为0
            img_final_rgb = img_rgb_processed
            img_final_ir = np.zeros_like(img_ir_processed)
        elif modality_type == 'ir':
            # IR模态：使用IR图像，RGB通道设为0
            img_final_rgb = np.zeros_like(img_rgb_processed)
            img_final_ir = img_ir_processed

        # 合并RGB和IR图像
        img_combined = np.concatenate((img_final_rgb, img_final_ir), axis=2)
        tensor = torch.from_numpy(np.transpose(img_combined, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        try:
            grayscale_cam = model_method(tensor, [self.target])
        except AttributeError as e:
            print(f"Attribute error in Grad-CAM method: {e}")
            return None

        grayscale_cam = grayscale_cam[0, :]
        
        # 返回对应的基础图像用于可视化
        if modality_type == 'rgb':
            base_img = img_rgb_processed
        else:  # ir
            base_img = img_ir_processed
            
        return grayscale_cam, base_img

    def process_image_pair(self, rgb_path, ir_path, save_dir):
        """处理一对RGB和IR图像"""
        try:
            # 读取RGB和IR图像
            img_rgb_orig = self.load_and_preprocess_rgb_image(rgb_path)
            img_ir_orig = self.load_and_preprocess_ir_image(ir_path)

            # 为两个模型分别生成RGB和IR的热力图
            results = {}
            
            # Baseline模型
            baseline_rgb_cam, rgb_base = self.generate_heatmap_for_model(
                self.baseline_method, img_rgb_orig.copy(), img_ir_orig.copy(), 'rgb'
            )
            baseline_ir_cam, ir_base = self.generate_heatmap_for_model(
                self.baseline_method, img_rgb_orig.copy(), img_ir_orig.copy(), 'ir'
            )
            
            # Best模型
            best_rgb_cam, _ = self.generate_heatmap_for_model(
                self.best_method, img_rgb_orig.copy(), img_ir_orig.copy(), 'rgb'
            )
            best_ir_cam, _ = self.generate_heatmap_for_model(
                self.best_method, img_rgb_orig.copy(), img_ir_orig.copy(), 'ir'
            )
            
            if all(cam is not None for cam in [baseline_rgb_cam, baseline_ir_cam, best_rgb_cam, best_ir_cam]):
                results = {
                    'rgb_base': rgb_base,
                    'ir_base': ir_base,
                    'baseline_rgb': baseline_rgb_cam,
                    'baseline_ir': baseline_ir_cam,
                    'best_rgb': best_rgb_cam,
                    'best_ir': best_ir_cam
                }

            # 生成综合对比图
            base_name = os.path.splitext(os.path.basename(rgb_path))[0]
            
            if results:
                self.save_comparison_figure(results, base_name, save_dir)
                return True
            else:
                print(f"Failed to generate heatmaps for {base_name}")
                return False
                
        except Exception as e:
            print(f"Error processing image pair {rgb_path}: {e}")
            return False

    def save_comparison_figure(self, results, base_name, save_dir):
        """生成并保存包含两个模型对比的热力图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Dual Model Comparison - {base_name}', fontsize=16, fontweight='bold')
        
        # 第一行：RGB图像
        # 原图
        axes[0, 0].imshow(cv2.cvtColor((results['rgb_base'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        axes[0, 0].set_title('RGB\nOriginal Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Baseline模型RGB热力图
        baseline_rgb_cam_image = show_cam_on_image(results['rgb_base'], results['baseline_rgb'], use_rgb=True)
        axes[0, 1].imshow(baseline_rgb_cam_image)
        axes[0, 1].set_title('RGB\nBaseline Heatmap', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Best模型RGB热力图
        best_rgb_cam_image = show_cam_on_image(results['rgb_base'], results['best_rgb'], use_rgb=True)
        axes[0, 2].imshow(best_rgb_cam_image)
        axes[0, 2].set_title('RGB\nBest Model Heatmap', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # 第二行：IR图像
        # 原图
        axes[1, 0].imshow(cv2.cvtColor((results['ir_base'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        axes[1, 0].set_title('IR\nOriginal Image', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Baseline模型IR热力图
        baseline_ir_cam_image = show_cam_on_image(results['ir_base'], results['baseline_ir'], use_rgb=True)
        axes[1, 1].imshow(baseline_ir_cam_image)
        axes[1, 1].set_title('IR\nBaseline Heatmap', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Best模型IR热力图
        best_ir_cam_image = show_cam_on_image(results['ir_base'], results['best_ir'], use_rgb=True)
        axes[1, 2].imshow(best_ir_cam_image)
        axes[1, 2].set_title('IR\nBest Model Heatmap', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(save_dir, f'{base_name}_dual_model_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved dual model comparison figure to: {comparison_path}")

    def process_dataset(self, rgb_dir, ir_dir, save_dir, max_images=None):
        """批量处理数据集中的所有图像对"""
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取RGB目录中的所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        rgb_images = []
        for ext in image_extensions:
            rgb_images.extend([f for f in os.listdir(rgb_dir) if f.lower().endswith(ext)])
        
        print(f"Found {len(rgb_images)} RGB images in {rgb_dir}")
        
        # 限制处理的图像数量（如果指定了max_images）
        if max_images is not None:
            rgb_images = rgb_images[:max_images]
            print(f"Processing first {max_images} images")
        
        # 处理每个图像对
        successful = 0
        failed = 0
        
        for rgb_file in tqdm(rgb_images, desc="Processing images"):
            rgb_path = os.path.join(rgb_dir, rgb_file)
            
            # 查找对应的IR图像
            ir_file = rgb_file  # 假设文件名相同
            ir_path = os.path.join(ir_dir, ir_file)
            
            # 如果IR图像不存在，尝试不同的扩展名
            if not os.path.exists(ir_path):
                name_without_ext = os.path.splitext(rgb_file)[0]
                found = False
                for ext in image_extensions:
                    potential_path = os.path.join(ir_dir, name_without_ext + ext)
                    if os.path.exists(potential_path):
                        ir_path = potential_path
                        found = True
                        break
                
                if not found:
                    print(f"IR image not found for {rgb_file}")
                    failed += 1
                    continue
            
            # 处理图像对
            if self.process_image_pair(rgb_path, ir_path, save_dir):
                successful += 1
            else:
                failed += 1
        
        print(f"Processing completed: {successful} successful, {failed} failed")

    def __call__(self, rgb_path, ir_path, save_path, max_images=None):
        """主调用函数，支持单个图像对或整个数据集"""
        # 如果输入是目录，则处理整个数据集
        if os.path.isdir(rgb_path) and os.path.isdir(ir_path):
            self.process_dataset(rgb_path, ir_path, save_path, max_images)
        else:
            # 处理单个图像对
            if os.path.isfile(rgb_path) and os.path.isfile(ir_path):
                self.process_image_pair(rgb_path, ir_path, save_path)
            else:
                print("Error: Both RGB and IR paths must be either files or directories")


def get_dual_params():
    params = {
        'baseline_weight': r'runs/best_pt/dronevehicle/baseline.pt',  # 请替换为实际的baseline模型路径
        'best_weight': r'runs/best_pt/dronevehicle/dsam-pccl.pt',          # 请替换为实际的best模型路径
        'device': 'cuda:0',
        'method': 'GradCAM',  # 或GradCAMPlusPlus等
        'layer': [28,31,32],  # 中间层索引
        'backward_type': 'all',  # 类别或边界框权重
        'conf_threshold': 0.1,  # 置信度阈值    
        'ratio': 0.05,  # 用于计算的目标比例
        'renormalize': True  # 是否在边界框内归一化热力图
    }
    return params


if __name__ == '__main__':
    # 使用双模型比较
    dual_model = DualModelYOLOv8Heatmap(**get_dual_params())
    
    # 处理整个数据集
    rgb_dir = '/data/datasets/DroneVehicle1003/images/val/00002.jpg'
    ir_dir = '/data/datasets/DroneVehicle1003/images_infrared/val/00002.jpg'
    save_dir = './dual_model_heatmap_results'
    
    # 可选：限制处理的图像数量（用于测试）
    max_images = None  # 设置为None处理所有图像，或设置为数字如100处理前100张
    
    dual_model(rgb_dir, ir_dir, save_dir, max_images)