# 多模态水平边框热力图
import warnings
 
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys
import numpy as np
 
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
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
        logits_ = result[:, 5:] # 类别分数
        boxes_ = result[:, :5] # 旋转框
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], \
                torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], \
                xywhr2xyxyxyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()
 
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
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)
 
 
class yolov8_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, renormalize):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
 
        target = yolov8_target(backward_type, conf_threshold, ratio)
 
        target_layers = [model.model[l] for l in layer]
 
        method = eval(method)(model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)
 
        # colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int64)
        colors = [(0, 0, 255) for i in range(len(model_names))]
        self.__dict__.update(locals())
 
    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result
 
    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1]
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(grayscale_cam.copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized
 
    def process(self, img_path, imgir_path, save_path):
        # img process
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image at path: {img_path}")
        img_rgb = img
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
 
        # Load IR image
        imgir = cv2.imread(imgir_path)
        if imgir is None:
            raise ValueError(f"Failed to read IR image at path: {imgir_path}")
        img_ir = imgir
        imgir = letterbox(imgir)[0]
        imgir = cv2.cvtColor(imgir, cv2.COLOR_BGR2RGB)
        imgir = np.float32(imgir) / 255.0
 
        # 合并RGB和IR图像 - 假设模型接受6通道输入
        img = np.concatenate((img, imgir), axis=2)
 
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
 
        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            print(f"Attribute error in Grad-CAM method: {e}")
            return
 
        grayscale_cam = grayscale_cam[0, :]
 
        # Use RGB part for visualization
        cam_image = show_cam_on_image(img[..., 3:], grayscale_cam, use_rgb=True)
 
        # 直接保存热力图图像
        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)
 
    def __call__(self, img_path, imgir_path, save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)
 
        if os.path.isdir(img_path):
            for img_file in os.listdir(img_path):
                img_file_path = os.path.join(img_path, img_file)
                imgir_file_path = os.path.join(imgir_path, img_file)
                if not os.path.isfile(imgir_file_path):
                    imgir_file_path = imgir_file_path.replace('.jpg', '.png')  # 尝试不同的扩展名
                save_file_path = os.path.join(save_path, img_file)
 
                try:
                    self.process(img_file_path, imgir_file_path, save_file_path)
                except Exception as e:
                    print(f"Error processing {img_file_path}: {e}")
        else:
            self.process(img_path, imgir_path, f'{save_path}/result.png')
 
 
def get_params():
    params = {
        'weight': r'runs/best_pt/dronevehicle/dsam-pccl.pt',
        'device': 'cuda:0',
        'method': 'LayerCAM',  # 或GradCAMPlusPlus等
        'layer': [31,37],  # 中间层索引
        'backward_type': 'all',  # 类别或边界框权重
        'conf_threshold': 0.2,  # 置信度阈值
        'ratio': 0.2,  # 用于计算的目标比例
        'renormalize': True  # 是否在边界框内归一化热力图
    }
    return params
 
 
if __name__ == '__main__':
    model = yolov8_heatmap(**get_params())
    model(
        '/data/datasets/DroneVehicle/images/val/00130.jpg',
       '/data/datasets/DroneVehicl/images_infrared/val/00130.jpg',
        './heat-result'
    )