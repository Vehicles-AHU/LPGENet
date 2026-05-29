# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Check a model's accuracy on a test or val split of a dataset.

Usage:
    $ yolo mode=val model=yolo11n.pt data=coco8.yaml imgsz=640

Usage - formats:
    $ yolo mode=val model=yolo11n.pt                 # PyTorch
                          yolo11n.torchscript        # TorchScript
                          yolo11n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                          yolo11n_openvino_model     # OpenVINO
                          yolo11n.engine             # TensorRT
                          yolo11n.mlpackage          # CoreML (macOS-only)
                          yolo11n_saved_model        # TensorFlow SavedModel
                          yolo11n.pb                 # TensorFlow GraphDef
                          yolo11n.tflite             # TensorFlow Lite
                          yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolo11n_paddle_model       # PaddlePaddle
                          yolo11n.mnn                # MNN
                          yolo11n_ncnn_model         # NCNN
                          yolo11n_imx_model          # Sony IMX
                          yolo11n_rknn_model         # Rockchip RKNN
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode

class BaseValidator:
    """
    BaseValidator.

    A base class for creating validators.

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        names (dict): Class names.
        seen: Records the number of images seen so far during validation.
        stats: Placeholder for statistics during validation.
        confusion_matrix: Placeholder for a confusion matrix.
        nc: Number of classes.
        iouv: (torch.Tensor): IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (dict): Dictionary to store JSON validation results.
        speed (dict): Dictionary with keys 'preprocess', 'inference', 'loss', 'postprocess' and their respective
                      batch processing times in milliseconds.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
        callbacks (dict): Dictionary to store various callback functions.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
            _callbacks (dict): Dictionary to store various callback functions.
        """
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()



    #######################################################
    #
    # 模型验证
    #
    #######################################################
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Executes validation process, running inference on dataloader and computing performance metrics."""
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            ######################模型预热#####################
            #model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup
            model.warmup(imgsz=(1 if pt else self.args.batch, self.args.ch, imgsz, imgsz))  # warmup
            #######################模型预热#####################
        
        
        # 这段代码继续描述了 BaseValidator 类的 __call__ 方法中的操作，涉及到回调函数的执行、性能分析工具的初始化、进度条的设置、度量指标的初始化以及准备用于存储结果的数据结构。
        self.run_callbacks("on_val_start")
        
        # 一、创建了四个 Profile 对象的元组。
        # 每个 Profile 对象都用于性能分析，监控不同阶段（如预处理、推理、损失计算和后处理）的时间消耗。 device=self.device 参数指定了性能分析应该在哪个设备上进行，通常是模型运行的设备。
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        # 初始化一个空列表 self.jdict 。这个列表用于存储验证过程中的中间结果或最终结果，以便后续处理或保存
        self.jdict = []  # empty before each val

        # 二、开始一个循环，遍历 bar （一个  TQDM  进度条对象，包装了 self.dataloader ）。 
        # batch_i 是批次索引， batch 是当前批次的数据。
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # 1、使用 Profile 对象 dt[0] 来记录预处理阶段的时间消耗
            with dt[0]:
                # 调用 self.preprocess 方法对当前批次的数据进行预处理，并将处理后的数据重新赋值给 batch 。
                batch = self.preprocess(batch)

            # 2、使用 Profile 对象 dt[1] 来记录推理阶段的时间消耗
            with dt[1]:
                # 调用模型的推理方法，传入预处理后的图像数据 batch["img"] 和是否进行数据增强的标志 augment ，得到预测结果 preds 。
                preds = model(batch["img"], augment=augment)
                
                # # 从预测结果中提取类别概率分布
                # if hasattr(preds, 'boxes'):
                #     boxes_data = preds.boxes.data
                #     if boxes_data is not None and boxes_data.shape[1] > 6:
                #         # 假设前6列为: x1, y1, x2, y2, conf, class_id
                #         # 第7列开始是类别概率
                #         class_probabilities = boxes_data[:, 6:]  # 形状: [num_detections, num_classes]
                        
                #         # 对类别概率进行softmax（如果模型输出的是logits）
                #         # class_probabilities = torch.softmax(class_probabilities, dim=1)
                        
                #         # 将类别概率保存到preds对象中
                #         preds_with_probs.class_probs = class_probabilities
                #     else:
                #         preds_with_probs.class_probs = None
                
                # preds = preds_with_probs


            # 3、使用 Profile 对象 dt[2] 来记录损失计算阶段的时间消耗
            with dt[2]:
                # 如果当前处于训练模式（即 self.training 为 True ），则执行以下代码。
                if self.training:
                    # # 调用模型的损失计算方法，传入批次数据和预测结果，并将计算得到的损失累加到 self.loss 。
                    self.loss += model.loss(batch, preds)[1]

            # 4、使用 Profile 对象 dt[3] 来记录后处理阶段的时间消耗
            with dt[3]:
                # 调用 self.postprocess 方法对预测结果进行后处理，并将处理后的结果重新赋值给 preds 。
                preds = self.postprocess(preds)

            # 调用 self.update_metrics 方法，根据预测结果和批次数据更新度量指标
            self.update_metrics(preds, batch)

            # 如果配置了绘图（ self.args.plots 为 True ）且当前批次索引小于3，则执行以下代码
            if self.args.plots: # if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)
            # 在处理完每个批次之后，调用 run_callbacks 方法，触发所有注册为 "on_val_batch_end" 事件的回调函数。
            self.run_callbacks("on_val_batch_end")


        # 这个循环是验证过程中的核心，它确保每个批次的数据都被正确地处理，并且相关的性能指标被更新。通过回调函数，可以在每个批次的开始和结束时执行额外的操作，增加了代码的灵活性和扩展性。
        # 这段代码是 BaseValidator 类的 __call__ 方法中的最后部分，它处理验证过程结束后的一系列操作，包括获取统计数据、检查统计数据、计算速度、最终确定度量指标、打印结果以及触发验证结束的回调函数。
        # 调用 self.get_stats 方法来获取当前的统计数据，这些数据包括准确率、损失值等度量指标
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            # 打印fps
            LOGGER.info(f'FPS:{(1000 / sum(self.speed.values())):.2f}')
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Runs all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    def build_dataset(self, img_path):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        return batch

    def postprocess(self, preds):
        """Preprocesses the predictions."""
        return preds

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        pass

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes and returns all metrics."""
        pass

    def get_stats(self):
        """Returns statistics about the model's performance."""
        return {}

    def check_stats(self, stats):
        """Checks statistics."""
        pass

    def print_results(self):
        """Prints the results of the model's predictions."""
        pass

    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Returns the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni):
        """Plots validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plots YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass
