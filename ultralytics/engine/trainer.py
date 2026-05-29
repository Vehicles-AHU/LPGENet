# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolo11n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import ultralytics.utils.globals as globals_value 

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
)


class BaseTrainer:
    """
    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolo11n -> yolo11n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            # self.get_dataset 方法被调用来获取训练集和测试集。
            self.trainset, self.testset = self.get_dataset() # 获取的是数据集的完整路径
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"

        # 绘图索引。初始化了一个绘图索引列表，用于指定在训练过程中哪些度量指标需要被绘制成图表。这里的索引0、1、2可能对应于 loss_names 中的损失名称。
        self.plot_idx = [0, 1, 2]

        # HUB
        self.hub_session = None

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)


    ################################################################
    # 训练方法调用：
    # 其中主要做 GPU 判断和 分布式判断和设置
    #   1、如果是分布式的话，通过生成的训练的 .py 临时文件和命令启动训练
    #   2、单个进程的训练走的是 engine/trainer.py 中的 def _do_train
    ################################################################
    def train(self):
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning(
                    "WARNING ⚠️ 'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.args.batch = 16

            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        # 单个进程的训练走这里
        else:
            self._do_train(world_size)

    # 学习率配置
    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf'] 余弦退火
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # 线形递减
            # 指数衰减公式：lr = lr0 * (lrf)^(epoch/epochs)
            # 最终学习率 = lr0 * lrf（当epoch=epochs时）
            # self.lf = lambda x: self.args.lrf ** (x / self.epochs)  # 指数衰减

        # 使用 PyTorch 的 LambdaLR 调度器初始化 self.scheduler 
        # self.optimizer ：优化器实例，负责更新模型参数。 
        # lr_lambda ：学习率调整函数（ self.lf ），用于根据当前轮数动态调整学习率。 
        # LambdaLR 调度器会根据 lr_lambda 函数的返回值调整优化器的学习率。
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)


    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )






    ##########################################################
    # 模型训练前的准备工作
    #   1、这段代码定义了 BaseTrainer 类中的一个私有方法 _setup_train ，
    #   2、它负责在正确的进程上构建数据加载器和优化器，为训练做准备。
    #   3、world_size：表示参与分布式训练的进程总数
    ##########################################################
    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""
        # # 
        # if world_size > 1:
        #     self.model =  nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # 1、基本处理
        self.run_callbacks("on_pretrain_routine_start")# 运行预训练开始的回调函数
        ckpt = self.setup_model()# 设置模型并返回检查点（checkpoint）信息。
        self.model = self.model.to(self.device) # 将模型移动到指定的设备（GPU或CPU）
        self.set_model_attributes()         # # 设置模型属性。


        # 2、这段代码处理模型中某些层的冻结逻辑，即设置这些层的参数在训练过程中不被更新。
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True



        # 3、这段代码处理自动混合精度（Automatic Mixed Precision, AMP）的设置，这是一种用于加速训练并减少内存使用的技術，特别是在支持CUDA的GPU上。
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks



        # 4、检查是否处于分布式训练环境中（ RANK 大于-1且 world_size 大于1）。
        if RANK > -1 and world_size > 1:  # DDP
            # 从rank 0广播AMP标志到所有其他进程，确保所有进程都知道是否启用AMP。
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        # 将AMP标志转换为布尔值。
        self.amp = bool(self.amp)  # as boolean
        # 根据AMP标志创建一个 GradScaler 实例，用于在AMP训练中调整梯度。
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )



        # 5、再次检查是否处于分布式训练环境中。
        if world_size > 1:
            # 如果是分布式训练环境，将模型包装为 DistributedDataParallel 实例，以便在多个GPU上并行训练。 
            #      device_ids=[RANK] 指定了当前进程应该使用的GPU编号， 
            #      find_unused_parameters=True DDP训练时，是否检查未使用的参数
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)





        # 6、这段代码处理图像大小的检查和批量大小的调整，以确保训练的顺利进行
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # 7、Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()
        batch_size = self.batch_size // max(world_size, 1)


         # 8、Dataloaders：准备train_loader和test_loader 
         # 1）在get_dataloader中调用了build_dataset方法，在调用build_yolo_dataset读取了数据集数据
         # 2）再进入YOLODataset类，同时初始化其父类BaseDataset
         # 3）BaseDataset中有个self.transforms = self.build_transforms(hyp=hyp)属性，但build_transforms的实现由YOLODataset进行实现的
         # 4）build_transforms中有Format，其中有判断，如果是obb类型，则labels需要从xyxyxyxy转化为xywhr（实际上是把segment的4x2数组转化为xywhr格式）
         # 5）BaseDataset每次调用__getitem__时，会调用self.transforms，同时格式化labels
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
         # 检查当前进程是否为主进程（ RANK 为-1表示单GPU环境， RANK 为0表示分布式训练的主进程）。
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            # 
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()


        # 8、Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )


        # 9、Scheduler配置
        self._setup_scheduler()


        # 10、早停设置：根据超参数patience
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")








    ######################################################################################
    #
    #
    # 模型训练具体实现的方法
    #
    #
    ######################################################################################
    def _do_train(self, world_size=1):
        #############################
        # 一、这个条件判断当前是否是多GPU训练环境（即 world_size 大于1）。
        # 完成训练前的准备工作
        # 并加载数据集，labels格式化等
        #############################
        if world_size > 1:
            self._setup_ddp(world_size) # 如果是，则调用 _setup_ddp 方法来设置分布式数据并行（DDP）环境
        #  调用 _setup_train 方法来完成训练前的准备工作，包括构建数据加载器、优化器、学习率调度器等。
        self._setup_train(world_size)




        #############################
        # 二、记录日志等相关参数
        #############################
        nb = len(self.train_loader)  # batch_size大小
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # 计算预热（warmup）迭代次数。
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        # 使用 LOGGER 记录训练的基本信息，包括图像大小、数据加载器工作进程数、结果日志保存目录以及训练周期或时间。
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        # 这个条件判断是否设置了关闭镶嵌（mosaic）数据增强的参数
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch




        ########################################################
        #
        # 三、开始一个无限循环，用于执行训练周期，直到满足某个条件（如达到最大训练周期数或满足早停条件）
        #
        ########################################################
        # 清空优化器中的梯度。这是为了确保在训练开始时，任何从先前训练周期恢复的梯度都被清零，从而保证训练的稳定性。
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            ########################################################
            # 1、每一个epoch之前的准备工作
            ########################################################
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            # 使用 warnings 模块的上下文管理器来捕获和处理警告。
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                # 更新学习率调度器。这个步骤通常在每个周期开始或结束时调用，以根据预设的策略调整学习率。
                self.scheduler.step()

            # 将模型设置为训练模式。
            self.model.train()

            # 这个条件判断是否处于分布式训练环境中。 RANK 是一个全局变量，用于标识当前进程在分布式训练中的编号。如果 RANK 不等于-1（即不是单GPU训练），则执行以下操作
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # 这个条件判断是否到达了关闭镶嵌（mosaic）数据增强的周期。 self.args.close_mosaic 是一个参数，指定在训练的最后几个周期关闭mosaic数据增强。
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            


            ########################################################
            # 2、遍历训练数据加载器中的每个batch
            ########################################################            
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # 2.1 Warmup的相关配置
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])




                ####################################################################################################################
                # 2.2 前向传播和反向传播
                # 这段代码是 BaseTrainer 类中的 _do_train 方法的一部分，它负责执行模型的前向传播和反向传播。
                # 使用PyTorch的 autocast 上下文管理器，它会自动将模型的前向传播转换为混合精度（FP16和FP32）计算。 self.amp 是 GradScaler 实例，用于自动混合精度训练。
                ####################################################################################################################
                with autocast(self.amp):
                    # 2.2.1 对批次数据进行预处理。 
                    # self.preprocess_batch 方法包括数据增强、归一化等操作，以准备输入模型的数据。
                    batch = self.preprocess_batch(batch)
                    
                    # 2.2.2 将预处理后的数据 batch 输入模型，计算损失。
                    # 模型返回 损失值 和 损失项的详细列表 。
                    if globals_value.cls_vector_flag:
                        globals_value.cls_vector_arr = []
                        globals_value.epoch = epoch
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )


                # 3、反向传播。
                # 使用 GradScaler 的 scale 方法对损失进行缩放，以防止在混合精度训练中梯度下溢。然后调用 backward() 方法进行反向传播，计算梯度。
                self.scaler.scale(self.loss).backward()


                # 4、这段代码是 BaseTrainer 类中的 _do_train 方法的一部分，
                # 它负责在训练过程中确定何时执行优化器步骤（即更新模型参数）以及根据时间条件停止训练。
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break




                # 5、日志记录和进度更新。
                # 这个条件判断当前进程是否是主进程（在单GPU或分布式训练的主节点中， RANK 为-1或0）。
                if RANK in {-1, 0}:
                    # 计算总损失 self.tloss 的维度长度，如果 self.tloss 是一个多维张量，则取第一个维度的长度；如果是标量，则长度为1。
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                     # 更新进度条描述，显示 当前epoch 、 GPU内存使用情况 、 损失值 、 批次大小 和 图像大小 。 % 操作符用于格式化字符串。
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                     # 如果设置了绘图参数 self.args.plots 为真，并且当前迭代编号 ni 在绘图索引列表 self.plot_idx 中，则执行绘图操作。
                    if self.args.plots and ni in self.plot_idx:
                         # 绘制训练样本，用于可视化训练过程中的图像或特征图。
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            # 当前batch训练完成
            # 1、学习率记录和EMA更新。
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
             # 再次检查当前进程是否是主进程。
            if RANK in {-1, 0}:
                # 判断是否是最后一个训练周期。
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # 这段代码是 BaseTrainer 类中的 _do_train 方法的一部分，
                # 它负责在训练过程中执行验证（validation）和根据验证结果以及训练时间限制来决定是否停止训练。
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    ###########################################
                    # 进行模型的验证
                    ###########################################
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                # 检查是否设置了训练时间限制
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # 这个条件判断是否需要保存模型。这可能发生在以下几种情况 ： 
                #       self.args.save 为真，表示根据配置需要定期保存模型。 
                #       final_epoch 为真，表示当前是最后一个训练周期，需要保存最终的模型状态。
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")



            # 2、学习率调度器和时间管理。
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            # 回调和内存管理。
            self._clear_memory()

            # 3、早停逻辑。
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1


        # 四、最终验证和日志记录。
        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def auto_batch(self, max_num_obj=0):
        """Get batch size by calculating memory occupation of model."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # returns batch size

    def _get_memory(self):
        """Get accelerator memory utilization in GB."""
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
        elif self.device.type == "cpu":
            memory = 0
        else:
            memory = torch.cuda.memory_reserved()
        return memory / (2**30)

    def _clear_memory(self):
        """Clear accelerator memory on different platforms."""
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        """Read results.csv into a dict using pandas."""
        import pandas as pd  # scope for faster 'import ultralytics'

        return pd.read_csv(self.csv).to_dict(orient="list")

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
        # if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
        #    (self.wdir / "last_mosaic.pt").write_bytes(serialized_ckpt)  # save mosaic checkpoint


    # 这段代码定义了一个名为 get_dataset 的方法，它用于根据指定的任务和数据源加载和验证数据集。
    # 这是 get_dataset 方法的定义，它是一个实例方法，只接受 self 参数。
    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                # 如果条件满足，调用 check_det_dataset 函数来验证和加载检测数据集
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        self.data = data
        # data中是数据集的完整路径
        return data["train"], data.get("val") or data.get("test")

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")  # header
        t = time.time() - self.train_time_start
        with open(self.csv, "a") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"  # update best.pt train_metrics from last.pt
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args
                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer


def get_activation(name):
    def hook(module, input, output):
        globals.features_output[name] = output.detach().cpu()
    return hook