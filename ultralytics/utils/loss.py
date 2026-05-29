# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist
import ultralytics.utils.globals as globals_value 
from scipy.cluster.hierarchy import linkage, fcluster


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        
        # 1、IoU loss = 1 - iou
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # 2、DFL loss
        if self.dfl_loss:
            # 把真实框的坐标减小到对应的reg_max（16）的尺度，坐标再转化为ltrb的格式
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()





##############################################################################
# YOLOv8 矩形框损失函数
##############################################################################
class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.focal = FocalLoss()
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    # targets [32,6] 当前batch中，一共有32个目标。6：第0位：batch中图片的id，第1位：类别id，第2-5位：bbox框
    # batch_size : 4
    # scale_tensor: [640,640,640,640]
    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # 32个目标属于哪一个图像，保存图像索引
            _, counts = i.unique(return_counts=True) # 统计每张图片目标的数量：[ 6, 11,  3, 12]
            counts = counts.to(dtype=torch.int32)
            # 因为每张图片中最多有12个目标，所以设置out大小为[4,12,5]
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        # 返回：[4,12,5] 坐标转化为xyxy格式，没有目标的位置，用0占位
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)



    # preds: 0:[4,69,80,80]  1:[4,69,40,40]  2:[4,69,20,20]
    # batch：真实值
    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds # feats=preds
        # pred_distri：[4,64,8400] 
        # pred_scores：[4,5,8400] 
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        
        
        
        ######################################################
        # 一、数据预处理
        ######################################################
        # 1、预测值
        #   pred_distri：[4,8400,64] 8400个框
        #   pred_scores：[4,8400,5]  每个框对应的类别
        ######################################################
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        # imgsz:图片原尺寸 ([640,640])
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        # anchor_points [8400,2]为8400个框的中心坐标，stride_tensor[8400,1] 为框的偏移量
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # 真实数据
        # targets [32,6] 当前batch中，一共有32个目标。6：第0位：batch中图片的id，第1位：类别id，第2-5位：bbox框
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        
        ###################
        # 2、真实值
        #   gt_labels：[4,12,1]  真实目标的类别
        #   gt_bboxes:[4,12,4]   真实目标框
        #   mask_gt：[4,12,1] 掩码，表示12个目标中，哪个是有目标的，哪个是占空的
        ##################
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)



        # Pboxes
        ######################################################
        # 二、正样本匹配：self.assigner
        # 基于TaskAlignedLearning进行正负样本分配
        ######################################################
        # 把预测概率分布的坐标，转化为xyxy的坐标
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2


        # 一个anchor point 负责一个真实框
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(), # 8400个预测框对应的类别 [4,8400,5]
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), # 8400个框
            anchor_points * stride_tensor, # 锚点 * 缩放系数
            gt_labels, # 真实类别标签
            gt_bboxes, # 真实框
            mask_gt, # 掩码
        )






        ######################################################
        # 三、计算Loss
        ######################################################        
        # 1、Cls loss                
        target_scores_sum = max(target_scores.sum(), 1)
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
                
        # 2、Bbox loss 和 DFL损失
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain  7.5
        loss[1] *= self.hyp.cls  # cls gain  0.5 
        loss[2] *= self.hyp.dfl  # dfl gain  1.5

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


"""
    OBB Loss函数的实现
"""
class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        
        self.model = model
        
        # self.assigner 是一个负责为预测框分配真实标签的组件。 RotatedTaskAlignedAssigner 是一个类，用于旋转边界框的任务对齐分配。
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)

        # 这个类用于计算旋转目标检测任务中的训练损失，
        # 返回： IoU 损失  和 Distribution Focal Loss (DFL)。
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)



    # 这段代码定义了一个名为 preprocess 的方法，它用于预处理目标（即真实标签）数据，使其与输入批次大小相匹配，并输出一个处理后的张量。
    # 这是 preprocess 方法的定义，它接受三个参数。
    # 1.self ：类的实例引用。
    # 2.targets ：目标数据。
    # 3.batch_size ：批次大小。
    # 4.scale_tensor ：用于调整边界框尺寸的张量
    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out


    # 这段代码定义了一个名为 __call__ 的方法，它是 v8OBBLoss 类的一部分，用于计算旋转YOLO模型的损失。这个方法接收模型的预测结果 preds 和一批数据 batch ，然后计算包括边界框损失、类别损失和方向损失在内的总损失。
    # 1.self ：这是一个对当前类实例的引用，允许访问类的属性和方法。
    # 2.preds ：这是一个包含模型预测结果的参数。 preds 通常是一个张量或张量的列表/元组，包含了模型对输入数据的预测输出，如预测的边界框、类别概率和旋转角度等。
    # 3.batch ：这是一个包含一批数据的真实标签信息的参数。 batch 通常是一个字典或张量的列表/元组，包含了与 preds 中的预测相对应的真实标签数据，如真实边界框的位置、类别标签等。
    #       preds：2个元素
    #           0:[b,69,80,80],[b,69,40,40],[b,69,20,20] 其中69=16x4+5 ， 对应8400个预测框和其类别
    #           1:[b,1,8400] 每个预测框的旋转角度
    #       batch：
    #           cls:[929,1] 真实的batch张图片中有929个目标，1表示的是类别：其值为0-4
    #           bboxes:[929,5] 真实的32个目标对应的框xywhr
    #           batch_idx:[929]  当前的目标属于该batch的哪一个图片
    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        #######################################################
        #
        # 一、数据预处理
        #
        #######################################################
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        
        # 1、preds数据预处理：将不同维度的框cat起来
        #   pred_distri：[b,8400,16x4] 每个图片预测8400个框
        #   pred_scores：[b,8400,5]  对应的每个框的类别
        #   pred_angle： [b,8400,1]  对应每个框的偏转角度
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()


        # 2、生成8400个框的Anchor中心点：anchor_points * stride_tensor，得到的就是实际映射到640x640图像的真实框中心点
        # anchor_points：[8400,2] ，其中值为[[0.5,0.5],[0.5,1.0].........,[19.5,19.5]]
        # stride_tensor:缩放比例 [8400,1] 其中的值都是8/16/32 [[8],[8].......[16],[16]........[32],[32]]
        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # 原始图像输入的尺寸[640,640]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)


        # 3、targets预处理
        # gt_bboxes [b,94,5] ：指一张图片中最多94个框
        # gt_labels [b,94,1] ：
        # mask_gt：[b,94,1] :每张图片中94个gt_box站位，有效的为True，无效的为False
        try:
            batch_idx = batch["batch_idx"].view(-1, 1) # [929,1]
            # 3.1 targets：[b,929,7] 929个真实框，7：1+1+5（图像id+类别id+bboxs）
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            # 3.2 从 targets 中提取宽度和高度，并将其乘以图像尺寸，得到 实际的 宽度[929] 和 高度[929]
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            # 3.3 过滤掉宽度和高度小于2的边界框，以稳定训练过程。
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            # 3.4 调用 preprocess 方法对目标张量进行预处理，包括将边界框坐标归一化到 [0, 1] 范围内。
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            # 3.5 将预处理后的目标张量 targets 分割成 真实标签 gt_labels 和 真实边界框 gt_bboxes 。
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e







        #######################################################################
        #
        # 二、正样本匹配：目的是筛选更好的预测框
        # target_bboxes：[32, 8400, 5]
        # target_scores：[32, 8400, 5]
        # fg_mask：[32, 8400] # mask
        #######################################################################
        # 调用 bbox_decode 方法将 预测的分布 pred_distri 、 锚点 anchor_points 和 预测的角度 pred_angle 解码成 预测的边界框 pred_bboxes 。
        # 这些边界框以 xyxy 格式表示，即每个边界框由四个坐标值组成：x1, y1, x2, y2。
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # 转化为xyxyr, (b, 8400, 5)
        # 克隆预测的边界框 pred_bboxes 并将其从计算图中分离，以便用于分配器而不会影响梯度计算。
        bboxes_for_assigner = pred_bboxes.clone().detach()

        # 将 bboxes_for_assigner 的前四个元素（即 xyxy 格式的坐标）乘以步长张量 stride_tensor ，以将边界框坐标从 特征图尺寸 调整到 原始图像尺寸 
        bboxes_for_assigner[..., :4] *= stride_tensor
        # 调用分配器 assigner 方法为每个预测框分配真实标签和目标。
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), # 将预测分数通过 sigmoid 函数转换为概率
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor, # 将锚点调整到原始图像尺寸
            gt_labels,
            gt_bboxes,
            mask_gt,
        )


        #######################################################################
        #
        # 三、损失函数计算
        #
        #######################################################################
        # 损失函数计算
        # 这里求和有小数的原因是在正负样本分配的最后target_scores乘以了一个动态权重
        target_scores_sum = max(target_scores.sum(), 1)

        # 分类损失
        # pred_scores:[4,8400,5]  target_scores:[4,8400,5]
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox损失
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain


        ###################### cls 对比学习 ############################
        # 获取特征向量
        # class 向量：rgb、ir、fusion
        isTraining = self.model.training # train or eval
        if globals_value.cls_vector_flag and isTraining:
            # 类别特征
            cls_vector = globals_value.cls_vector_arr  
            B,C,_,_ = cls_vector[0].shape
            # 类别特征向量
            vector = torch.cat([cls_vector[i].reshape(B,C,-1) for i in range(3)],dim=2).permute(0, 2, 1).contiguous()
            # 使用 numpy.loadtxt 读取文件
            prototype_vector = np.loadtxt("vectors/dronevehicle_output_vectors.txt", delimiter=",")
            prototype_vector = torch.tensor(prototype_vector,dtype=torch.float32,device=self.device) # [5,768]
            

            # PCCL损失
            # contrast_loss = dual_contrastive_loss(
            #     vector, 
            #     prototype_vector,
            #     fg_mask,
            #     target_scores
            # )
            #prototype_loss =  globals_value.prototype_ratio * contrast_loss


            # triplet损失
            # contrast_loss = triplet_loss(
            #     vector, 
            #     prototype_vector,
            #     fg_mask,
            #     target_scores
            # )
            # prototype_loss =  globals_value.prototype_triplet_ratio * contrast_loss

            # center损失
            contrast_loss = center_loss(
                vector, 
                prototype_vector,
                fg_mask,
                target_scores
            )
            prototype_loss =  globals_value.prototype_center_ratio * contrast_loss

            loss[1] = loss[1]  + prototype_loss 
        ##################### end #############################


        

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]




# def dual_contrastive_loss(cls_vector, 
#                           prototype_vector, 
#                           fg_mask,
#                           target_scores,
#                           temperature=0.07,
#                           weights = [0.5,0.005,0.25,0.01,0.5]):
#     """
#     对比损失函数：使用softmax+交叉熵实现InfoNCE Loss
#     Args:
#         cls_vector (Tensor): 类特征向量 [B, 8400, 128]
#         prototype_vector (Tensor): 原型向量 [5,128]
#         fg_mask (BoolTensor): 正样本掩码 [B, 8400]
#         target_scores: gt labels  [B,8400,5]
#         temperature (float): 温度系数
#     Returns:
#         contrastive_loss (Tensor): 对比损失值
#     """
#     gt_labels = torch.argmax(target_scores[fg_mask],dim=1) # 8309
#     device = gt_labels.device

#     # 1、展平特征维度 [B*N, D]
#     B, N, D = cls_vector.shape
#     fg_mask = fg_mask.view(-1)
#     cls_vector = cls_vector.view(B*N, D)

#     # 2、筛选正样本特征 [num_pos, D]
#     num_pos = gt_labels.size(0)
#     cls_pos = cls_vector[fg_mask] # [8309,128]
#     prototype_pos = prototype_vector[gt_labels].detach()  # [8309, 5] 
    
#     # 3、L2特征归一化
#     cls_norm = F.normalize(cls_pos, p=2, dim=1)
#     prototype_norm = F.normalize(prototype_pos, p=2, dim=1)

#     # 4、计算相似度矩阵
#     sim_cls = torch.matmul(cls_norm, prototype_norm.T) / temperature  # [num_pos, num_pos]

#     # 5、构建标签 (对角线为正样本)
#     labels = torch.arange(num_pos, device=device)  # 8309,
#     # 根据gt_labels,给labels_one_hot加权重
#     labels_weights = [weights[gt_labels[i]] for i in range(num_pos)]
#     labels_weights = torch.tensor(labels_weights).to(device=device)
   
   
#     # 6、计算交叉熵损失
#     loss = F.cross_entropy(sim_cls, labels, labels_weights)

#     return loss



# def dual_contrastive_loss(cls_vector, 
#                           prototype_vector, 
#                           fg_mask,
#                           target_scores,
#                           temperature=0.07,
#                           weights = [10,1,10,1,10]):
#     """
#     优化后的原型对比损失函数
#     Args:
#         cls_vector (Tensor): 类特征向量 [B, 8400, D]
#         prototype_vector (Tensor): 原型向量 [num_classes, D]
#         fg_mask (BoolTensor): 正样本掩码 [B, 8400]
#         target_scores: gt labels  [B,8400,num_classes]
#         temperature (float): 温度系数
#     Returns:
#         contrastive_loss (Tensor): 对比损失值
#     """
#     # 获取类别信息
#     num_classes = prototype_vector.size(0)
    
#     # 展平特征维度 [B*N, D]
#     B, N, D = cls_vector.shape
#     flat_vector = cls_vector.reshape(B*N, D)
#     flat_target = target_scores.reshape(B*N, num_classes)
#     flat_mask = fg_mask.view(B*N)
#     device = target_scores.device

#     # 1、筛选正样本特征 [num_pos, D]
#     pos_vector = flat_vector[flat_mask]  # [num_pos, D]
#     pos_targets = flat_target[flat_mask]
#     num_pos = pos_vector.size(0)

#     if num_pos == 0:
#         return torch.tensor(0.0, device=cls_vector.device)

#     # 2、L2特征归一化
#     pos_norm = F.normalize(pos_vector, p=2, dim=1) # [3540,128]
#     proto_norm = F.normalize(prototype_vector, p=2, dim=1)  # [5, 128]

#     # 3、计算样本-原型相似度矩阵 [3540, 5]
#     sim_matrix = torch.matmul(pos_norm, proto_norm.T) / temperature
    

#     # ################## 标签平滑 #####################
#     # # 计算每个样本的置信度
#     # confidence = pos_targets.max(dim=1).values  # [num_pos]


#     # # smoothed_target = pos_targets + smooth_matrix
#     # indices = torch.arange(num_pos, device=pos_targets.device)

#     # # 获取每个样本的top1类别
#     # top1_values, top1_indices = torch.max(pos_targets, dim=1)
#     # # 计算自适应平滑量
#     # adaptive_smoothing = smoothing * (1 - confidence)  # confidence: [num_pos]
#     # reduction = top1_values * adaptive_smoothing       # [num_pos]

#     # # 初始化平滑矩阵
#     # smooth_matrix = torch.zeros_like(pos_targets)

#     # # 处理主导类别（top1位置）
#     # smooth_matrix[indices, top1_indices] = -reduction
#     # # 处理非主导类别（排除top1的位置）
#     # if num_classes > 1:
#     #     # 创建非主导类别的掩码
#     #     mask = torch.ones_like(pos_targets, dtype=torch.bool)
#     #     mask[indices, top1_indices] = False
        
#     #     # 计算每个非主导类别应分配的值
#     #     per_non_dominant = reduction / (num_classes - 1)
        
#     #     # 将平滑值分配到非主导类别
#     #     smooth_matrix[mask] = per_non_dominant.repeat_interleave(num_classes - 1)


#     # smoothed_target = pos_targets + smooth_matrix



#     # 计算对比损失
#     # log_prob = F.log_softmax(sim_matrix, dim=-1)
#     # loss = F.kl_div(log_prob, pos_targets, reduction='batchmean', log_target=False)
#     labels_weights = torch.tensor(weights).to(device=device)
#     loss = F.cross_entropy(sim_matrix,pos_targets,reduction='mean',weight=labels_weights)
    
#     return loss







    """
    优化后的原型对比损失函数
    Args:
        cls_vector (Tensor): 类特征向量 [B, 8400, D]
        prototype_vector (Tensor): 原型向量 [num_classes, D]
        fg_mask (BoolTensor): 正样本掩码 [B, 8400]
        target_scores: gt labels  [B,8400,num_classes]
        temperature (float): 温度系数
    Returns:
        contrastive_loss (Tensor): 对比损失值
    """
def dual_contrastive_loss(cls_vector, 
                          prototype_vector, 
                          fg_mask,
                          target_scores,
                          temperature=0.07,
                          weights = [10,1,1,10,10]):
    # 获取类别信息
    num_classes = prototype_vector.size(0)
    # 展平特征维度 [B*N, D]
    B, N, D = cls_vector.shape
    flat_vector = cls_vector.reshape(B*N, D)
    flat_target = target_scores.reshape(B*N, num_classes)
    flat_mask = fg_mask.view(B*N)
    device = target_scores.device

    # 1、筛选正样本特征
    # 类别特征向量pos_vector：Bx8400x128 -> num_posx128
        # 确保flat_mask是布尔类型
    if flat_mask.dtype != torch.bool:
        flat_mask = flat_mask.bool()
    pos_vector = flat_vector[flat_mask] 
    pos_targets = flat_target[flat_mask] # 类别特征向量的标签：num_pos
    num_pos = pos_vector.size(0)

    if num_pos == 0:
        return torch.tensor(0.0, device=cls_vector.device)

    # 2、L2特征归一化
    # 类别特征向量归一化pos_norm：num_posx128
    pos_norm = F.normalize(pos_vector, p=2, dim=1) # [num_pos,128]
    # 原型向量归一化proto_norm：5x128
    proto_norm = F.normalize(prototype_vector, p=2, dim=1)  # [5, 128]

    # 3、计算样本-原型相似度矩阵 [num_pos, 5]
    sim_matrix = torch.matmul(pos_norm, proto_norm.T) / temperature


    # 4、计算对比损失
    labels_weights = torch.tensor(weights).to(device=device)
    loss = F.cross_entropy(sim_matrix,pos_targets,reduction='mean',weight=labels_weights)

    # 5、权重
    pccl_weight = get_pccl_weight(globals_value.epoch) 
    loss = pccl_weight * loss

    return loss




def get_pccl_weight(epoch, total_epochs = 300, initial_weight=1.0, final_weight=0.01):
    # if epoch > total_epochs:
    #     return final_weight
    return final_weight + 0.5 * (initial_weight - final_weight) * (1 + math.cos(math.pi * epoch / total_epochs))


# 中心损失
def center_loss(cls_vector, prototype_vector, fg_mask, target_scores, temperature=0.07, weights = [10,1,10,10,10]):
    """
    固定中心损失，接口与 dual_contrastive_loss 一致。
    使用 prototype_vector 作为固定的类中心，计算正样本特征与对应中心的欧氏距离。
    """
    B, N, D = cls_vector.shape
    num_classes = prototype_vector.size(0)
    device = cls_vector.device
    
    # 展平
    flat_vector = cls_vector.reshape(B*N, D)
    flat_target = target_scores.reshape(B*N, num_classes)
    flat_mask = fg_mask.view(B*N).bool()
    
    # 筛选正样本
    pos_vector = flat_vector[flat_mask]            # [num_pos, D]
    pos_labels = flat_target[flat_mask]            # [num_pos, num_classes]
    num_pos = pos_vector.size(0)
    
    if num_pos == 0:
        return torch.tensor(0.0, device=device)
    
    # 将 one-hot 标签转换为类别索引
    pos_label_idx = pos_labels.argmax(dim=1)       # [num_pos]
    
    # 选择对应的原型向量
    selected_prototypes = prototype_vector[pos_label_idx]  # [num_pos, D]
    
    # 计算欧氏距离（平方）
    dist_sq = torch.sum((pos_vector - selected_prototypes) ** 2, dim=1)  # [num_pos]
    
    # 应用类别权重
    if weights is not None:
        weights_tensor = torch.tensor(weights, device=device)
        sample_weights = weights_tensor[pos_label_idx]  # [num_pos]
        loss = torch.mean(dist_sq * sample_weights)
    else:
        loss = torch.mean(dist_sq)

    # 权重
    pccl_weight = get_pccl_weight(globals_value.epoch) 
    loss = pccl_weight * loss
    
    return loss



# 三元组损失
def triplet_loss(cls_vector, prototype_vector, fg_mask, target_scores, temperature=0.07, weights=None, margin=0.2):
    """
    批量难样本三元组损失，接口与 dual_contrastive_loss 一致。
    """
    B, N, D = cls_vector.shape
    device = cls_vector.device

    # 展平
    flat_vector = cls_vector.reshape(B*N, D)
    flat_target = target_scores.reshape(B*N, -1)
    flat_mask = fg_mask.view(B*N).bool()

    # 筛选正样本
    pos_vector = flat_vector[flat_mask]                     # [num_pos, D]
    pos_labels = flat_target[flat_mask]                     # [num_pos, num_classes]
    num_pos = pos_vector.size(0)

    if num_pos < 2:
        return torch.tensor(0.0, device=device)

    # 归一化（可选，但通常三元组损失用欧氏距离时不强制归一化）
    # pos_vector = F.normalize(pos_vector, p=2, dim=1)

    pos_label_idx = pos_labels.argmax(dim=1)                 # [num_pos]

    # 1. 计算所有正样本两两之间的欧氏距离矩阵 [num_pos, num_pos]
    dist_mat = torch.cdist(pos_vector, pos_vector, p=2) ** 2  # 平方欧氏距离

    # 2. 创建同类掩码和异类掩码
    same_class_mask = (pos_label_idx.unsqueeze(1) == pos_label_idx.unsqueeze(0))  # [num_pos, num_pos]
    # 排除自身（对角线）
    same_class_mask = same_class_mask.float()
    same_class_mask.fill_diagonal_(0)

    diff_class_mask = (pos_label_idx.unsqueeze(1) != pos_label_idx.unsqueeze(0)).float()  # [num_pos, num_pos]

    # 3. 对于每个锚点，获取最难正样本距离（同类中最大距离）和最硬负样本距离（异类中最小距离）
    # 将同类中无效位置（无同类样本）的距离设为 -inf，以便后续 max 忽略
    same_dist = dist_mat * same_class_mask
    # 将0值（排除自身后为0）替换为一个很小的负数，避免影响 max（但这里我们直接用 max，0不会成为最大值）
    # 更安全：将 same_class_mask 为0的位置设为 -inf
    same_dist = torch.where(same_class_mask.bool(), same_dist, torch.tensor(-float('inf'), device=device))
    hardest_pos_dist, _ = torch.max(same_dist, dim=1)       # [num_pos]

    # 对于异类，将无效位置（无异类样本）的距离设为 inf，以便后续 min 忽略
    diff_dist = dist_mat * diff_class_mask
    diff_dist = torch.where(diff_class_mask.bool(), diff_dist, torch.tensor(float('inf'), device=device))
    hardest_neg_dist, _ = torch.min(diff_dist, dim=1)       # [num_pos]

    # 4. 计算三元组损失
    loss_per_anchor = torch.clamp(hardest_pos_dist - hardest_neg_dist + margin, min=0.0)
    
    # 5. 处理可能无效的锚点（例如某锚点没有同类样本或没有异类样本）
    valid_mask = (hardest_pos_dist != -float('inf')) & (hardest_neg_dist != float('inf'))
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)
    
    loss = loss_per_anchor[valid_mask].mean()

    # 6. 应用类别权重（可选，此处可映射到每个锚点）
    if weights is not None:
        weights_tensor = torch.tensor(weights, device=device)
        anchor_weights = weights_tensor[pos_label_idx]      # [num_pos]
        loss = (loss_per_anchor * anchor_weights)[valid_mask].mean()

    # 7. 余弦衰减权重
    pccl_weight = get_pccl_weight(globals_value.epoch) 
    loss = pccl_weight * loss

    return loss