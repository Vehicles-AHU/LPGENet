# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG, RANK


class OBBTrainer(yolo.detect.DetectionTrainer):
    """
        cfg : 默认的cfg配置yaml文件的参数
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a OBBTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        #########################################################
        # 实例化 OBBModel
        #########################################################
        model = OBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of OBBValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
