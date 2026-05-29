# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from ultralytics.utils import ROOT, yaml_load

# 这段代码定义了一个名为 YOLO 的类，它继承自 Model 类，并通过动态加载和初始化不同的子模块来支持多种任务（如分类、检测、分割等）。
# 定义了一个名为 YOLO 的类，继承自 Model 类。 Model 是父类，包含一些通用的模型初始化和管理功能。
class YOLO(Model):

    # 定义了 YOLO 类的初始化方法 __init__ ，它接受以下参数 ：
    #   model ：默认值为 "yolo11n.pt" ，表示模型文件的路径或名称。
    #   task ：表示任务类型（如分类、检测等），默认为 None 。
    #   verbose ：布尔值，表示是否输出详细信息，默认为 False 。
    def __init__(self, model="yolo11n.pt", task=None, verbose=False):

        # 使用 Path 类（可能来自 pathlib 模块）将 model 参数转换为路径对象，方便后续对文件名和扩展名的操作
        # model：来自于训练输入的yaml文件，这里是ultralytics/cfg/models/v8/yolov8.yaml
        path = Path(model)

        # 如果是 -world模型
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        # 其他模型
        else:
            ######################################################
            # 调用父类 Model 的初始化方法，将 model 、 task 和 verbose 参数传递给父类。
            # 完成父类Model初始化后，YOLO对象初始化完成，开始调用YOLO的train方法
            ######################################################
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    """YOLO-World object detection model."""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes
