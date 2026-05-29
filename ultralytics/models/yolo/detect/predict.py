# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolo11n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        ################## 将结果框分别画在两个模态的图上#################
        results = []
        ir_result = []
        for i, pred in enumerate(preds):
            # orig_img = orig_imgs[i]
            # 先推理可见光
            orig_img = orig_imgs[i][..., 3:]    # 此时传入进来的im0的前三通道是红外，后三通道是可见光
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
            # 再推理红外
            ir_path = img_path.split('images')
            ir_path = str(ir_path[0] + 'images_infrared' + ir_path[1])
            if orig_imgs[i].shape[-1] >= 4:
                ir_img = orig_imgs[i][..., :3]
                ir_result.append(Results(ir_img, path=ir_path, names=self.model.names, boxes=pred))
        return results, ir_result
        ################## 将结果框分别画在两个模态的图上#################









    def construct_results(self, preds, img, orig_imgs):
        """
        Constructs a list of result objects from the predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes and scores.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.

        Returns:
            (list): List of result objects containing the original images, image paths, class names, and bounding boxes.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Constructs the result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes and scores.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.

        Returns:
            (Results): The result object containing the original image, image path, class names, and bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
