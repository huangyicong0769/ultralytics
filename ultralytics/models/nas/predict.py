# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class NASPredictor(BasePredictor):
    """
    Ultralytics YOLO NAS Predictor for object detection.

    This class extends the `BasePredictor` from Ultralytics engine and is responsible for post-processing the
    raw predictions generated by the YOLO NAS models. It applies operations like non-maximum suppression and
    scaling the bounding boxes to fit the original image dimensions.

    Attributes:
        args (Namespace): Namespace containing various configurations for post-processing including confidence threshold,
            IoU threshold, agnostic NMS flag, maximum detections, and class filtering options.
        model (torch.nn.Module): The YOLO NAS model used for inference.
        batch (List): Batch of inputs for processing.

    Examples:
        >>> from ultralytics import NAS
        >>> model = NAS("yolo_nas_s")
        >>> predictor = model.predictor

        Assume that raw_preds, img, orig_imgs are available
        >>> results = predictor.postprocess(raw_preds, img, orig_imgs)

    Notes:
        Typically, this class is not instantiated directly. It is used internally within the `NAS` class.
    """

    def postprocess(self, preds_in, img, orig_imgs):
        """Postprocess predictions and returns a list of Results objects."""
        # Convert boxes from xyxy to xywh format and concatenate with class scores
        boxes = ops.xyxy2xywh(preds_in[0][0])
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)

        # Apply non-maximum suppression to filter overlapping detections
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            soft=self.args.soft_nms,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            # Scale bounding boxes to match original image dimensions
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
