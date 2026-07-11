"""
Grad-CAM for the pipeline. Two entry points:
  - gradcam_yolo(): explains *why* the detector fired on a region (which
    pixels most influenced the objectness/class score of a chosen box)
  - gradcam_unet(): explains which pixels drove the segmentation decision
    for a chosen class channel

This is the piece that turns "black box model says caries" into "here's the
exact evidence region the model is keying on" - the single most requested
feature in medical-AI interviews.

Usage:
    from explainability.gradcam import gradcam_unet
    heatmap = gradcam_unet(unet_model, image_tensor, target_class=1, target_layer=unet_model.up4)
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, score_fn) -> np.ndarray:
        """score_fn(model_output) -> scalar tensor to backprop from."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = score_fn(output)
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # global-average-pool gradients
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def gradcam_unet(model, image_tensor, target_class: int, target_layer=None):
    """image_tensor: (1, C, H, W). target_layer defaults to model.up4.conv
    (last conv block before the output head - captures the finest spatial
    detail while still being semantically meaningful)."""
    target_layer = target_layer or model.up4.conv
    cam_engine = GradCAM(model, target_layer)

    def score_fn(output):
        # mean logit for the target class channel = "how strongly does the
        # model believe this class is present, spatially"
        return output[:, target_class, :, :].mean()

    return cam_engine.generate(image_tensor, score_fn)


def gradcam_yolo(model, image, box_index: int = 0, target_layer=None):
    """Works with an ultralytics YOLO model's underlying torch module.
    target_layer should be the last conv block before detection head, e.g.
    model.model.model[-2]. Requires a raw forward pass with gradients enabled,
    which ultralytics' high-level `.predict()` doesn't expose - use the
    underlying `.model` submodule directly.
    """
    torch_model = model.model  # underlying nn.Module
    target_layer = target_layer or list(torch_model.modules())[-3]
    cam_engine = GradCAM(torch_model, target_layer)

    def score_fn(output):
        # output[0] shape ~ (batch, num_preds, 4+nc); take max objectness*cls
        # score across all predictions as the explained signal for this image
        preds = output[0] if isinstance(output, (list, tuple)) else output
        return preds[..., 4:].max()

    return cam_engine.generate(image, score_fn)


def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    heatmap_color = cv2.resize(heatmap_color, (image_bgr.shape[1], image_bgr.shape[0]))
    return cv2.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)
