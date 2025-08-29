import io
import os
from typing import Tuple, Dict

import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
import pydicom
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian
import cv2

# Labels démo
CLASSES = ["normal", "pneumonia", "covid19", "pulmonary_embolism"]

# ---------- Détection/lecture DICOM ----------
def _has_dicom_preamble(data: bytes) -> bool:
    # "DICM" à l'offset 128
    return len(data) >= 132 and data[128:132] == b"DICM"

def _looks_like_dicom(data: bytes) -> bool:
    try:
        ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True, force=False)
        fm = getattr(ds, "file_meta", None)
        ts = getattr(fm, "TransferSyntaxUID", None) if fm else None
        return ("PixelData" in ds) and (ts is not None)
    except Exception:
        return False

def dicom_to_pil(data: bytes, default_wc: float = -600.0, default_ww: float = 1500.0) -> Image.Image:
    ds = pydicom.dcmread(io.BytesIO(data), force=True)

    # Si TransferSyntaxUID manquant, on met un défaut non compressé
    if not getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None):
        if not getattr(ds, "file_meta", None):
            ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    arr = ds.pixel_array.astype(np.float32)

    # Rescale HU-like
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    arr = arr * slope + intercept

    # Windowing (prend valeurs DICOM si dispo)
    wc_val = getattr(ds, "WindowCenter", default_wc)
    ww_val = getattr(ds, "WindowWidth", default_ww)
    try:
        wc = float(wc_val[0] if isinstance(wc_val, (list, tuple)) else wc_val)
        ww = float(ww_val[0] if isinstance(ww_val, (list, tuple)) else ww_val)
    except Exception:
        wc, ww = default_wc, default_ww

    lower, upper = wc - ww / 2.0, wc + ww / 2.0
    arr = np.clip(arr, lower, upper)
    arr = (arr - lower) / max(upper - lower, 1e-6)  # [0,1]
    arr8 = (arr * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(arr8).convert("RGB")

# ---------- Prétraitement tolérant ----------
def preprocess_image_to_tensor(data: bytes, filename: str = "") -> Tuple[Image.Image, torch.Tensor]:
    """
    Stratégie tolérante :
    1) Essayer comme image standard (PNG/JPG)
    2) Essayer comme DICOM (signature/preamble/extension)
    3) Re-PIL fallback
    """
    def _to_tensor(pil: Image.Image):
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return pil, tfm(pil).unsqueeze(0)

    tried = []
    is_ext_dcm = filename.lower().endswith((".dcm", ".dicom"))

    # 1) PNG/JPG d'abord si pas d'extension DICOM
    if not is_ext_dcm:
        try:
            pil = Image.open(io.BytesIO(data)).convert("RGB")
            return _to_tensor(pil)
        except Exception as e:
            tried.append(f"PIL1:{e}")

    # 2) DICOM si signature/preamble/extension
    try:
        if _has_dicom_preamble(data) or _looks_like_dicom(data) or is_ext_dcm:
            pil = dicom_to_pil(data)
            return _to_tensor(pil)
    except Exception as e:
        tried.append(f"DICOM:{e}")

    # 3) Re-PIL fallback
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return _to_tensor(pil)
    except Exception as e:
        tried.append(f"PIL2:{e}")
        raise ValueError("Could not parse image. " + " | ".join(tried))

# ---------- Inference + Grad-CAM ----------
class InferenceEngine:
    def __init__(self, num_classes: int = len(CLASSES), model_path: str = "models/model.pt"):
        self.device = torch.device("cpu")
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, num_classes)

        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded fine-tuned weights from {model_path}")
            except Exception as e:
                print("Failed to load fine-tuned weights:", e)

        self.model.eval().to(self.device)
        self.target_layer_name = "layer4"

    def predict(self, tensor: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            logits = self.model(tensor.to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return {cls: float(p) for cls, p in zip(CLASSES, probs)}

    def _get_target_layer(self):
        return getattr(self.model, self.target_layer_name)

class _ActivationsAndGradients:
    def __init__(self, model, target_layer):
        self.gradients = None
        self.activations = None
        self.h1 = target_layer.register_forward_hook(self.forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.h1.remove()
        self.h2.remove()

def compute_gradcam(engine: InferenceEngine, tensor: torch.Tensor) -> np.ndarray:
    model = engine.model
    target_layer = engine._get_target_layer()
    hook = _ActivationsAndGradients(model, target_layer)

    tensor = tensor.requires_grad_(True)
    logits = model(tensor)
    target_class = int(torch.argmax(logits, dim=1))
    score = logits[:, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)

    gradients = hook.gradients
    activations = hook.activations

    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1)
    cam = torch.relu(cam)
    cam = cam[0].cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    hook.remove()
    return cam

def overlay_heatmap_on_pil(pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    pil_resized = pil.resize((heatmap.shape[1], heatmap.shape[0]))
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
    base = np.array(pil_resized).astype(np.float32)
    overlay = (alpha * heatmap_color + (1 - alpha) * base).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)

def save_gradcam_overlay(engine: InferenceEngine, pil: Image.Image, tensor, out_path: str):
    heatmap = compute_gradcam(engine, tensor)
    overlay = overlay_heatmap_on_pil(pil, heatmap, alpha=0.35)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    overlay.save(out_path)

# Exports explicites
__all__ = ["InferenceEngine", "preprocess_image_to_tensor", "save_gradcam_overlay"]
