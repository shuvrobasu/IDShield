#!/usr/bin/env python3
"""
IdentityShield Pro - Image Privacy Protection Tool
A comprehensive face detection and identity masking application.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser, simpledialog
from PIL import Image, ImageTk, ImageFilter, ImageDraw
import cv2
import numpy as np
import os
import copy
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum, auto
import threading
try:
    import fitz
except ImportError:
    fitz = None

SETTINGS_FILE = Path(__file__).with_name("settings.sec")
DEFAULT_SETTINGS = {
    "sam2_dir": "",
    "sam2_default": "Base+",
    "sam2_models": {
        "Large": ["sam2_hiera_large.pt", "sam2_hiera_l.yaml"],
        "Base+": ["sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"],
        "Small": ["sam2_hiera_small.pt", "sam2_hiera_s.yaml"],
        "Tiny": ["sam2_hiera_tiny.pt", "sam2_hiera_t.yaml"],
    },
    "yolo_person": "yolov8x.pt",
    "yolo_face": "",
    "east_model": "",
    "output_dir": "",
    "recent_files": [],
}
SETTINGS = copy.deepcopy(DEFAULT_SETTINGS)


def load_settings():
    """Merge settings.sec (JSON) over the defaults; a missing or broken file keeps defaults."""
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    for k in DEFAULT_SETTINGS:
        if k in data and type(data[k]) is type(DEFAULT_SETTINGS[k]):
            SETTINGS[k] = data[k]
    if not SETTINGS["sam2_models"]:
        SETTINGS["sam2_models"] = copy.deepcopy(DEFAULT_SETTINGS["sam2_models"])


def save_settings():
    SETTINGS_FILE.write_text(json.dumps(SETTINGS, indent=2), encoding="utf-8")


load_settings()
os.environ.setdefault("YOLO_VERBOSE", "False")  # ponytail: mutes ultralytics WARNINGs (settings reset, NMS time limit); set "True" to debug

# ─── Data Models ───────────────────────────────────────────────────────────────

class ToolType(Enum):
    SELECT = auto()
    AUTO_DETECT = auto()
    PEOPLE = auto()
    RECTANGLE = auto()
    POLYGON = auto()
    FREEHAND = auto()
    BRUSH = auto()
    CROP = auto()          # NEW


class MaskType(Enum):
    PIXELATE = "Pixelate"
    GAUSSIAN = "Gaussian Blur"
    SOLID = "Solid Fill"
    NOISE = "Noise Fill"


@dataclass
class MaskRegion:
    """Represents a masked region on the image."""
    region_id: str
    tool_type: ToolType
    points: List[Tuple[int, int]]  # image-space coordinates
    mask_type: MaskType = MaskType.PIXELATE
    block_size: int = 15
    blur_strength: int = 25
    fill_color: Tuple[int, int, int] = (0, 0, 0)
    opacity: float = 1.0
    visible: bool = True
    label: str = ""

    def get_bbox(self):
        if not self.points:
            return (0, 0, 0, 0)
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))


# ─── Theme / Style ────────────────────────────────────────────────────────────

class AppTheme:
    BG_DARK = "#1a1a2e"
    BG_MEDIUM = "#16213e"
    BG_LIGHT = "#0f3460"
    BG_PANEL = "#1e2a4a"
    ACCENT = "#e94560"
    ACCENT_HOVER = "#ff6b81"
    TEXT = "#eaeaea"
    TEXT_DIM = "#8899aa"
    BORDER = "#2a3a5a"
    CANVAS_BG = "#0d1117"
    SUCCESS = "#2ecc71"
    WARNING = "#f39c12"
    BUTTON_BG = "#233560"
    BUTTON_ACTIVE = "#2a4a7f"
    SCROLLBAR = "#3a4a6a"
    ENTRY_BG = "#0f1a30"


# ─── Image Processing Engine ──────────────────────────────────────────────────

class ImageProcessor:
    """Handles all image processing operations."""

    _face_cascade = None

    @classmethod
    def get_face_cascade(cls):
        if cls._face_cascade is None:
            # cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            except AttributeError:
                    cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
            cls._face_cascade = cv2.CascadeClassifier(cascade_path)

        return cls._face_cascade

    face_engine = "Haar"
    face_note = ""
    _yolo_face = None

    @classmethod
    def _get_yolo_face(cls):
        """Load the YOLO face model once; False (with face_note) if unavailable."""
        if cls._yolo_face is None:
            if not os.path.exists(SETTINGS["yolo_face"]):
                cls._yolo_face = False
                cls.face_note = "YOLO face weights not set (File > Settings); using Haar"
            else:
                try:
                    from ultralytics import YOLO
                    cls._yolo_face = YOLO(SETTINGS["yolo_face"])
                except Exception as e:
                    cls._yolo_face = False
                    cls.face_note = f"YOLO face model failed to load: {e}"
        return cls._yolo_face

    @staticmethod
    def _detect_faces_yolo(model, image, min_size, confidence_threshold):
        # ponytail: slider scaled to half, so the default 50% = YOLO's usual 0.25
        res = model(image, conf=confidence_threshold * 0.5, imgsz=1280, verbose=False)[0]
        h, w = image.shape[:2]
        faces = []
        for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            if min(x2 - x1, y2 - y1) < min_size:
                continue
            faces.append((x1, y1, x2 - x1, y2 - y1))
        ImageProcessor.face_engine = "YOLO"
        return faces

    @staticmethod
    def detect_faces(image: np.ndarray, scale_factor=1.1,
                     min_neighbors=5, min_size=30, confidence_threshold=0.5) -> List[Tuple[int, int, int, int]]:
        """YOLO face model (handles tilted/profile faces); Haar frontal-only fallback."""
        model = ImageProcessor._get_yolo_face()
        if model:
            try:
                return ImageProcessor._detect_faces_yolo(
                    model, image, min_size, confidence_threshold)
            except Exception as e:
                ImageProcessor.face_note = f"YOLO face failed: {e}"
        ImageProcessor.face_engine = "Haar"
        return ImageProcessor._detect_faces_haar(
            image, scale_factor, min_neighbors, min_size, confidence_threshold)

    @staticmethod
    def _detect_faces_haar(image: np.ndarray, scale_factor=1.1,
                           min_neighbors=5, min_size=30, confidence_threshold=0.5) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        cascade = ImageProcessor.get_face_cascade()

        # Adjust min_neighbors based on confidence_threshold
        adjusted_neighbors = max(1, int(min_neighbors * confidence_threshold * 2))

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=adjusted_neighbors,
            minSize=(min_size, min_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces] if len(faces) > 0 else []

    @staticmethod
    def apply_pixelate(image: np.ndarray, mask: np.ndarray, block_size: int) -> np.ndarray:
        result = image.copy()
        h, w = image.shape[:2]
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                y2 = min(y + block_size, h)
                x2 = min(x + block_size, w)
                roi_mask = mask[y:y2, x:x2]
                if roi_mask.any():
                    avg_color = image[y:y2, x:x2].mean(axis=(0, 1)).astype(np.uint8)
                    block = np.full_like(image[y:y2, x:x2], avg_color)
                    mask_3ch = np.stack([roi_mask] * 3, axis=-1) if len(image.shape) == 3 else roi_mask
                    result[y:y2, x:x2] = np.where(mask_3ch, block, result[y:y2, x:x2])
        return result

    @staticmethod
    def apply_gaussian(image: np.ndarray, mask: np.ndarray, strength: int) -> np.ndarray:
        ksize = strength | 1  # ensure odd
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        mask_3ch = np.stack([mask] * 3, axis=-1) if len(image.shape) == 3 else mask
        result = np.where(mask_3ch, blurred, image)
        return result

    @staticmethod
    def apply_solid(image: np.ndarray, mask: np.ndarray,
                    color: Tuple[int, int, int]) -> np.ndarray:
        result = image.copy()
        color_bgr = (color[2], color[1], color[0])
        result[mask > 0] = color_bgr
        return result

    @staticmethod
    def apply_noise(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = image.copy()
        noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)
        mask_3ch = np.stack([mask] * 3, axis=-1) if len(image.shape) == 3 else mask
        result = np.where(mask_3ch, noise, result)
        return result

    @staticmethod
    def create_mask_from_region(region: MaskRegion, img_shape: Tuple[int, int]) -> np.ndarray:
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if region.tool_type == ToolType.RECTANGLE or region.tool_type == ToolType.AUTO_DETECT:
            if len(region.points) >= 2:
                x1, y1 = region.points[0]
                x2, y2 = region.points[1]
                x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
                y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
                mask[y1:y2, x1:x2] = 255
        elif region.tool_type in (ToolType.POLYGON, ToolType.FREEHAND, ToolType.BRUSH):
            if len(region.points) >= 3:
                pts = np.array(region.points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

        return mask

    @staticmethod
    def apply_mask_region(image: np.ndarray, region: MaskRegion) -> np.ndarray:
        if not region.visible:
            return image

        mask = ImageProcessor.create_mask_from_region(region, image.shape)
        if not mask.any():
            return image

        if region.mask_type == MaskType.PIXELATE:
            result = ImageProcessor.apply_pixelate(image, mask, region.block_size)
        elif region.mask_type == MaskType.GAUSSIAN:
            result = ImageProcessor.apply_gaussian(image, mask, region.blur_strength)
        elif region.mask_type == MaskType.SOLID:
            result = ImageProcessor.apply_solid(image, mask, region.fill_color)
        elif region.mask_type == MaskType.NOISE:
            result = ImageProcessor.apply_noise(image, mask)
        else:
            return image

        if region.opacity < 1.0:
            mask_3ch = np.stack([mask] * 3, axis=-1) if len(image.shape) == 3 else mask
            result = np.where(
                mask_3ch,
                cv2.addWeighted(image, 1 - region.opacity, result, region.opacity, 0),
                image
            )

        return result

    @staticmethod
    def strip_exif(pil_image: Image.Image) -> Image.Image:
        data = list(pil_image.getdata())
        clean = Image.new(pil_image.mode, pil_image.size)
        clean.putdata(data)
        return clean


# ─── History Manager ──────────────────────────────────────────────────────────

class HistoryManager:
    def __init__(self, max_history=50):
        self.max_history = max_history
        self.undo_stack: List[List[MaskRegion]] = []
        self.redo_stack: List[List[MaskRegion]] = []

    def push(self, state: List[MaskRegion]):
        self.undo_stack.append(copy.deepcopy(state))
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self, current_state: List[MaskRegion]) -> Optional[List[MaskRegion]]:
        if not self.undo_stack:
            return None
        self.redo_stack.append(copy.deepcopy(current_state))
        return self.undo_stack.pop()

    def redo(self, current_state: List[MaskRegion]) -> Optional[List[MaskRegion]]:
        if not self.redo_stack:
            return None
        self.undo_stack.append(copy.deepcopy(current_state))
        return self.redo_stack.pop()

    def clear(self):
        self.undo_stack.clear()
        self.redo_stack.clear()


# ─── Custom Widgets ───────────────────────────────────────────────────────────

class ToolTip:
    """Hover tooltip for a widget and everything inside it."""

    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip = None
        self.job = None
        self._bind_tree(widget)

    def _bind_tree(self, w):
        w.bind("<Enter>", self._schedule, add="+")
        w.bind("<Leave>", self._leave, add="+")
        w.bind("<ButtonPress>", self._hide, add="+")
        for child in w.winfo_children():
            self._bind_tree(child)

    def _schedule(self, event=None):
        if self.tip is not None or self.job is not None:
            return  # still inside the widget (moved between its children)
        self.job = self.widget.after(self.delay, self._show)

    def _leave(self, event=None):
        inside = self.widget.winfo_containing(*self.widget.winfo_pointerxy())
        if inside is not None and str(inside).startswith(str(self.widget)):
            return  # pointer moved onto a child, not out of the widget
        self._hide()

    def _show(self):
        self.job = None
        if self.tip is not None or not self.widget.winfo_exists():
            return
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        tk.Label(self.tip, text=self.text, justify="left", bg=AppTheme.BG_LIGHT,
                 fg=AppTheme.TEXT, font=("Segoe UI", 9), relief="solid",
                 borderwidth=1, padx=8, pady=5, wraplength=280).pack()
        self.tip.update_idletasks()
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        x = min(x, self.widget.winfo_screenwidth() - self.tip.winfo_reqwidth() - 8)
        self.tip.wm_geometry(f"+{max(0, x)}+{y}")

    def _hide(self, event=None):
        if self.job is not None:
            self.widget.after_cancel(self.job)
            self.job = None
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class ModernSlider(tk.Frame):
    """Custom styled slider with label and value display."""
    def __init__(self, parent, label, from_=0, to=100, value=50,
                 command=None, fmt="{:.0f}", tip=None, **kwargs):
        super().__init__(parent, bg=AppTheme.BG_PANEL, **kwargs)
        self.command = command
        self.fmt = fmt

        header = tk.Frame(self, bg=AppTheme.BG_PANEL)
        header.pack(fill=tk.X, padx=5)

        tk.Label(header, text=label, bg=AppTheme.BG_PANEL,
                 fg=AppTheme.TEXT_DIM, font=("Segoe UI", 9)).pack(side=tk.LEFT)

        self.val_label = tk.Label(header, text=fmt.format(value),
                                   bg=AppTheme.BG_PANEL, fg=AppTheme.ACCENT,
                                   font=("Segoe UI", 9, "bold"))
        self.val_label.pack(side=tk.RIGHT)

        self.var = tk.DoubleVar(value=value)
        self.scale = ttk.Scale(self, from_=from_, to=to, variable=self.var,
                               orient=tk.HORIZONTAL, command=self._on_change)
        self.scale.pack(fill=tk.X, padx=5, pady=(0, 5))
        if tip:
            ToolTip(self, tip)

    def _on_change(self, val):
        self.val_label.config(text=self.fmt.format(float(val)))
        if self.command:
            self.command(float(val))

    def get(self):
        return self.var.get()

    def set(self, val):
        self.var.set(val)
        self.val_label.config(text=self.fmt.format(val))


class ToolButton(tk.Frame):
    """Custom styled tool button."""
    def __init__(self, parent, text, icon="", command=None, tip=None, **kwargs):
        super().__init__(parent, bg=AppTheme.BG_PANEL, **kwargs)
        self.command = command
        self.active = False

        self.btn = tk.Label(
            self, text=f" {icon} {text}", bg=AppTheme.BUTTON_BG,
            fg=AppTheme.TEXT, font=("Segoe UI", 10),
            padx=10, pady=6, anchor="w", cursor="hand2"
        )
        self.btn.pack(fill=tk.X, padx=5, pady=2)

        self.btn.bind("<Enter>", self._on_enter)
        self.btn.bind("<Leave>", self._on_leave)
        self.btn.bind("<Button-1>", self._on_click)
        if tip:
            ToolTip(self, tip)

    def _on_enter(self, e):
        if not self.active:
            self.btn.config(bg=AppTheme.BUTTON_ACTIVE)

    def _on_leave(self, e):
        if not self.active:
            self.btn.config(bg=AppTheme.BUTTON_BG)

    def _on_click(self, e):
        if self.command:
            self.command()

    def set_active(self, active):
        self.active = active
        if active:
            self.btn.config(bg=AppTheme.ACCENT, fg="white")
        else:
            self.btn.config(bg=AppTheme.BUTTON_BG, fg=AppTheme.TEXT)


class ActionButton(tk.Label):
    """Styled action button."""
    def __init__(self, parent, text, command=None, accent=False, tip=None, **kwargs):
        bg = AppTheme.ACCENT if accent else AppTheme.BUTTON_BG
        super().__init__(
            parent, text=text, bg=bg, fg="white",
            font=("Segoe UI", 10, "bold" if accent else "normal"),
            padx=12, pady=6, cursor="hand2", **kwargs
        )
        self.default_bg = bg
        self.command = command
        self.bind("<Enter>", lambda e: self.config(
            bg=AppTheme.ACCENT_HOVER if accent else AppTheme.BUTTON_ACTIVE))
        self.bind("<Leave>", lambda e: self.config(bg=self.default_bg))
        self.bind("<Button-1>", lambda e: command() if command else None)
        if tip:
            ToolTip(self, tip)


# ─── Main Application ─────────────────────────────────────────────────────────

def _with_face(boxes, faces):
    """Indices of boxes that own a face. Each face goes to ONE box (the one whose head position it fits best),
    so a big box that merely overlaps someone else's face (an arm, clothing) gets none and is dropped."""
    owners = set()
    for (x, y, fw, fh) in faces:
        fx = x + fw / 2
        fy = y + fh / 2
        best = None
        best_d = 0.0
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if not (x1 <= fx <= x2 and y1 <= fy <= y2):
                continue
            d = np.hypot(fx - (x1 + x2) / 2, fy - (y1 + 0.15 * (y2 - y1))) / max(1.0, x2 - x1)
            if best is None or d < best_d:
                best = i
                best_d = d
        if best is not None:
            owners.add(best)
    return sorted(owners)


def _pick_whole_person(masks, scores, box):
    """Of SAM's candidate masks keep the largest one nearly as confident as the best (whole body, not a part),
    clipped to its box so it cannot run off into the background."""
    best = float(np.max(scores))
    ok = [i for i in range(len(masks)) if scores[i] >= best - 0.05]
    pick = max(ok, key=lambda i: int((masks[i] > 0).sum()))
    m = masks[pick] > 0
    x1, y1, x2, y2 = [int(v) for v in box]
    clip = np.zeros_like(m)
    clip[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = True
    return m & clip


def _filter_boxes(boxes, confs, img_area):
    """Indices of boxes worth segmenting, best first: drops tiny, duplicate and fragment boxes."""
    kept = []
    for i in np.argsort(-confs):
        x1, y1, x2, y2 = boxes[i]
        area = max(1.0, (x2 - x1) * (y2 - y1))
        if area < 0.002 * img_area:
            continue
        drop = False
        for j in kept:
            a1, b1, a2, b2 = boxes[j]
            inter = max(0.0, min(x2, a2) - max(x1, a1)) * max(0.0, min(y2, b2) - max(y1, b1))
            iou = inter / (area + (a2 - a1) * (b2 - b1) - inter)
            if iou > 0.6:
                drop = True
            elif inter / area > 0.8 and confs[i] < 0.5:
                drop = True
            if drop:
                break
        if not drop:
            kept.append(int(i))
    return kept


def _fill_gaps(masks):
    """Fill gaps inside each person's outline (convex hull minus mask), never taking another person's pixels.
    Gaps bigger than a fifth of the person (a raised arm, a wide background wedge) are left alone."""
    if not masks:
        return masks
    h, w = masks[0].shape
    for i, m in enumerate(masks):
        others = np.zeros((h, w), bool)
        for j, o in enumerate(masks):
            if j != i:
                others |= o
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        hull = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(np.vstack(cnts)), 1)
        gap = (hull > 0) & ~m & ~others
        n, lab, st, _ = cv2.connectedComponentsWithStats(gap.astype(np.uint8), connectivity=4)
        limit = 0.20 * int(m.sum())
        for c in range(1, n):
            if st[c, cv2.CC_STAT_AREA] <= limit:
                m[lab == c] = True
    return masks


def _clean_people(masks, boxes):
    """Masks (best first) -> one clean, non-overlapping mask per real person.
    Drops fragments and duplicates, fills thin gaps (fingers over faces), gives shared pixels to one person."""
    h, w = masks[0].shape
    keep = []
    for i, m in enumerate(masks):
        x1, y1, x2, y2 = boxes[i]
        area = int(m.sum())
        if area < 100 or area < 0.10 * (x2 - x1) * (y2 - y1):
            continue
        if any((m & masks[j]).sum() > 0.8 * min(area, masks[j].sum()) for j in keep):
            continue
        keep.append(i)
    src = [masks[i] for i in keep]
    bxs = [boxes[i] for i in keep]
    out = []
    for i, m in enumerate(src):
        x1, y1, x2, y2 = [int(v) for v in bxs[i]]
        k = int(min(61, max(3, 0.10 * min(x2 - x1, y2 - y1)))) | 1
        r1, r2 = max(0, y1 - k), min(h, y2 + k)
        c1, c2 = max(0, x1 - k), min(w, x2 + k)
        crop = m[r1:r2, c1:c2]
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(crop.astype(np.uint8), cv2.MORPH_CLOSE, kern).astype(bool)
        other = np.zeros_like(crop)
        for j, o in enumerate(src):
            if j != i:
                other |= o[r1:r2, c1:c2]
        grown = m.copy()
        grown[r1:r2, c1:c2] = crop | (closed & ~other)
        out.append(grown)
    count = np.zeros((h, w), np.uint8)
    for m in out:
        count += m
    ys, xs = np.nonzero(count > 1)
    if len(ys):
        best = np.full(len(ys), np.inf)
        owner = np.full(len(ys), -1)
        for i, m in enumerate(out):
            x1, y1, x2, y2 = bxs[i]
            d = np.hypot(xs - (x1 + x2) / 2, ys - (y1 + 0.18 * (y2 - y1))) / max(1.0, x2 - x1)
            d = np.where(m[ys, xs], d, np.inf)
            better = d < best
            best[better] = d[better]
            owner[better] = i
        for i, m in enumerate(out):
            m[ys, xs] = owner == i
    res_m = []
    res_b = []
    for i, m in enumerate(out):
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n > 2:
            big = st[1:, cv2.CC_STAT_AREA].max()
            ids = [c for c in range(1, n) if st[c, cv2.CC_STAT_AREA] >= 0.03 * big]
            m = np.isin(lab, ids)
        if int(m.sum()) >= 100:
            res_m.append(m)
            res_b.append(bxs[i])
    return _fill_gaps(res_m), res_b


class IdentityShieldApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🛡️ IdentityShield Pro - Image Privacy Protection")
        # self.root.geometry("1400x850")
        # self.root.state("zoomed")

        if self.root.tk.call("tk", "windowingsystem") == "win32":
            self.root.state("zoomed")
        else:
            self.root.geometry(
                f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0"
            )
        self.root.minsize(1100, 700)

        self.root.configure(bg=AppTheme.BG_DARK)

        # State
        self.current_image: Optional[np.ndarray] = None
        self.display_image: Optional[ImageTk.PhotoImage] = None
        self.original_image: Optional[np.ndarray] = None
        self.file_list: List[str] = []
        self.current_file_index: int = -1
        self.mask_regions: List[MaskRegion] = []
        self.selected_region: Optional[MaskRegion] = None
        self.current_tool: ToolType = ToolType.SELECT
        self.history = HistoryManager()
        self.region_counter = 0
        self.pdf_document = None
        self.pdf_path = None
        self.pdf_page_index = 0
        self.pdf_page_masks = {}
        self.pdf_page_sizes = {}
        self.pdf_render_scale = 2.0

        # Mouse handling
        self.lmb_down = False
        self.rmb_down = False
        self.both_btn_pan = False

        # Canvas state
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.canvas_drag_start = None
        self.is_panning = False

        # Drawing state
        self.drawing = False
        self.draw_start = None
        self.draw_points: List[Tuple[int, int]] = []
        self.temp_canvas_items = []

        # Properties defaults
        self.mask_type_var = tk.StringVar(value=MaskType.PIXELATE.value)
        self.block_size_var = tk.IntVar(value=15)
        self.blur_strength_var = tk.IntVar(value=25)
        self.opacity_var = tk.DoubleVar(value=100.0)
        self.confidence_var = tk.DoubleVar(value=50.0)
        self.min_face_var = tk.IntVar(value=30)
        self.fill_color = (0, 0, 0)
        self.strip_exif_var = tk.BooleanVar(value=True)

        self._configure_styles()
        self._build_ui()
        self._bind_shortcuts()

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Dark.TFrame", background=AppTheme.BG_DARK)
        style.configure("Panel.TFrame", background=AppTheme.BG_PANEL)
        style.configure("TScale", background=AppTheme.BG_PANEL,
                        troughcolor=AppTheme.BG_DARK, sliderthickness=14)
        style.map("TScale", background=[("active", AppTheme.ACCENT)])

        style.configure("Dark.TLabel", background=AppTheme.BG_DARK,
                        foreground=AppTheme.TEXT, font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=AppTheme.BG_PANEL,
                        foreground=AppTheme.TEXT, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=AppTheme.BG_PANEL,
                        foreground=AppTheme.ACCENT, font=("Segoe UI", 11, "bold"))

        style.configure("Dark.TRadiobutton", background=AppTheme.BG_PANEL,
                        foreground=AppTheme.TEXT, font=("Segoe UI", 9),
                        indicatorcolor=AppTheme.ACCENT)
        style.map("Dark.TRadiobutton",
                  background=[("active", AppTheme.BG_PANEL)],
                  indicatorcolor=[("selected", AppTheme.ACCENT)])

        # In _configure_styles(), ADD/MODIFY the checkbutton style:
        style.configure("Dark.TCheckbutton", background=AppTheme.BG_PANEL,
                        foreground=AppTheme.TEXT, font=("Segoe UI", 9))
        style.map("Dark.TCheckbutton",
                  foreground=[("active", AppTheme.TEXT), ("hover", AppTheme.TEXT)],
                  background=[("active", AppTheme.BG_PANEL), ("hover", AppTheme.BG_PANEL)])

        style.configure("Accent.TButton",
                        background=AppTheme.ACCENT,
                        foreground="white",
                        font=("Segoe UI", 10, "bold"),
                        padding=(10, 5))

    def _build_ui(self):
        # Menu bar
        self._build_menu()

        # Toolbar
        self._build_toolbar()

        # Main content area
        content = tk.Frame(self.root, bg=AppTheme.BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        # Left panel
        self._build_left_panel(content)

        # Canvas
        self._build_canvas(content)

        # Right panel
        self._build_right_panel(content)

        # Status bar
        self._build_status_bar()

    # ── Menu Bar ────────────────────────────────────────────────────────

    def _build_menu(self):
        menubar = tk.Menu(self.root, bg=AppTheme.BG_MEDIUM, fg=AppTheme.TEXT,
                         activebackground=AppTheme.ACCENT,
                         activeforeground="white", font=("Segoe UI", 10))

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg=AppTheme.BG_MEDIUM,
                           fg=AppTheme.TEXT, activebackground=AppTheme.ACCENT,
                           font=("Segoe UI", 10))
        file_menu.add_command(label="📂 Open Image...", command=self.open_image,
                             accelerator="Ctrl+O")
        file_menu.add_command(label="Open PDF...", command=self.open_pdf)
        self.recent_menu = tk.Menu(file_menu, tearoff=0, bg=AppTheme.BG_MEDIUM,
                                   fg=AppTheme.TEXT, activebackground=AppTheme.ACCENT,
                                   font=("Segoe UI", 10))
        file_menu.add_cascade(label="🕘 Recent Files", menu=self.recent_menu)
        file_menu.configure(postcommand=self._refresh_recent)
        file_menu.add_command(label="📁 Open Folder...", command=self.open_folder,
                             accelerator="Ctrl+Shift+O")
        file_menu.add_separator()
        file_menu.add_command(label="💾 Save", command=self.save_image,
                             accelerator="Ctrl+S")
        file_menu.add_command(label="💾 Save As...", command=self.save_as,
                             accelerator="Ctrl+Shift+S")
        file_menu.add_command(label="Save Redacted PDF...", command=self.save_pdf_as)
        file_menu.add_command(label="Search/Regex Redact All PDF Pages...", command=self.redact_pdf_terms)
        file_menu.add_command(label="📤 Export...", command=self.export_image,
                             accelerator="Ctrl+E")
        file_menu.add_separator()
        file_menu.add_command(label="⚙ Settings...", command=self.open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0, bg=AppTheme.BG_MEDIUM,
                           fg=AppTheme.TEXT, activebackground=AppTheme.ACCENT,
                           font=("Segoe UI", 10))
        edit_menu.add_command(label="↩ Undo", command=self.undo,
                             accelerator="Ctrl+Z")
        edit_menu.add_command(label="↪ Redo", command=self.redo,
                             accelerator="Ctrl+Y")
        # In _build_menu(), inside edit_menu, ADD before separator:
        edit_menu.add_command(label="📋 Paste from Clipboard", command=self.paste_from_clipboard,
                              accelerator="Ctrl+V")
        edit_menu.add_separator()

        edit_menu.add_command(label="🗑 Clear All Masks",
                             command=self.clear_all_masks)
        edit_menu.add_command(label="🔄 Reset Image",
                             command=self.reset_image)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0, bg=AppTheme.BG_MEDIUM,
                           fg=AppTheme.TEXT, activebackground=AppTheme.ACCENT,
                           font=("Segoe UI", 10))
        view_menu.add_command(label="🔍 Zoom In", command=self.zoom_in,
                             accelerator="+")
        view_menu.add_command(label="🔍 Zoom Out", command=self.zoom_out,
                             accelerator="-")
        view_menu.add_command(label="🔍 Fit to Window",
                             command=self.zoom_fit, accelerator="Ctrl+0")
        view_menu.add_command(label="🔍 100%",
                             command=lambda: self.set_zoom(1.0), accelerator="Ctrl+1")
        menubar.add_cascade(label="View", menu=view_menu)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0, bg=AppTheme.BG_MEDIUM,
                            fg=AppTheme.TEXT, activebackground=AppTheme.ACCENT,
                            font=("Segoe UI", 10))
        tools_menu.add_command(label="🔍 Auto-Detect Faces",
                              command=self.auto_detect, accelerator="Ctrl+D")
        # In _build_menu(), inside tools_menu, ADD after auto-detect:
        tools_menu.add_command(label="✂ Crop Tool", accelerator="C",
                               command=lambda: self.set_tool(ToolType.CROP))
        tools_menu.add_separator()
        tools_menu.add_command(label="🖼 Remove Background",
                               command=self.remove_background)
        tools_menu.add_command(label="🔤 Remove Text",
                               command=self.remove_text)
        tools_menu.add_command(label="🧍 Detect People",
                               command=self.extract_humans)
        tools_menu.add_separator()
        tools_menu.add_command(label="▭ Rectangle Tool", accelerator="R",
                              command=lambda: self.set_tool(ToolType.RECTANGLE))
        tools_menu.add_command(label="⬠ Polygon Tool", accelerator="P",
                              command=lambda: self.set_tool(ToolType.POLYGON))
        tools_menu.add_command(label="✏ Freehand Tool", accelerator="F",
                              command=lambda: self.set_tool(ToolType.FREEHAND))
        tools_menu.add_command(label="🖌 Brush Tool", accelerator="B",
                              command=lambda: self.set_tool(ToolType.BRUSH))
        menubar.add_cascade(label="Tools", menu=tools_menu)

        # Batch menu
        batch_menu = tk.Menu(menubar, tearoff=0, bg=AppTheme.BG_MEDIUM,
                            fg=AppTheme.TEXT, activebackground=AppTheme.ACCENT,
                            font=("Segoe UI", 10))
        batch_menu.add_command(label="🔍 Detect All in Folder",
                              command=self.batch_detect)
        batch_menu.add_command(label="📤 Export All Processed",
                              command=self.batch_export)
        menubar.add_cascade(label="Batch", menu=batch_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0, bg=AppTheme.BG_MEDIUM,
                           fg=AppTheme.TEXT, activebackground=AppTheme.ACCENT,
                           font=("Segoe UI", 10))
        help_menu.add_command(label="⌨ Keyboard Shortcuts",
                             command=self.show_shortcuts, accelerator="Ctrl+?")
        help_menu.add_command(label="ℹ About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    # ── Toolbar ─────────────────────────────────────────────────────────

    def _build_toolbar(self):
        toolbar = tk.Frame(self.root, bg=AppTheme.BG_MEDIUM, height=40)
        toolbar.pack(fill=tk.X, padx=4, pady=(4, 2))
        toolbar.pack_propagate(False)

        buttons = [
            ("📂 Open", self.open_image),
            ("PDF", self.open_pdf),
            ("📁 Folder", self.open_folder),
            ("💾 Save", self.save_image),
            ("💾 Save As", self.save_as),
            # In _build_toolbar(), ADD to the buttons list after ("💾 Save As", self.save_as):
            ("📋 Paste", self.paste_from_clipboard),
            (None, None),
            ("↩ Undo", self.undo),
            ("↪ Redo", self.redo),
            (None, None),
            ("🔄 Reset", self.reset_image),
            ("📤 Export", self.export_image),
            ("Redact PDF", self.save_pdf_as),
            ("Find+Redact", self.redact_pdf_terms),
            (None, None),
            ("◀ Prev", self.prev_image),
            ("▶ Next", self.next_image),
        ]

        for text, cmd in buttons:
            if text is None:
                sep = tk.Frame(toolbar, bg=AppTheme.BORDER, width=2, height=28)
                sep.pack(side=tk.LEFT, padx=6, pady=6)
            else:
                btn = tk.Label(
                    toolbar, text=text, bg=AppTheme.BUTTON_BG,
                    fg=AppTheme.TEXT, font=("Segoe UI", 9),
                    padx=8, pady=4, cursor="hand2"
                )
                btn.pack(side=tk.LEFT, padx=2, pady=4)
                btn.bind("<Enter>", lambda e, b=btn: b.config(bg=AppTheme.BUTTON_ACTIVE))
                btn.bind("<Leave>", lambda e, b=btn: b.config(bg=AppTheme.BUTTON_BG))
                btn.bind("<Button-1>", lambda e, c=cmd: c() if c else None)

    # ── Left Panel ──────────────────────────────────────────────────────

    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg=AppTheme.BG_PANEL, width=200)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        left.pack_propagate(False)

        # Tools section
        tk.Label(left, text="── TOOLS ──", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.ACCENT, font=("Segoe UI", 10, "bold")).pack(
            fill=tk.X, padx=10, pady=(10, 5))

        self.tool_buttons = {}
        tool_tips = {
            ToolType.SELECT: "Click a mask to select, move or resize it.",
            ToolType.RECTANGLE: "Drag to draw a rectangular mask.",
            ToolType.POLYGON: "Click to add points; double-click to finish the shape.",
            ToolType.FREEHAND: "Draw any shape by dragging; release to finish.",
            ToolType.BRUSH: "Paint a mask with a brush. Width follows Block Size.",
            ToolType.CROP: "Drag to choose the area to keep, then press Enter.",
            ToolType.AUTO_DETECT: "Find every face and add a mask over each one.",
            ToolType.PEOPLE: "Find every person, then redact their face or body, remove them, or save each as a cut-out PNG.",
        }
        tools = [
            (ToolType.SELECT, "Select", "🔘"),
            (ToolType.AUTO_DETECT, "Auto-Detect", "🔍"),
            (ToolType.PEOPLE, "Detect People", "🧍"),
            (ToolType.RECTANGLE, "Rectangle", "▭"),
            (ToolType.POLYGON, "Polygon", "⬠"),
            (ToolType.FREEHAND, "Freehand", "✏"),
            (ToolType.BRUSH, "Brush Mask", "🖌"),
            # In _build_left_panel(), ADD to tools list after BRUSH:
            (ToolType.CROP, "Crop", "✂"),
        ]

        for tool_type, name, icon in tools:
            btn = ToolButton(left, name, icon,
                            command=lambda t=tool_type: self.set_tool(t),
                            tip=tool_tips.get(tool_type))
            btn.pack(fill=tk.X)
            self.tool_buttons[tool_type] = btn

        # Actions section
        tk.Frame(left, bg=AppTheme.BORDER, height=1).pack(fill=tk.X, padx=10, pady=10)
        tk.Label(left, text="── ACTIONS ──", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.ACCENT, font=("Segoe UI", 10, "bold")).pack(
            fill=tk.X, padx=10, pady=(0, 5))

        ActionButton(left, "  ▶ Apply All Masks  ", self.apply_all_masks,
                    accent=True).pack(fill=tk.X, padx=10, pady=2)
        ActionButton(left, "  👁 Preview  ", self.preview_result).pack(
            fill=tk.X, padx=10, pady=2)
        ActionButton(left, "  🗑 Clear Masks  ", self.clear_all_masks).pack(
            fill=tk.X, padx=10, pady=2)
        # In _build_left_panel(), ADD to ACTIONS section after "Clear Masks" button:
        ActionButton(left, "  🖼 Remove Background  ", self.remove_background).pack(
            fill=tk.X, padx=10, pady=2)
        # In _build_left_panel(), ADD to ACTIONS section after "Remove Background" button:
        ActionButton(left, "  🔤 Remove Text  ", self.remove_text).pack(
            fill=tk.X, padx=10, pady=2)
        ActionButton(left, "  ⚙ Settings  ", self.open_settings,
                     tip="Model files, folders and defaults. Saved to settings.sec.").pack(
            fill=tk.X, padx=10, pady=2)

        # File list section
        tk.Frame(left, bg=AppTheme.BORDER, height=1).pack(fill=tk.X, padx=10, pady=10)
        tk.Label(left, text="── FILE LIST ──", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.ACCENT, font=("Segoe UI", 10, "bold")).pack(
            fill=tk.X, padx=10, pady=(0, 5))

        list_frame = tk.Frame(left, bg=AppTheme.BG_DARK)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))

        self.file_listbox = tk.Listbox(
            list_frame, bg=AppTheme.BG_DARK, fg=AppTheme.TEXT,
            selectbackground=AppTheme.ACCENT, selectforeground="white",
            font=("Segoe UI", 9), borderwidth=0, highlightthickness=0,
            activestyle="none"
        )
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                  command=self.file_listbox.yview,
                                  bg=AppTheme.SCROLLBAR, troughcolor=AppTheme.BG_DARK)
        self.file_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.pack(fill=tk.BOTH, expand=True)
        self.file_listbox.bind("<<ListboxSelect>>", self._on_file_select)

        self.file_count_label = tk.Label(
            left, text="No files loaded", bg=AppTheme.BG_PANEL,
            fg=AppTheme.TEXT_DIM, font=("Segoe UI", 9))
        self.file_count_label.pack(padx=10, pady=(0, 10))

    # ── Canvas ──────────────────────────────────────────────────────────

    def _build_canvas(self, parent):
        canvas_frame = tk.Frame(parent, bg=AppTheme.CANVAS_BG)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            canvas_frame, bg=AppTheme.CANVAS_BG,
            highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas bindings
        # self.canvas.bind("<Button-1>", self._on_canvas_click)
        # self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        # self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        # self.canvas.bind("<Button-3>", self._on_canvas_right_click)
        # self.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        # self.canvas.bind("<MouseWheel>", self._on_canvas_scroll)
        # self.canvas.bind("<Button-2>", self._on_middle_click)
        # self.canvas.bind("<B2-Motion>", self._on_middle_drag)
        # self.canvas.bind("<ButtonRelease-2>", self._on_middle_release)
        # self.canvas.bind("<Configure>", self._on_canvas_resize)
        # self.canvas.bind("<Motion>", self._on_canvas_motion)
        # In _build_canvas(), REPLACE all mouse bindings with:
        self.canvas.bind("<Button-1>", self._on_lmb_down)
        self.canvas.bind("<B1-Motion>", self._on_lmb_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_lmb_up)
        self.canvas.bind("<Button-3>", self._on_rmb_down)
        self.canvas.bind("<B3-Motion>", self._on_rmb_motion)
        self.canvas.bind("<ButtonRelease-3>", self._on_rmb_up)
        self.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        self.canvas.bind("<MouseWheel>", self._on_canvas_scroll)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        # Middle mouse also pans solo
        self.canvas.bind("<Button-2>", self._on_mmb_down)
        self.canvas.bind("<B2-Motion>", self._on_mmb_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_mmb_up)

        # Placeholder
        self.canvas.create_text(
            400, 300, text="Drop an image or use File → Open to get started",
            fill=AppTheme.TEXT_DIM, font=("Segoe UI", 14), tags="placeholder"
        )

    # ── Right Panel ─────────────────────────────────────────────────────

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=AppTheme.BG_PANEL, width=220)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
        right.pack_propagate(False)

        # Scrollable content
        canvas = tk.Canvas(right, bg=AppTheme.BG_PANEL, highlightthickness=0)
        scrollbar = tk.Scrollbar(right, orient=tk.VERTICAL, command=canvas.yview,
                                  bg=AppTheme.SCROLLBAR, troughcolor=AppTheme.BG_PANEL)
        scroll_frame = tk.Frame(canvas, bg=AppTheme.BG_PANEL)

        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=205)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(fill=tk.BOTH, expand=True)

        # Properties section
        tk.Label(scroll_frame, text="── PROPERTIES ──", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.ACCENT, font=("Segoe UI", 10, "bold")).pack(
            fill=tk.X, padx=10, pady=(10, 5))

        # Block size slider
        # In _build_right_panel(), REPLACE the block_slider creation:

        # Block size slider + custom entry
        block_frame = tk.Frame(scroll_frame, bg=AppTheme.BG_PANEL)
        block_frame.pack(fill=tk.X, padx=5, pady=2)

        self.block_slider = ModernSlider(
            block_frame, "Block Size:", from_=3, to=200,
            value=15, command=self._on_block_size_change, fmt="{:.0f}px",
            tip="Pixelate block size in pixels (bigger = coarser). Also sets the brush width for the Brush tool.")
        self.block_slider.pack(fill=tk.X, side=tk.LEFT, expand=True)

        self.block_entry = tk.Entry(
            block_frame, width=5, bg=AppTheme.ENTRY_BG, fg=AppTheme.ACCENT,
            font=("Segoe UI", 9, "bold"), insertbackground=AppTheme.TEXT,
            justify="center", borderwidth=1, relief="solid"
        )
        self.block_entry.insert(0, "15")
        self.block_entry.pack(side=tk.RIGHT, padx=(4, 5), pady=(18, 5))
        self.block_entry.bind("<Return>", self._on_block_entry)
        self.block_entry.bind("<FocusOut>", self._on_block_entry)

        # Mosaic type
        tk.Label(scroll_frame, text="Mosaic Type:", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.TEXT_DIM, font=("Segoe UI", 9)).pack(
            anchor="w", padx=15, pady=(8, 2))

        for mt in MaskType:
            rb = ttk.Radiobutton(
                scroll_frame, text=mt.value, variable=self.mask_type_var,
                value=mt.value, style="Dark.TRadiobutton",
                command=self._on_mask_type_change
            )
            rb.pack(anchor="w", padx=20)

        # Color button (for solid fill)
        self.color_frame = tk.Frame(scroll_frame, bg=AppTheme.BG_PANEL)
        self.color_frame.pack(fill=tk.X, padx=15, pady=5)
        tk.Label(self.color_frame, text="Fill Color:", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.TEXT_DIM, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.color_swatch = tk.Label(
            self.color_frame, bg="#000000", width=3, height=1,
            relief="solid", borderwidth=1, cursor="hand2")
        self.color_swatch.pack(side=tk.RIGHT, padx=5)
        self.color_swatch.bind("<Button-1>", self._pick_color)

        # Blur strength
        self.blur_slider = ModernSlider(
            scroll_frame, "Blur Strength:", from_=3, to=99,
            value=25, command=self._on_blur_change, fmt="{:.0f}",
            tip="Strength of the Gaussian Blur mask. Higher = stronger blur.")
        self.blur_slider.pack(fill=tk.X, padx=5, pady=2)

        # Opacity
        self.opacity_slider = ModernSlider(
            scroll_frame, "Opacity:", from_=0, to=100,
            value=100, command=self._on_opacity_change, fmt="{:.0f}%",
            tip="How strongly the mask covers the original. 100% = fully masked.")
        self.opacity_slider.pack(fill=tk.X, padx=5, pady=2)

        # Detection settings
        tk.Frame(scroll_frame, bg=AppTheme.BORDER, height=1).pack(
            fill=tk.X, padx=10, pady=10)
        tk.Label(scroll_frame, text="── DETECTION ──", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.ACCENT, font=("Segoe UI", 10, "bold")).pack(
            fill=tk.X, padx=10, pady=(0, 5))

        self.confidence_slider = ModernSlider(
            scroll_frame, "Confidence:", from_=10, to=100,
            value=25, fmt="{:.0f}%",
            tip="Detection sensitivity. Lower finds more faces and people but can add duplicates or false hits; higher is stricter and may miss partly hidden people. The default 25% suits most photos.")
        self.confidence_slider.pack(fill=tk.X, padx=5, pady=2)

        self.min_face_slider = ModernSlider(
            scroll_frame, "Min Face Size:", from_=10, to=200,
            value=30, fmt="{:.0f}px",
            tip="Ignore faces smaller than this many pixels. Raise it to skip tiny background faces.")
        self.min_face_slider.pack(fill=tk.X, padx=5, pady=2)

        # Faces found
        self.faces_label = tk.Label(
            scroll_frame, text="Faces Found: 0", bg=AppTheme.BG_PANEL,
            fg=AppTheme.TEXT, font=("Segoe UI", 10))
        self.faces_label.pack(fill=tk.X, padx=15, pady=(8, 2))

        # Mask list
        tk.Frame(scroll_frame, bg=AppTheme.BORDER, height=1).pack(
            fill=tk.X, padx=10, pady=10)
        tk.Label(scroll_frame, text="── MASK LIST ──", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.ACCENT, font=("Segoe UI", 10, "bold")).pack(
            fill=tk.X, padx=10, pady=(0, 5))

        self.mask_list_frame = tk.Frame(scroll_frame, bg=AppTheme.BG_PANEL)
        self.mask_list_frame.pack(fill=tk.X, padx=5)

        # Export settings
        tk.Frame(scroll_frame, bg=AppTheme.BORDER, height=1).pack(
            fill=tk.X, padx=10, pady=10)
        tk.Label(scroll_frame, text="── EXPORT ──", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.ACCENT, font=("Segoe UI", 10, "bold")).pack(
            fill=tk.X, padx=10, pady=(0, 5))

        ttk.Checkbutton(
            scroll_frame, text="Strip EXIF/Metadata",
            variable=self.strip_exif_var, style="Dark.TCheckbutton"
        ).pack(anchor="w", padx=15)

    # ── Status Bar ──────────────────────────────────────────────────────

    def _build_status_bar(self):
        status = tk.Frame(self.root, bg=AppTheme.BG_MEDIUM, height=28)
        status.pack(fill=tk.X, padx=4, pady=(0, 4))
        status.pack_propagate(False)

        self.status_label = tk.Label(
            status, text="Ready", bg=AppTheme.BG_MEDIUM,
            fg=AppTheme.TEXT, font=("Segoe UI", 9), anchor="w")
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.zoom_label = tk.Label(
            status, text="Zoom: 100%", bg=AppTheme.BG_MEDIUM,
            fg=AppTheme.TEXT_DIM, font=("Segoe UI", 9))
        self.zoom_label.pack(side=tk.LEFT, padx=10)

        self.size_label = tk.Label(
            status, text="", bg=AppTheme.BG_MEDIUM,
            fg=AppTheme.TEXT_DIM, font=("Segoe UI", 9))
        self.size_label.pack(side=tk.LEFT, padx=10)

        self.mask_count_label = tk.Label(
            status, text="Masks: 0", bg=AppTheme.BG_MEDIUM,
            fg=AppTheme.TEXT_DIM, font=("Segoe UI", 9))
        self.mask_count_label.pack(side=tk.LEFT, padx=10)

        self.tool_label = tk.Label(
            status, text="Tool: Select", bg=AppTheme.BG_MEDIUM,
            fg=AppTheme.ACCENT, font=("Segoe UI", 9))
        self.tool_label.pack(side=tk.RIGHT, padx=10)

        self.coord_label = tk.Label(
            status, text="", bg=AppTheme.BG_MEDIUM,
            fg=AppTheme.TEXT_DIM, font=("Segoe UI", 9))
        self.coord_label.pack(side=tk.RIGHT, padx=10)

    # ── Keyboard Shortcuts ──────────────────────────────────────────────

    def _bind_shortcuts(self):
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-O>", lambda e: self.open_folder())
        self.root.bind("<Control-s>", lambda e: self.save_image())
        self.root.bind("<Control-S>", lambda e: self.save_as())
        self.root.bind("<Control-e>", lambda e: self.export_image())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-d>", lambda e: self.auto_detect())
        self.root.bind("<Control-0>", lambda e: self.zoom_fit())
        self.root.bind("<Control-1>", lambda e: self.set_zoom(1.0))
        self.root.bind("<plus>", lambda e: self.zoom_in())
        self.root.bind("<equal>", lambda e: self.zoom_in())
        self.root.bind("<minus>", lambda e: self.zoom_out())
        self.root.bind("<Delete>", lambda e: self.delete_selected_mask())
        self.root.bind("<Escape>", lambda e: self._cancel_drawing())
        self.root.bind("<Return>", lambda e: self._finish_polygon())
        # In _bind_shortcuts(), ADD:
        self.root.bind("c", lambda e: self.set_tool(ToolType.CROP))
        # In _bind_shortcuts(), ADD these lines:
        self.root.bind("<Control-v>", lambda e: self.paste_from_clipboard())
        self.root.bind("<Command-v>", lambda e: self.paste_from_clipboard())  # macOS

        self.root.bind("r", lambda e: self.set_tool(ToolType.RECTANGLE))
        self.root.bind("p", lambda e: self.set_tool(ToolType.POLYGON))
        self.root.bind("f", lambda e: self.set_tool(ToolType.FREEHAND))
        self.root.bind("b", lambda e: self.set_tool(ToolType.BRUSH))
        self.root.bind("v", lambda e: self.set_tool(ToolType.SELECT))
        self.root.bind("d", lambda e: self.set_tool(ToolType.AUTO_DETECT))

    # ── Tool Management ─────────────────────────────────────────────────

    def set_tool(self, tool: ToolType):
        if tool == ToolType.AUTO_DETECT:
            self.auto_detect()
            return
        if tool == ToolType.PEOPLE:
            self.extract_humans()
            return

        self._cancel_drawing()
        self.current_tool = tool

        for t, btn in self.tool_buttons.items():
            btn.set_active(t == tool)

        tool_names = {
            ToolType.SELECT: "Select",
            ToolType.RECTANGLE: "Rectangle",
            ToolType.POLYGON: "Polygon",
            ToolType.FREEHAND: "Freehand",
            ToolType.BRUSH: "Brush",
            # In set_tool(), ADD to tool_names dict:
            ToolType.CROP: "Crop",
        }
        self.tool_label.config(text=f"Tool: {tool_names.get(tool, 'Select')}")

        cursors = {
            ToolType.SELECT: "arrow",
            ToolType.RECTANGLE: "crosshair",
            ToolType.POLYGON: "crosshair",
            ToolType.FREEHAND: "pencil",
            ToolType.BRUSH: "circle",
            # In set_tool(), ADD to cursors dict:
            ToolType.CROP: "crosshair",
        }
        self.canvas.config(cursor=cursors.get(tool, "arrow"))
        self._update_status(f"Tool: {tool_names.get(tool, 'Select')}")

    # ── Coordinate Conversion ───────────────────────────────────────────

    def _canvas_to_image(self, cx, cy):
        """Convert canvas coordinates to image coordinates."""
        if self.current_image is None:
            return (0, 0)
        ix = int((cx - self.pan_offset[0]) / self.zoom_level)
        iy = int((cy - self.pan_offset[1]) / self.zoom_level)
        h, w = self.current_image.shape[:2]
        ix = max(0, min(ix, w - 1))
        iy = max(0, min(iy, h - 1))
        return (ix, iy)

    def _image_to_canvas(self, ix, iy):
        """Convert image coordinates to canvas coordinates."""
        cx = ix * self.zoom_level + self.pan_offset[0]
        cy = iy * self.zoom_level + self.pan_offset[1]
        return (cx, cy)

    # ADD this new method to IdentityShieldApp class:
    def paste_from_clipboard(self):
        """Paste image from system clipboard (Windows/macOS)."""
        try:
            from PIL import ImageGrab
            clip_img = ImageGrab.grabclipboard()
            if clip_img is None:
                messagebox.showinfo("Clipboard", "No image found in clipboard.")
                return
            if isinstance(clip_img, list):
                # File paths on clipboard
                for f in clip_img:
                    if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}:
                        self._load_image(f)
                        self.file_list = [f]
                        self.current_file_index = 0
                        self._update_file_list()
                        return
                messagebox.showinfo("Clipboard", "No valid image file in clipboard.")
                return
            # PIL Image object
            rgb = np.array(clip_img.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self.current_image = bgr
            self.original_image = bgr.copy()
            self.mask_regions.clear()
            self.selected_region = None
            self.history.clear()
            self.region_counter = 0
            self.file_list = []
            self.current_file_index = -1
            self._update_file_list()
            self.zoom_fit()
            self._update_status("Pasted image from clipboard")
            self.root.title("🛡️ IdentityShield Pro - [Clipboard Image]")
        except ImportError:
            messagebox.showerror("Error", "Clipboard paste requires Pillow with ImageGrab support.")
        except Exception as e:
            messagebox.showerror("Clipboard Error", str(e))

    # ── Canvas Events ───────────────────────────────────────────────────

    # def _on_canvas_click(self, event):
    #     if self.current_image is None:
    #         return
    #
    #     ix, iy = self._canvas_to_image(event.x, event.y)
    #
    #     if self.current_tool == ToolType.SELECT:
    #         self._select_mask_at(ix, iy)
    #     elif self.current_tool == ToolType.RECTANGLE:
    #         self.drawing = True
    #         self.draw_start = (ix, iy)
    #     elif self.current_tool == ToolType.POLYGON:
    #         self.draw_points.append((ix, iy))
    #         self._draw_polygon_preview()
    #     elif self.current_tool == ToolType.FREEHAND:
    #         self.drawing = True
    #         self.draw_points = [(ix, iy)]
    #     elif self.current_tool == ToolType.BRUSH:
    #         self.drawing = True
    #         self.draw_points = [(ix, iy)]
    #     # In _on_canvas_click(), ADD case:
    #     elif self.current_tool == ToolType.CROP:
    #         self.drawing = True
    #         self.draw_start = (ix, iy)
    #
    # def _on_canvas_drag(self, event):
    #     if self.current_image is None:
    #         return
    #
    #     ix, iy = self._canvas_to_image(event.x, event.y)
    #
    #     if self.current_tool == ToolType.RECTANGLE and self.drawing:
    #         self._draw_rect_preview(self.draw_start, (ix, iy))
    #     elif self.current_tool == ToolType.FREEHAND and self.drawing:
    #         self.draw_points.append((ix, iy))
    #         self._draw_freehand_preview()
    #     elif self.current_tool == ToolType.BRUSH and self.drawing:
    #         self.draw_points.append((ix, iy))
    #         self._draw_brush_preview()
    #
    # def _on_canvas_release(self, event):
    #     if self.current_image is None:
    #         return
    #
    #     ix, iy = self._canvas_to_image(event.x, event.y)
    #
    #     if self.current_tool == ToolType.RECTANGLE and self.drawing:
    #         self.drawing = False
    #         if self.draw_start and (abs(ix - self.draw_start[0]) > 3 or
    #                                  abs(iy - self.draw_start[1]) > 3):
    #             self._create_rect_mask(self.draw_start, (ix, iy))
    #         self.draw_start = None
    #         self._clear_temp_canvas()
    #
    #     elif self.current_tool == ToolType.FREEHAND and self.drawing:
    #         self.drawing = False
    #         if len(self.draw_points) > 5:
    #             self._create_freehand_mask(self.draw_points)
    #         self.draw_points = []
    #         self._clear_temp_canvas()
    #
    #     elif self.current_tool == ToolType.BRUSH and self.drawing:
    #         self.drawing = False
    #         if len(self.draw_points) > 3:
    #             self._create_brush_mask(self.draw_points)
    #         self.draw_points = []
    #         self._clear_temp_canvas()
    #
    # # In _on_canvas_release(), ADD case before the final elif blocks:
    #     elif self.current_tool == ToolType.CROP and self.drawing:
    #     self.drawing = False
    #     if self.draw_start and (abs(ix - self.draw_start[0]) > 10 or
    #                             abs(iy - self.draw_start[1]) > 10):
    #         self._apply_crop(self.draw_start, (ix, iy))
    #     self.draw_start = None
    #     self._clear_temp_canvas()
    #
    # def _on_canvas_double_click(self, event):
    #     if self.current_tool == ToolType.POLYGON and len(self.draw_points) >= 3:
    #         self._finish_polygon()
    #
    # def _on_canvas_right_click(self, event):
    #     if self.current_tool == ToolType.POLYGON and self.draw_points:
    #         self._cancel_drawing()
    #         return
    #
    #     ix, iy = self._canvas_to_image(event.x, event.y)
    #     region = self._find_region_at(ix, iy)
    #     if region:
    #         self._show_context_menu(event, region)
    #
    # def _on_canvas_scroll(self, event):
    #     if event.delta > 0:
    #         self.zoom_in(center=(event.x, event.y))
    #     else:
    #         self.zoom_out(center=(event.x, event.y))
    #
    # def _on_middle_click(self, event):
    #     self.is_panning = True
    #     self.canvas_drag_start = (event.x, event.y)
    #     self.canvas.config(cursor="fleur")
    #
    # def _on_middle_drag(self, event):
    #     if self.is_panning and self.canvas_drag_start:
    #         dx = event.x - self.canvas_drag_start[0]
    #         dy = event.y - self.canvas_drag_start[1]
    #         self.pan_offset[0] += dx
    #         self.pan_offset[1] += dy
    #         self.canvas_drag_start = (event.x, event.y)
    #         self._refresh_canvas()
    #
    # def _on_middle_release(self, event):
    #     self.is_panning = False
    #     self.canvas_drag_start = None
    #     cursors = {
    #         ToolType.SELECT: "arrow",
    #         ToolType.RECTANGLE: "crosshair",
    #         ToolType.POLYGON: "crosshair",
    #         ToolType.FREEHAND: "pencil",
    #         ToolType.BRUSH: "circle",
    #     }
    #     self.canvas.config(cursor=cursors.get(self.current_tool, "arrow"))

    def _start_pan(self, event):
        """Initiate pan mode."""
        self.is_panning = True
        self.both_btn_pan = True
        self.canvas_drag_start = (event.x, event.y)
        self._pan_initial_cursor = self.canvas.cget("cursor")
        self.canvas.config(cursor="fleur")
        # Cancel any in-progress drawing
        if self.drawing:
            self.drawing = False
            self.draw_start = None
            self._clear_temp_canvas()

    def _do_pan(self, event):
        """Execute pan movement."""
        if self.is_panning and self.canvas_drag_start:
            dx = event.x - self.canvas_drag_start[0]
            dy = event.y - self.canvas_drag_start[1]
            self.pan_offset[0] += dx
            self.pan_offset[1] += dy
            self.canvas_drag_start = (event.x, event.y)
            self._refresh_canvas()

    def _stop_pan(self):
        """End pan mode."""
        self.is_panning = False
        self.both_btn_pan = False
        self.canvas_drag_start = None
        cursors = {
            ToolType.SELECT: "arrow",
            ToolType.RECTANGLE: "crosshair",
            ToolType.POLYGON: "crosshair",
            ToolType.FREEHAND: "pencil",
            ToolType.BRUSH: "circle",
            ToolType.CROP: "crosshair",
        }
        self.canvas.config(cursor=cursors.get(self.current_tool, "arrow"))

    # ── LMB handlers ──

    def _on_lmb_down(self, event):
        self.lmb_down = True
        if self.rmb_down:
            # Both buttons pressed → pan
            self._start_pan(event)
            return
        # Normal LMB behavior (delegated)
        if self.current_image is None:
            return
        self._lmb_origin = (event.x, event.y)
        self._on_canvas_click(event)

    def _on_lmb_motion(self, event):
        if self.both_btn_pan:
            self._do_pan(event)
            return
        if self.rmb_down and self.lmb_down:
            self._start_pan(event)
            return
        self._on_canvas_drag(event)

    def _on_lmb_up(self, event):
        self.lmb_down = False
        if self.both_btn_pan:
            self._stop_pan()
            return
        self._on_canvas_release(event)

    # ── RMB handlers ──

    def _on_rmb_down(self, event):
        self.rmb_down = True
        self._rmb_origin = (event.x, event.y)
        if self.lmb_down:
            # Both buttons pressed → pan
            self._start_pan(event)
            return

    def _on_rmb_motion(self, event):
        if self.both_btn_pan:
            self._do_pan(event)
            return
        if self.lmb_down and self.rmb_down:
            self._start_pan(event)

    def _on_rmb_up(self, event):
        self.rmb_down = False
        if self.both_btn_pan:
            self._stop_pan()
            return
        # If it was a simple right-click (no drag, no LMB) → context menu
        if hasattr(self, '_rmb_origin'):
            dx = abs(event.x - self._rmb_origin[0])
            dy = abs(event.y - self._rmb_origin[1])
            if dx < 4 and dy < 4:
                # Clean right-click → context menu or cancel polygon
                if self.current_tool == ToolType.POLYGON and self.draw_points:
                    self._cancel_drawing()
                elif self.current_image is not None:
                    ix, iy = self._canvas_to_image(event.x, event.y)
                    region = self._find_region_at(ix, iy)
                    if region:
                        self._show_context_menu(event, region)

    # ── MMB handlers (middle mouse solo pan) ──

    def _on_mmb_down(self, event):
        self.is_panning = True
        self.canvas_drag_start = (event.x, event.y)
        self._pan_initial_cursor = self.canvas.cget("cursor")
        self.canvas.config(cursor="fleur")

    def _on_mmb_drag(self, event):
        self._do_pan(event)

    def _on_mmb_up(self, event):
        self._stop_pan()

    # ── Restored original logic (called by LMB handlers) ──

    def _on_canvas_click(self, event):
        if self.current_image is None:
            return
        ix, iy = self._canvas_to_image(event.x, event.y)

        if self.current_tool == ToolType.SELECT:
            self._select_mask_at(ix, iy)
        elif self.current_tool == ToolType.RECTANGLE:
            self.drawing = True
            self.draw_start = (ix, iy)
        elif self.current_tool == ToolType.POLYGON:
            self.draw_points.append((ix, iy))
            self._draw_polygon_preview()
        elif self.current_tool == ToolType.FREEHAND:
            self.drawing = True
            self.draw_points = [(ix, iy)]
        elif self.current_tool == ToolType.BRUSH:
            self.drawing = True
            self.draw_points = [(ix, iy)]
        elif self.current_tool == ToolType.CROP:
            self.drawing = True
            self.draw_start = (ix, iy)

    # ADD this method to IdentityShieldApp, alongside the other canvas event handlers:

    def _on_canvas_double_click(self, event):
        """Handle double-click — finish polygon or select mask."""
        if self.both_btn_pan or self.is_panning:
            return
        if self.current_tool == ToolType.POLYGON and len(self.draw_points) >= 3:
            self._finish_polygon()
        elif self.current_tool == ToolType.SELECT and self.current_image is not None:
            # Double-click on select = zoom to fit
            self.zoom_fit()

    # ADD this method to IdentityShieldApp, alongside the other canvas event handlers:

    def _on_canvas_scroll(self, event):
        """Handle mouse wheel zoom."""
        if self.both_btn_pan or self.is_panning:
            return
        if event.delta > 0:
            self.zoom_in(center=(event.x, event.y))
        else:
            self.zoom_out(center=(event.x, event.y))

    def _on_canvas_drag(self, event):
        if self.current_image is None:
            return
        ix, iy = self._canvas_to_image(event.x, event.y)

        if self.current_tool == ToolType.RECTANGLE and self.drawing:
            self._draw_rect_preview(self.draw_start, (ix, iy))
        elif self.current_tool == ToolType.FREEHAND and self.drawing:
            self.draw_points.append((ix, iy))
            self._draw_freehand_preview()
        elif self.current_tool == ToolType.BRUSH and self.drawing:
            self.draw_points.append((ix, iy))
            self._draw_brush_preview()
        elif self.current_tool == ToolType.CROP and self.drawing:
            self._draw_crop_preview(self.draw_start, (ix, iy))

    def _on_canvas_release(self, event):
        if self.current_image is None:
            return
        ix, iy = self._canvas_to_image(event.x, event.y)

        if self.current_tool == ToolType.RECTANGLE and self.drawing:
            self.drawing = False
            if self.draw_start and (abs(ix - self.draw_start[0]) > 3 or
                                    abs(iy - self.draw_start[1]) > 3):
                self._create_rect_mask(self.draw_start, (ix, iy))
            self.draw_start = None
            self._clear_temp_canvas()

        elif self.current_tool == ToolType.FREEHAND and self.drawing:
            self.drawing = False
            if len(self.draw_points) > 5:
                self._create_freehand_mask(self.draw_points)
            self.draw_points = []
            self._clear_temp_canvas()

        elif self.current_tool == ToolType.BRUSH and self.drawing:
            self.drawing = False
            if len(self.draw_points) > 3:
                self._create_brush_mask(self.draw_points)
            self.draw_points = []
            self._clear_temp_canvas()

        elif self.current_tool == ToolType.CROP and self.drawing:
            self.drawing = False
            if self.draw_start and (abs(ix - self.draw_start[0]) > 10 or
                                    abs(iy - self.draw_start[1]) > 10):
                self._apply_crop(self.draw_start, (ix, iy))
            self.draw_start = None
            self._clear_temp_canvas()

    def _on_canvas_resize(self, event):
        if self.current_image is not None:
            self._refresh_canvas()

    def _on_canvas_motion(self, event):
        if self.current_image is not None:
            ix, iy = self._canvas_to_image(event.x, event.y)
            self.coord_label.config(text=f"({ix}, {iy})")

    # ── Drawing Previews ────────────────────────────────────────────────

    def _clear_temp_canvas(self):
        for item in self.temp_canvas_items:
            self.canvas.delete(item)
        self.temp_canvas_items.clear()

    # ADD these new methods to IdentityShieldApp:

    def _draw_crop_preview(self, start, end):
        """Draw crop rectangle with dimmed outside area."""
        self._clear_temp_canvas()
        cx1, cy1 = self._image_to_canvas(*start)
        cx2, cy2 = self._image_to_canvas(*end)

        # Dim overlay outside crop area
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        left = min(cx1, cx2)
        top = min(cy1, cy2)
        right = max(cx1, cx2)
        bottom = max(cy1, cy2)

        # Four dark rectangles around crop region
        for coords in [
            (0, 0, cw, top),  # top
            (0, bottom, cw, ch),  # bottom
            (0, top, left, bottom),  # left
            (right, top, cw, bottom),  # right
        ]:
            item = self.canvas.create_rectangle(
                *coords, fill="black", outline="", stipple="gray50"
            )
            self.temp_canvas_items.append(item)

        # Crop border
        item = self.canvas.create_rectangle(
            cx1, cy1, cx2, cy2,
            outline="#00ff88", width=2, dash=(6, 3)
        )
        self.temp_canvas_items.append(item)

        # Size label
        crop_w = abs(end[0] - start[0])
        crop_h = abs(end[1] - start[1])
        item = self.canvas.create_text(
            (cx1 + cx2) / 2, min(cy1, cy2) - 12,
            text=f"{crop_w}×{crop_h}", fill="#00ff88",
            font=("Segoe UI", 9, "bold")
        )
        self.temp_canvas_items.append(item)

    def _apply_crop(self, start, end):
        """Crop the image to selection."""
        x1, y1 = min(start[0], end[0]), min(start[1], end[1])
        x2, y2 = max(start[0], end[0]), max(start[1], end[1])
        h, w = self.current_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 5 or y2 - y1 < 5:
            return

        if messagebox.askyesno("Crop", f"Crop image to {x2 - x1}×{y2 - y1}?\n"
                                       "This will clear all masks."):
            self.current_image = self.current_image[y1:y2, x1:x2].copy()
            self.original_image = self.current_image.copy()
            self.mask_regions.clear()
            self.selected_region = None
            self.history.clear()
            self._update_mask_list()
            self.zoom_fit()
            self._update_status(f"Cropped to {x2 - x1}×{y2 - y1}")

    def _draw_rect_preview(self, start, end):
        self._clear_temp_canvas()
        cx1, cy1 = self._image_to_canvas(*start)
        cx2, cy2 = self._image_to_canvas(*end)
        item = self.canvas.create_rectangle(
            cx1, cy1, cx2, cy2,
            outline=AppTheme.ACCENT, width=2, dash=(5, 3),
            fill=AppTheme.ACCENT, stipple="gray25"
        )
        self.temp_canvas_items.append(item)

    def _draw_polygon_preview(self):
        self._clear_temp_canvas()
        if len(self.draw_points) < 2:
            if self.draw_points:
                cx, cy = self._image_to_canvas(*self.draw_points[0])
                item = self.canvas.create_oval(
                    cx-4, cy-4, cx+4, cy+4,
                    fill=AppTheme.ACCENT, outline=""
                )
                self.temp_canvas_items.append(item)
            return

        canvas_points = []
        for pt in self.draw_points:
            canvas_points.extend(self._image_to_canvas(*pt))

        item = self.canvas.create_polygon(
            *canvas_points, outline=AppTheme.ACCENT, width=2,
            fill=AppTheme.ACCENT, stipple="gray25"
        )
        self.temp_canvas_items.append(item)

        # Draw vertices
        for pt in self.draw_points:
            cx, cy = self._image_to_canvas(*pt)
            item = self.canvas.create_oval(
                cx-3, cy-3, cx+3, cy+3,
                fill=AppTheme.ACCENT, outline="white"
            )
            self.temp_canvas_items.append(item)

    def _draw_freehand_preview(self):
        self._clear_temp_canvas()
        if len(self.draw_points) < 2:
            return
        canvas_points = []
        for pt in self.draw_points:
            canvas_points.extend(self._image_to_canvas(*pt))
        item = self.canvas.create_line(
            *canvas_points, fill=AppTheme.ACCENT, width=2, smooth=True
        )
        self.temp_canvas_items.append(item)

    def _draw_brush_preview(self):
        self._clear_temp_canvas()
        brush_size = max(5, int(self.block_slider.get()))
        for pt in self.draw_points[::2]:  # skip every other for performance
            cx, cy = self._image_to_canvas(*pt)
            r = brush_size * self.zoom_level / 2
            item = self.canvas.create_oval(
                cx-r, cy-r, cx+r, cy+r,
                fill=AppTheme.ACCENT, outline="", stipple="gray50"
            )
            self.temp_canvas_items.append(item)

    # ── Mask Creation ───────────────────────────────────────────────────

    def _next_region_id(self):
        self.region_counter += 1
        return f"mask_{self.region_counter}"

    def _get_current_mask_type(self) -> MaskType:
        val = self.mask_type_var.get()
        for mt in MaskType:
            if mt.value == val:
                return mt
        return MaskType.PIXELATE

    def _create_rect_mask(self, start, end):
        self.history.push(self.mask_regions[:])
        x1, y1 = min(start[0], end[0]), min(start[1], end[1])
        x2, y2 = max(start[0], end[0]), max(start[1], end[1])

        region = MaskRegion(
            region_id=self._next_region_id(),
            tool_type=ToolType.RECTANGLE,
            points=[(x1, y1), (x2, y2)],
            mask_type=self._get_current_mask_type(),
            block_size=int(self.block_slider.get()),
            blur_strength=int(self.blur_slider.get()),
            fill_color=self.fill_color,
            opacity=self.opacity_slider.get() / 100.0,
            label=f"Rect #{self.region_counter}"
        )
        self.mask_regions.append(region)
        self._update_mask_list()
        self._refresh_canvas()
        self._update_status(f"Rectangle mask created")

    def _create_freehand_mask(self, points):
        self.history.push(self.mask_regions[:])
        region = MaskRegion(
            region_id=self._next_region_id(),
            tool_type=ToolType.FREEHAND,
            points=points[:],
            mask_type=self._get_current_mask_type(),
            block_size=int(self.block_slider.get()),
            blur_strength=int(self.blur_slider.get()),
            fill_color=self.fill_color,
            opacity=self.opacity_slider.get() / 100.0,
            label=f"Freehand #{self.region_counter}"
        )
        self.mask_regions.append(region)
        self._update_mask_list()
        self._refresh_canvas()
        self._update_status(f"Freehand mask created")

    def _create_brush_mask(self, points):
        self.history.push(self.mask_regions[:])
        brush_size = max(5, int(self.block_slider.get()))
        expanded = []
        for px, py in points:
            for angle in range(0, 360, 30):
                r = brush_size / 2
                ex = int(px + r * np.cos(np.radians(angle)))
                ey = int(py + r * np.sin(np.radians(angle)))
                expanded.append((ex, ey))

        # Create convex hull-ish from brush strokes
        if len(expanded) >= 3:
            pts_array = np.array(expanded, dtype=np.int32)
            hull = cv2.convexHull(pts_array)
            hull_points = [(int(p[0][0]), int(p[0][1])) for p in hull]
        else:
            hull_points = expanded

        region = MaskRegion(
            region_id=self._next_region_id(),
            tool_type=ToolType.BRUSH,
            points=hull_points,
            mask_type=self._get_current_mask_type(),
            block_size=int(self.block_slider.get()),
            blur_strength=int(self.blur_slider.get()),
            fill_color=self.fill_color,
            opacity=self.opacity_slider.get() / 100.0,
            label=f"Brush #{self.region_counter}"
        )
        self.mask_regions.append(region)
        self._update_mask_list()
        self._refresh_canvas()
        self._update_status(f"Brush mask created")

    def _finish_polygon(self):
        if len(self.draw_points) >= 3:
            self.history.push(self.mask_regions[:])
            region = MaskRegion(
                region_id=self._next_region_id(),
                tool_type=ToolType.POLYGON,
                points=self.draw_points[:],
                mask_type=self._get_current_mask_type(),
                block_size=int(self.block_slider.get()),
                blur_strength=int(self.blur_slider.get()),
                fill_color=self.fill_color,
                opacity=self.opacity_slider.get() / 100.0,
                label=f"Polygon #{self.region_counter}"
            )
            self.mask_regions.append(region)
            self._update_mask_list()
            self._update_status(f"Polygon mask created")

        self.draw_points = []
        self._clear_temp_canvas()
        self._refresh_canvas()

    def _cancel_drawing(self):
        self.drawing = False
        self.draw_start = None
        self.draw_points = []
        self._clear_temp_canvas()

    # ── Selection ───────────────────────────────────────────────────────

    def _find_region_at(self, ix, iy) -> Optional[MaskRegion]:
        for region in reversed(self.mask_regions):
            if not region.visible:
                continue
            mask = ImageProcessor.create_mask_from_region(region, self.current_image.shape)
            if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]:
                if mask[iy, ix] > 0:
                    return region
        return None

    def _select_mask_at(self, ix, iy):
        region = self._find_region_at(ix, iy)
        self.selected_region = region
        self._update_mask_list()
        self._refresh_canvas()
        if region:
            self._update_status(f"Selected: {region.label}")

    # ── Context Menu ────────────────────────────────────────────────────

    def _show_context_menu(self, event, region):
        menu = tk.Menu(self.root, tearoff=0, bg=AppTheme.BG_MEDIUM,
                      fg=AppTheme.TEXT, activebackground=AppTheme.ACCENT,
                      font=("Segoe UI", 10))
        menu.add_command(label=f"🎯 {region.label}", state="disabled")
        menu.add_separator()
        menu.add_command(label="🗑 Delete Mask",
                        command=lambda: self._delete_region(region))
        menu.add_command(label="👁 Toggle Visibility",
                        command=lambda: self._toggle_region_visibility(region))
        menu.add_separator()

        type_menu = tk.Menu(menu, tearoff=0, bg=AppTheme.BG_MEDIUM,
                           fg=AppTheme.TEXT, activebackground=AppTheme.ACCENT)
        for mt in MaskType:
            type_menu.add_command(
                label=mt.value,
                command=lambda m=mt, r=region: self._change_region_type(r, m)
            )
        menu.add_cascade(label="Change Type", menu=type_menu)

        menu.add_separator()
        menu.add_command(label="📋 Duplicate",
                        command=lambda: self._duplicate_region(region))

        menu.tk_popup(event.x_root, event.y_root)

    def _delete_region(self, region):
        self.history.push(self.mask_regions[:])
        self.mask_regions = [r for r in self.mask_regions if r.region_id != region.region_id]
        if self.selected_region and self.selected_region.region_id == region.region_id:
            self.selected_region = None
        self._update_mask_list()
        self._refresh_canvas()

    def _toggle_region_visibility(self, region):
        region.visible = not region.visible
        self._update_mask_list()
        self._refresh_canvas()

    def _change_region_type(self, region, mask_type):
        self.history.push(self.mask_regions[:])
        region.mask_type = mask_type
        self._refresh_canvas()

    def _duplicate_region(self, region):
        self.history.push(self.mask_regions[:])
        new_region = copy.deepcopy(region)
        new_region.region_id = self._next_region_id()
        new_region.label = f"{region.label} (copy)"
        # Offset slightly
        new_region.points = [(p[0]+10, p[1]+10) for p in new_region.points]
        self.mask_regions.append(new_region)
        self._update_mask_list()
        self._refresh_canvas()

    # ── Canvas Rendering ────────────────────────────────────────────────

    def _refresh_canvas(self):
        if self.current_image is None:
            return

        self.canvas.delete("all")

        # Apply all visible masks to create preview
        img = self.current_image.copy()
        for region in self.mask_regions:
            if region.visible:
                img = ImageProcessor.apply_mask_region(img, region)

        # Convert to RGB for PIL
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Apply zoom
        h, w = img.shape[:2]
        new_w = max(1, int(w * self.zoom_level))
        new_h = max(1, int(h * self.zoom_level))
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        self.display_image = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(
            self.pan_offset[0], self.pan_offset[1],
            anchor="nw", image=self.display_image
        )

        # Draw mask region outlines
        for region in self.mask_regions:
            if not region.visible:
                continue
            is_selected = (self.selected_region and
                          self.selected_region.region_id == region.region_id)
            color = "#00ff88" if is_selected else AppTheme.ACCENT
            width = 3 if is_selected else 1
            dash = () if is_selected else (4, 2)

            if region.tool_type in (ToolType.RECTANGLE, ToolType.AUTO_DETECT):
                if len(region.points) >= 2:
                    cx1, cy1 = self._image_to_canvas(*region.points[0])
                    cx2, cy2 = self._image_to_canvas(*region.points[1])
                    self.canvas.create_rectangle(
                        cx1, cy1, cx2, cy2,
                        outline=color, width=width, dash=dash
                    )
            else:
                if len(region.points) >= 3:
                    canvas_pts = []
                    for pt in region.points:
                        canvas_pts.extend(self._image_to_canvas(*pt))
                    self.canvas.create_polygon(
                        *canvas_pts, outline=color, fill="",
                        width=width, dash=dash
                    )

        self._update_info()

    def _update_info(self):
        self.mask_count_label.config(text=f"Masks: {len(self.mask_regions)}")
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_level * 100)}%")
        if self.current_image is not None:
            h, w = self.current_image.shape[:2]
            self.size_label.config(text=f"Size: {w}×{h}")

    # ── Mask List UI ────────────────────────────────────────────────────

    def _update_mask_list(self):
        for widget in self.mask_list_frame.winfo_children():
            widget.destroy()

        for region in self.mask_regions:
            is_selected = (self.selected_region and
                          self.selected_region.region_id == region.region_id)
            bg = AppTheme.BG_LIGHT if is_selected else AppTheme.BG_DARK

            row = tk.Frame(self.mask_list_frame, bg=bg)
            row.pack(fill=tk.X, pady=1)

            # Visibility toggle
            vis_text = "👁" if region.visible else "👁‍🗨"
            vis_btn = tk.Label(row, text=vis_text, bg=bg, fg=AppTheme.TEXT,
                               cursor="hand2", font=("Segoe UI", 9))
            vis_btn.pack(side=tk.LEFT, padx=2)
            vis_btn.bind("<Button-1>",
                        lambda e, r=region: self._toggle_region_visibility(r))

            # Label
            label = tk.Label(row, text=region.label, bg=bg,
                            fg=AppTheme.TEXT if region.visible else AppTheme.TEXT_DIM,
                            font=("Segoe UI", 9), anchor="w")
            label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            label.bind("<Button-1>",
                      lambda e, r=region: self._select_region(r))

            # Delete
            del_btn = tk.Label(row, text="✕", bg=bg, fg=AppTheme.ACCENT,
                               cursor="hand2", font=("Segoe UI", 9))
            del_btn.pack(side=tk.RIGHT, padx=2)
            del_btn.bind("<Button-1>",
                        lambda e, r=region: self._delete_region(r))

        self.faces_label.config(
            text=f"Faces Found: {sum(1 for r in self.mask_regions if r.tool_type == ToolType.AUTO_DETECT)}")

    def _select_region(self, region):
        self.selected_region = region
        self._update_mask_list()
        self._refresh_canvas()

    # ── File Operations ─────────────────────────────────────────────────
    # ── PDF Operations ─────────────────────────────────────────────────
    def _remember_pdf_masks(self):
        if self.pdf_document is not None:
            self.pdf_page_masks[self.pdf_page_index] = copy.deepcopy(self.mask_regions)

    def _load_pdf_page(self, index):
        page = self.pdf_document.load_page(index)
        pix = page.get_pixmap(matrix=fitz.Matrix(self.pdf_render_scale, self.pdf_render_scale), alpha=False)
        data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        self.current_image = cv2.cvtColor(data, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
        self.original_image = self.current_image.copy()
        self.pdf_page_index = index
        self.pdf_page_sizes[index] = (pix.width, pix.height)
        self.mask_regions = copy.deepcopy(self.pdf_page_masks.get(index, []))
        self.selected_region = None
        self.history.clear()
        self.region_counter = len(self.mask_regions)
        self.zoom_fit()
        self._update_mask_list()
        self._update_status(f"PDF page {index + 1} of {self.pdf_document.page_count}")
        self.root.title(f"IdentityShield Pro - {os.path.basename(self.pdf_path)} (page {index + 1})")

    def open_pdf(self, path=None):
        if fitz is None:
            messagebox.showerror("PDF support unavailable", "Install it with: sudo apt install -y python3-fitz")
            return
        if path is None:
            path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if not path:
            return
        try:
            if self.pdf_document is not None:
                self.pdf_document.close()
            self.pdf_document = fitz.open(path)
            if not self.pdf_document.page_count:
                raise ValueError("The PDF has no pages.")
            self.pdf_path, self.pdf_page_index = path, 0
            self.pdf_page_masks, self.pdf_page_sizes = {}, {}
            self.file_list, self.current_file_index = [path], 0
            self._update_file_list()
            self._load_pdf_page(0)
            self._add_recent(path)
        except Exception as e:
            self.pdf_document, self.pdf_path = None, None
            messagebox.showerror("PDF Error", str(e))

    def save_pdf_as(self):
        if self.pdf_document is None:
            messagebox.showinfo("No PDF", "Open a PDF first.")
            return
        self._remember_pdf_masks()
        pages = {i: r for i, r in self.pdf_page_masks.items() if r}
        if not pages:
            messagebox.showwarning("No masks", "Draw or auto-detect at least one mask first.")
            return
        source = Path(self.pdf_path)
        output = filedialog.asksaveasfilename(defaultextension=".pdf", initialdir=source.parent, initialfile=f"{source.stem}_redacted.pdf", filetypes=[("PDF files", "*.pdf")])
        if not output:
            return
        if os.path.abspath(output) == os.path.abspath(self.pdf_path):
            messagebox.showerror("Choose a new file", "The original PDF is never overwritten.")
            return
        if not messagebox.askyesno("Permanent redaction", "Save a new PDF with permanent black redactions?\n\nThe original will remain unchanged."):
            return
        doc = None
        try:
            doc = fitz.open(self.pdf_path)
            for index, regions in pages.items():
                page = doc.load_page(index)
                width, height = self.pdf_page_sizes[index]
                for region in regions:
                    if region.visible:
                        x1, y1, x2, y2 = region.get_bbox()
                        rect = fitz.Rect(x1 * page.rect.width / width, y1 * page.rect.height / height, x2 * page.rect.width / width, y2 * page.rect.height / height)
                        if not rect.is_empty:
                            page.add_redact_annot(rect, fill=(0, 0, 0))
                page.apply_redactions()
            doc.save(output, garbage=4, deflate=True)
            self._update_status(f"Saved redacted PDF: {os.path.basename(output)}")
        except Exception as e:
            messagebox.showerror("PDF Save Error", str(e))
        finally:
            if doc is not None:
                doc.close()


    def redact_pdf_terms(self):
        if self.pdf_document is None or not self.pdf_path:
            messagebox.showinfo("No PDF", "Open a PDF first.")
            return
        query = simpledialog.askstring("Redact all PDF pages", "Text or pattern to redact:", parent=self.root)
        if not query:
            return
        use_regex = messagebox.askyesno("Match type", "Use a regular expression?\n\nYes: match each PDF text token with a regex.\nNo: find the exact text phrase.", parent=self.root)
        pattern = None
        if use_regex:
            try:
                import re
                pattern = re.compile(query, re.IGNORECASE)
            except re.error as e:
                messagebox.showerror("Invalid regular expression", str(e))
                return
        source = Path(self.pdf_path)
        output = filedialog.asksaveasfilename(defaultextension=".pdf", initialdir=source.parent, initialfile=f"{source.stem}_search_redacted.pdf", filetypes=[("PDF files", "*.pdf")])
        if not output:
            return
        if os.path.abspath(output) == os.path.abspath(self.pdf_path):
            messagebox.showerror("Choose a new file", "The original PDF is never overwritten.")
            return
        if not messagebox.askyesno("Permanent redaction", "Create a separate PDF with all matching text permanently removed?"):
            return
        doc = None
        total = 0
        try:
            doc = fitz.open(self.pdf_path)
            for page in doc:
                rects = []
                if pattern:
                    for word in page.get_text("words", sort=True):
                        if pattern.search(word[4]):
                            rects.append(fitz.Rect(word[:4]))
                else:
                    rects = page.search_for(query)
                for rect in rects:
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                if rects:
                    page.apply_redactions()
                    total += len(rects)
            if not total:
                messagebox.showinfo("No matches", "No matching selectable PDF text was found. Scanned PDFs need OCR support.")
                return
            doc.save(output, garbage=4, deflate=True)
            self._update_status(f"Saved {total} permanent PDF redactions: {os.path.basename(output)}")
            messagebox.showinfo("Saved", f"Saved {total} permanent redactions to:\n{output}")
        except Exception as e:
            messagebox.showerror("PDF Redaction Error", str(e))
        finally:
            if doc is not None:
                doc.close()

    def open_image(self, path=None):
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("All files", "*.*")
        ]
        if path is None:
            path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            ok = self._load_image(path)
            self.file_list = [path]
            self.current_file_index = 0
            self._update_file_list()
            if ok:
                self._add_recent(path)

    def _add_recent(self, path):
        path = os.path.abspath(path)
        recent = [p for p in SETTINGS["recent_files"] if p != path]
        SETTINGS["recent_files"] = ([path] + recent)[:10]
        try:
            save_settings()
        except OSError:
            pass  # ponytail: recent list is a convenience, a read-only settings.sec just means it is not remembered

    def _refresh_recent(self):
        menu = self.recent_menu
        menu.delete(0, tk.END)
        for p in SETTINGS["recent_files"]:
            menu.add_command(label=f"{os.path.basename(p)}   —   {os.path.dirname(p)}",
                             command=lambda p=p: self.open_recent(p))
        if not SETTINGS["recent_files"]:
            menu.add_command(label="(none)", state=tk.DISABLED)
        else:
            menu.add_separator()
            menu.add_command(label="Clear Recent Files", command=self.clear_recent)

    def open_recent(self, path):
        if not os.path.exists(path):
            messagebox.showwarning("Recent Files", f"File not found:\n{path}")
            SETTINGS["recent_files"] = [p for p in SETTINGS["recent_files"] if p != path]
            try:
                save_settings()
            except OSError:
                pass
            return
        if path.lower().endswith(".pdf"):
            self.open_pdf(path)
        else:
            self.open_image(path)

    def clear_recent(self):
        SETTINGS["recent_files"] = []
        try:
            save_settings()
        except OSError:
            pass

    def open_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            self.file_list = sorted([
                os.path.join(folder, f) for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in extensions
            ])
            if self.file_list:
                self.current_file_index = 0
                self._load_image(self.file_list[0])
                self._update_file_list()
            else:
                messagebox.showinfo("No Images", "No image files found in folder.")

    def _load_image(self, path):
        if self.pdf_document is not None:
            self.pdf_document.close()
            self.pdf_document, self.pdf_path = None, None
            self.pdf_page_masks, self.pdf_page_sizes = {}, {}
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Could not read image: {path}")

            self.current_image = img
            self.original_image = img.copy()
            self.mask_regions.clear()
            self.selected_region = None
            self.history.clear()
            self.region_counter = 0

            self.zoom_fit()
            self._update_status(f"Loaded: {os.path.basename(path)}")
            self.root.title(f"🛡️ IdentityShield Pro - {os.path.basename(path)}")
            return True
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return False

    def _update_file_list(self):
        self.file_listbox.delete(0, tk.END)
        for f in self.file_list:
            self.file_listbox.insert(tk.END, os.path.basename(f))
        if self.current_file_index >= 0:
            self.file_listbox.selection_set(self.current_file_index)
            self.file_listbox.see(self.current_file_index)
        self.file_count_label.config(
            text=f"{len(self.file_list)} file(s)")

    def _on_file_select(self, event):
        sel = self.file_listbox.curselection()
        if sel:
            idx = sel[0]
            if idx != self.current_file_index:
                self.current_file_index = idx
                self._load_image(self.file_list[idx])

    def prev_image(self):
        if self.pdf_document is not None:
            self._remember_pdf_masks()
            if self.pdf_page_index > 0:
                self._load_pdf_page(self.pdf_page_index - 1)
            return
        if self.file_list and self.current_file_index > 0:
            self.current_file_index -= 1
            self._load_image(self.file_list[self.current_file_index])
            self._update_file_list()

    def next_image(self):
        if self.pdf_document is not None:
            self._remember_pdf_masks()
            if self.pdf_page_index < self.pdf_document.page_count - 1:
                self._load_pdf_page(self.pdf_page_index + 1)
            return
        if self.file_list and self.current_file_index < len(self.file_list) - 1:
            self.current_file_index += 1
            self._load_image(self.file_list[self.current_file_index])
            self._update_file_list()

    def save_image(self):
        if self.pdf_document is not None:
            self.save_pdf_as()
            return
        if self.current_image is None:
            return
        if self.file_list and self.current_file_index >= 0:
            path = self.file_list[self.current_file_index]
            self._save_to_path(path)
        else:
            self.save_as()

    def save_as(self):
        if self.pdf_document is not None:
            self.save_pdf_as()
            return
        if self.current_image is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")]
        )
        if path:
            self._save_to_path(path)

    def _save_to_path(self, path):
        img = self._get_processed_image()
        if self.strip_exif_var.get():
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img = ImageProcessor.strip_exif(pil_img)
            pil_img.save(path, quality=95)
        else:
            cv2.imwrite(path, img)
        self._update_status(f"Saved: {os.path.basename(path)}")

    def export_image(self):
        if self.pdf_document is not None:
            self.save_pdf_as()
            return
        if self.current_image is None:
            return

        export_win = tk.Toplevel(self.root)
        export_win.title("📤 Export Settings")
        export_win.geometry("400x350")
        export_win.configure(bg=AppTheme.BG_PANEL)
        export_win.transient(self.root)
        export_win.wait_visibility()
        export_win.grab_set()

        tk.Label(export_win, text="📤 Export Image", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.ACCENT, font=("Segoe UI", 14, "bold")).pack(pady=15)

        form = tk.Frame(export_win, bg=AppTheme.BG_PANEL)
        form.pack(fill=tk.X, padx=20)

        # Format
        tk.Label(form, text="Format:", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.TEXT, font=("Segoe UI", 10)).grid(
            row=0, column=0, sticky="w", pady=5)
        fmt_var = tk.StringVar(value="PNG")
        fmt_combo = ttk.Combobox(form, textvariable=fmt_var,
                                  values=["PNG", "JPEG", "BMP", "TIFF", "WEBP"],
                                  state="readonly", width=15)
        fmt_combo.grid(row=0, column=1, pady=5, padx=10)

        # Quality
        tk.Label(form, text="Quality:", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.TEXT, font=("Segoe UI", 10)).grid(
            row=1, column=0, sticky="w", pady=5)
        quality_var = tk.IntVar(value=95)
        ttk.Scale(form, from_=1, to=100, variable=quality_var,
                  orient=tk.HORIZONTAL).grid(row=1, column=1, pady=5, padx=10, sticky="ew")

        # Suffix
        tk.Label(form, text="Suffix:", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.TEXT, font=("Segoe UI", 10)).grid(
            row=2, column=0, sticky="w", pady=5)
        suffix_var = tk.StringVar(value="_redacted")
        tk.Entry(form, textvariable=suffix_var, bg=AppTheme.ENTRY_BG,
                 fg=AppTheme.TEXT, font=("Segoe UI", 10),
                 insertbackground=AppTheme.TEXT).grid(
            row=2, column=1, pady=5, padx=10, sticky="ew")

        # Strip EXIF
        strip_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(form, text="Strip EXIF/Metadata",
                        variable=strip_var, style="Dark.TCheckbutton").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=5)

        # Output path
        output_var = tk.StringVar()
        tk.Label(form, text="Output:", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.TEXT, font=("Segoe UI", 10)).grid(
            row=4, column=0, sticky="w", pady=5)

        path_frame = tk.Frame(form, bg=AppTheme.BG_PANEL)
        path_frame.grid(row=4, column=1, pady=5, padx=10, sticky="ew")
        path_entry = tk.Entry(path_frame, textvariable=output_var,
                              bg=AppTheme.ENTRY_BG, fg=AppTheme.TEXT,
                              font=("Segoe UI", 9), insertbackground=AppTheme.TEXT)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(path_frame, text="📂", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.TEXT, cursor="hand2", font=("Segoe UI", 12)).pack(
            side=tk.RIGHT, padx=2)

        def do_export():
            ext_map = {"PNG": ".png", "JPEG": ".jpg", "BMP": ".bmp",
                       "TIFF": ".tiff", "WEBP": ".webp"}
            ext = ext_map.get(fmt_var.get(), ".png")
            suffix = suffix_var.get()

            if output_var.get():
                save_path = output_var.get()
            else:
                save_path = filedialog.asksaveasfilename(
                    defaultextension=ext,
                    filetypes=[(fmt_var.get(), f"*{ext}")]
                )

            if save_path:
                img = self._get_processed_image()
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                if strip_var.get():
                    pil_img = ImageProcessor.strip_exif(pil_img)
                pil_img.save(save_path, quality=quality_var.get())
                self._update_status(f"Exported: {os.path.basename(save_path)}")
                export_win.destroy()

        btn_frame = tk.Frame(export_win, bg=AppTheme.BG_PANEL)
        btn_frame.pack(fill=tk.X, padx=20, pady=20)
        ActionButton(btn_frame, "Cancel",
                    lambda: export_win.destroy()).pack(side=tk.LEFT)
        ActionButton(btn_frame, "  Export  ", do_export,
                    accent=True).pack(side=tk.RIGHT)

    def _get_processed_image(self) -> np.ndarray:
        if self.current_image is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        img = self.current_image.copy()
        for region in self.mask_regions:
            if region.visible:
                img = ImageProcessor.apply_mask_region(img, region)
        return img

    # ── Face Detection ──────────────────────────────────────────────────

    def auto_detect(self):
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        self._update_status("Detecting faces...")
        self.root.update()

        confidence = self.confidence_slider.get() / 100.0
        min_size = int(self.min_face_slider.get())

        faces = ImageProcessor.detect_faces(
            self.current_image,
            confidence_threshold=confidence,
            min_size=min_size
        )

        if faces:
            self.history.push(self.mask_regions[:])
            for i, (x, y, w, h) in enumerate(faces):
                # Add padding
                pad = int(w * 0.1)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(self.current_image.shape[1], x + w + pad)
                y2 = min(self.current_image.shape[0], y + h + pad)

                region = MaskRegion(
                    region_id=self._next_region_id(),
                    tool_type=ToolType.AUTO_DETECT,
                    points=[(x1, y1), (x2, y2)],
                    mask_type=self._get_current_mask_type(),
                    block_size=int(self.block_slider.get()),
                    blur_strength=int(self.blur_slider.get()),
                    fill_color=self.fill_color,
                    opacity=self.opacity_slider.get() / 100.0,
                    label=f"Face #{i+1}"
                )
                self.mask_regions.append(region)

            self._update_mask_list()
            self._refresh_canvas()
            self._update_status(
                f"Detected {len(faces)} face(s) [{ImageProcessor.face_engine}] {ImageProcessor.face_note}")
        else:
            self._update_status("No faces detected")
            messagebox.showinfo("Detection", "No faces detected in the image.\n"
                               "Try adjusting confidence or minimum face size.")

    # ── Zoom & Pan ──────────────────────────────────────────────────────

    def zoom_in(self, center=None):
        self.set_zoom(min(self.zoom_level * 1.25, 10.0), center)

    def zoom_out(self, center=None):
        self.set_zoom(max(self.zoom_level / 1.25, 0.05), center)

    def set_zoom(self, level, center=None):
        if self.current_image is None:
            return

        old_zoom = self.zoom_level
        self.zoom_level = level

        if center:
            # Zoom toward cursor position
            cx, cy = center
            self.pan_offset[0] = cx - (cx - self.pan_offset[0]) * level / old_zoom
            self.pan_offset[1] = cy - (cy - self.pan_offset[1]) * level / old_zoom

        self._refresh_canvas()

    def zoom_fit(self):
        if self.current_image is None:
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 800, 600

        h, w = self.current_image.shape[:2]
        zoom_x = canvas_w / w
        zoom_y = canvas_h / h
        self.zoom_level = min(zoom_x, zoom_y) * 0.95

        # Center image
        new_w = w * self.zoom_level
        new_h = h * self.zoom_level
        self.pan_offset[0] = (canvas_w - new_w) / 2
        self.pan_offset[1] = (canvas_h - new_h) / 2

        self._refresh_canvas()

    # ── Undo/Redo ───────────────────────────────────────────────────────

    def undo(self):
        result = self.history.undo(self.mask_regions)
        if result is not None:
            self.mask_regions = result
            self.selected_region = None
            self._update_mask_list()
            self._refresh_canvas()
            self._update_status("Undone")

    def redo(self):
        result = self.history.redo(self.mask_regions)
        if result is not None:
            self.mask_regions = result
            self.selected_region = None
            self._update_mask_list()
            self._refresh_canvas()
            self._update_status("Redone")

    # ── Actions ─────────────────────────────────────────────────────────

    def apply_all_masks(self):
        if self.current_image is None or not self.mask_regions:
            return
        self.current_image = self._get_processed_image()
        self.mask_regions.clear()
        self.selected_region = None
        self.history.clear()
        self._update_mask_list()
        self._refresh_canvas()
        self._update_status("All masks applied permanently")

    def preview_result(self):
        if self.current_image is None:
            return
        self._refresh_canvas()
        self._update_status("Preview updated")

    def clear_all_masks(self):
        if not self.mask_regions:
            return
        self.history.push(self.mask_regions[:])
        self.mask_regions.clear()
        self.selected_region = None
        self._update_mask_list()
        self._refresh_canvas()
        self._update_status("All masks cleared")

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.mask_regions.clear()
            self.selected_region = None
            self.history.clear()
            self._update_mask_list()
            self._refresh_canvas()
            self._update_status("Image reset to original")

    # ADD this method to IdentityShieldApp:

    def remove_background(self):
        """Remove background using GrabCut or rembg if available."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        self._update_status("Removing background...")
        self.root.update()

        # Try rembg first (better results)
        try:
            from rembg import remove as rembg_remove
            rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            result_pil = rembg_remove(pil_img)

            # Ask for background color
            bg_choice = messagebox.askyesnocancel(
                "Background",
                "Background removed!\n\n"
                "Yes = White background\n"
                "No = Transparent (save as PNG)\n"
                "Cancel = Abort"
            )
            if bg_choice is None:
                self._update_status("Background removal cancelled")
                return

            if bg_choice:
                # White background
                bg = Image.new("RGBA", result_pil.size, (255, 255, 255, 255))
                bg.paste(result_pil, mask=result_pil.split()[3])
                result_rgb = np.array(bg.convert("RGB"))
            else:
                # Save as PNG with alpha
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG", "*.png")]
                )
                if save_path:
                    result_pil.save(save_path)
                    self._update_status(f"Saved transparent: {os.path.basename(save_path)}")
                return

            self.history.push(self.mask_regions[:])
            self.current_image = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            self.mask_regions.clear()
            self.selected_region = None
            self._update_mask_list()
            self._refresh_canvas()
            self._update_status("Background removed (rembg)")
            return

        except ImportError:
            pass

        # Fallback: GrabCut (OpenCV built-in)
        try:
            h, w = self.current_image.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            # Use center rectangle as initial foreground estimate
            margin_x = int(w * 0.05)
            margin_y = int(h * 0.05)
            rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

            cv2.grabCut(self.current_image, mask, rect, bgd_model, fgd_model,
                        5, cv2.GC_INIT_WITH_RECT)

            # Create mask where 0=bg, 2=probable bg -> set to white
            fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            bg_choice = messagebox.askyesno(
                "Background",
                "Background detected (GrabCut).\n\n"
                "Yes = White background\n"
                "No = Cancel\n\n"
                "Tip: Install 'rembg' for better results:\n"
                "pip install rembg"
            )
            if not bg_choice:
                self._update_status("Background removal cancelled")
                return

            result = self.current_image.copy()
            result[fg_mask == 0] = [255, 255, 255]

            self.history.push(self.mask_regions[:])
            self.current_image = result
            self.mask_regions.clear()
            self.selected_region = None
            self._update_mask_list()
            self._refresh_canvas()
            self._update_status("Background removed (GrabCut fallback)")

        except Exception as e:
            messagebox.showerror("Error", f"Background removal failed:\n{e}")
            self._update_status("Background removal failed")

    # ADD these methods to IdentityShieldApp:

    def remove_text(self):
        """Detect and remove text regions from image."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # Show options dialog
        text_win = tk.Toplevel(self.root)
        text_win.title("🔤 Text Removal")
        text_win.geometry("380x320")
        text_win.configure(bg=AppTheme.BG_PANEL)
        text_win.transient(self.root)
        text_win.wait_visibility()
        text_win.grab_set()

        tk.Label(text_win, text="🔤 Text Removal", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.ACCENT, font=("Segoe UI", 14, "bold")).pack(pady=15)

        tk.Label(text_win, text="Detection Method:", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.TEXT, font=("Segoe UI", 10)).pack(anchor="w", padx=20)

        method_var = tk.StringVar(value="east")

        methods_frame = tk.Frame(text_win, bg=AppTheme.BG_PANEL)
        methods_frame.pack(fill=tk.X, padx=25, pady=5)

        ttk.Radiobutton(methods_frame, text="EAST Deep Learning (best)",
                        variable=method_var, value="east",
                        style="Dark.TRadiobutton").pack(anchor="w")
        ttk.Radiobutton(methods_frame, text="MSER + Morphology (no model needed)",
                        variable=method_var, value="mser",
                        style="Dark.TRadiobutton").pack(anchor="w")
        ttk.Radiobutton(methods_frame, text="EasyOCR (if installed)",
                        variable=method_var, value="easyocr",
                        style="Dark.TRadiobutton").pack(anchor="w")

        tk.Label(text_win, text="Inpaint Method:", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.TEXT, font=("Segoe UI", 10)).pack(anchor="w", padx=20, pady=(10, 0))

        inpaint_var = tk.StringVar(value="telea")
        ip_frame = tk.Frame(text_win, bg=AppTheme.BG_PANEL)
        ip_frame.pack(fill=tk.X, padx=25, pady=5)

        ttk.Radiobutton(ip_frame, text="Telea Inpaint (smooth)",
                        variable=inpaint_var, value="telea",
                        style="Dark.TRadiobutton").pack(anchor="w")
        ttk.Radiobutton(ip_frame, text="Navier-Stokes Inpaint",
                        variable=inpaint_var, value="ns",
                        style="Dark.TRadiobutton").pack(anchor="w")
        ttk.Radiobutton(ip_frame, text="Mask only (create masks, don't remove)",
                        variable=inpaint_var, value="mask",
                        style="Dark.TRadiobutton").pack(anchor="w")

        btn_frame = tk.Frame(text_win, bg=AppTheme.BG_PANEL)
        btn_frame.pack(fill=tk.X, padx=20, pady=15)

        ActionButton(btn_frame, "Cancel",
                     lambda: text_win.destroy()).pack(side=tk.LEFT)
        ActionButton(btn_frame, "  Remove Text  ",
                     lambda: self._execute_text_removal(
                         method_var.get(), inpaint_var.get(), text_win),
                     accent=True).pack(side=tk.RIGHT)

    def _execute_text_removal(self, method, inpaint_method, dialog):
        """Execute text detection and removal."""
        dialog.destroy()
        self._update_status(f"Detecting text ({method})...")
        self.root.update()

        text_mask = None

        if method == "easyocr":
            text_mask = self._detect_text_easyocr()
        elif method == "east":
            text_mask = self._detect_text_east()

        # Fallback to MSER
        if text_mask is None or not text_mask.any():
            if method != "mser":
                self._update_status("Falling back to MSER method...")
                self.root.update()
            text_mask = self._detect_text_mser()

        if text_mask is None or not text_mask.any():
            messagebox.showinfo("Text Removal", "No text regions detected.")
            self._update_status("No text detected")
            return

        if inpaint_method == "mask":
            # Create mask regions instead of inpainting
            self.history.push(self.mask_regions[:])
            contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 100:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                region = MaskRegion(
                    region_id=self._next_region_id(),
                    tool_type=ToolType.RECTANGLE,
                    points=[(x, y), (x + w, y + h)],
                    mask_type=self._get_current_mask_type(),
                    block_size=int(self.block_slider.get()),
                    blur_strength=int(self.blur_slider.get()),
                    fill_color=self.fill_color,
                    opacity=self.opacity_slider.get() / 100.0,
                    label=f"Text #{count + 1}"
                )
                self.mask_regions.append(region)
                count += 1
            self._update_mask_list()
            self._refresh_canvas()
            self._update_status(f"Created {count} text masks")
            return

        # Inpaint
        self._update_status("Inpainting text regions...")
        self.root.update()

        # Dilate mask slightly for better coverage
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        text_mask = cv2.dilate(text_mask, kernel, iterations=2)

        if inpaint_method == "telea":
            result = cv2.inpaint(self.current_image, text_mask, 7, cv2.INPAINT_TELEA)
        else:
            result = cv2.inpaint(self.current_image, text_mask, 7, cv2.INPAINT_NS)

        self.history.push(self.mask_regions[:])
        self.current_image = result
        self.mask_regions.clear()
        self.selected_region = None
        self._update_mask_list()
        self._refresh_canvas()
        self._update_status("Text removed successfully")

    def _detect_text_mser(self) -> Optional[np.ndarray]:
        """Detect text using MSER + morphological operations."""
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        mser = cv2.MSER_create()
        mser.setMinArea(60)
        mser.setMaxArea(int(h * w * 0.05))

        regions, _ = mser.detectRegions(gray)

        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.fillPoly(mask, [hull], 255)

        # Morphological cleanup — connect nearby text, remove noise
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # Filter by aspect ratio (text regions are typically wide)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < 200:
                continue
            aspect = cw / max(ch, 1)
            if 0.2 < aspect < 20:  # reasonable text aspect ratio
                cv2.rectangle(filtered_mask, (x, y), (x + cw, y + ch), 255, -1)

        return filtered_mask

    def _detect_text_east(self) -> Optional[np.ndarray]:
        """Detect text using EAST deep learning model."""
        try:
            model_path = SETTINGS["east_model"] or "frozen_east_text_detection.pb"
            if not os.path.exists(model_path):
                # Check common locations
                for p in [
                    os.path.join(os.path.expanduser("~"), model_path),
                    os.path.join(os.path.dirname(__file__), model_path),
                ]:
                    if os.path.exists(p):
                        model_path = p
                        break
                else:
                    self._update_status("EAST model not found, falling back...")
                    return None

            img = self.current_image.copy()
            orig_h, orig_w = img.shape[:2]

            # EAST requires dimensions multiple of 32
            new_w = ((orig_w // 32) + 1) * 32
            new_h = ((orig_h // 32) + 1) * 32
            new_w = min(new_w, 1280)
            new_h = min(new_h, 1280)
            ratio_w = orig_w / new_w
            ratio_h = orig_h / new_h

            resized = cv2.resize(img, (new_w, new_h))
            blob = cv2.dnn.blobFromImage(resized, 1.0, (new_w, new_h),
                                         (123.68, 116.78, 103.94), True, False)

            net = cv2.dnn.readNet(model_path)
            net.setInput(blob)
            scores, geometry = net.forward([
                "feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"
            ])

            rects, confidences = self._decode_east(scores, geometry, 0.5)
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

            if len(rects) > 0:
                indices = cv2.dnn.NMSBoxesRotated(
                    rects, confidences, 0.5, 0.4)
                for i in indices.flatten():
                    vertices = cv2.boxPoints(rects[i])
                    vertices[:, 0] *= ratio_w
                    vertices[:, 1] *= ratio_h
                    vertices = vertices.astype(np.int32)
                    cv2.fillPoly(mask, [vertices], 255)

            return mask

        except Exception as e:
            self._update_status(f"EAST failed: {e}")
            return None

    def _decode_east(self, scores, geometry, min_confidence):
        """Decode EAST model outputs."""
        (num_rows, num_cols) = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(num_rows):
            scores_data = scores[0, 0, y]
            x0 = geometry[0, 0, y]
            x1 = geometry[0, 1, y]
            x2 = geometry[0, 2, y]
            x3 = geometry[0, 3, y]
            angles = geometry[0, 4, y]

            for x in range(num_cols):
                if scores_data[x] < min_confidence:
                    continue

                offset_x = x * 4.0
                offset_y = y * 4.0
                angle = angles[x]

                h = x0[x] + x2[x]
                w = x1[x] + x3[x]

                end_x = int(offset_x + (np.cos(angle) * x1[x]) + (np.sin(angle) * x2[x]))
                end_y = int(offset_y - (np.sin(angle) * x1[x]) + (np.cos(angle) * x2[x]))

                center_x = int(end_x - w / 2)
                center_y = int(end_y - h / 2)

                rects.append(((center_x, center_y), (w, h), -1 * angle * 180.0 / np.pi))
                confidences.append(float(scores_data[x]))

        return rects, confidences

    def _detect_text_easyocr(self) -> Optional[np.ndarray]:
        """Detect text using EasyOCR."""
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False)
            rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            results = reader.readtext(rgb)

            h, w = self.current_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            for (bbox, text, conf) in results:
                if conf < 0.3:
                    continue
                pts = np.array(bbox, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

            return mask if mask.any() else None

        except ImportError:
            self._update_status("EasyOCR not installed, falling back...")
            return None


    def open_settings(self):
        """Edit every path/model setting; saved to settings.sec (JSON) next to the script."""
        win = tk.Toplevel(self.root)
        win.title("⚙ Settings")
        win.geometry("640x560")
        win.configure(bg=AppTheme.BG_PANEL)
        win.transient(self.root)
        win.wait_visibility()
        win.grab_set()

        models = {k: list(v) for k, v in SETTINGS["sam2_models"].items()}
        font = ("Segoe UI", 10)
        form = tk.Frame(win, bg=AppTheme.BG_PANEL)
        form.pack(fill=tk.X, padx=20, pady=(15, 5))
        form.columnconfigure(1, weight=1)

        def path_row(r, label, key, kind, tip):
            tk.Label(form, text=label, bg=AppTheme.BG_PANEL, fg=AppTheme.TEXT,
                     font=font).grid(row=r, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=SETTINGS[key])
            entry = tk.Entry(form, textvariable=var, bg=AppTheme.ENTRY_BG, fg=AppTheme.TEXT,
                             insertbackground=AppTheme.TEXT, font=("Segoe UI", 9))
            entry.grid(row=r, column=1, sticky="ew", padx=8, pady=4)

            def browse():
                if kind == "dir":
                    p = filedialog.askdirectory(parent=win)
                else:
                    p = filedialog.askopenfilename(parent=win)
                if p:
                    var.set(p)

            b = tk.Label(form, text="📂", bg=AppTheme.BG_PANEL, fg=AppTheme.TEXT,
                         cursor="hand2", font=("Segoe UI", 12))
            b.grid(row=r, column=2)
            b.bind("<Button-1>", lambda e: browse())
            ToolTip(entry, tip)
            return var

        v_dir = path_row(0, "SAM 2 folder:", "sam2_dir", "dir", "Folder holding the SAM 2 checkpoint (.pt) files listed below.")
        v_person = path_row(1, "Person model:", "yolo_person", "file", "YOLO person weights. A bare name like yolov8x.pt is downloaded automatically.")
        v_face = path_row(2, "Face model:", "yolo_face", "file", "YOLO face weights (e.g. yolov8l-face.pt). Empty = basic Haar detector.")
        v_east = path_row(3, "EAST text model:", "east_model", "file", "frozen_east_text_detection.pb for Remove Text. Empty = look next to the script or in your home folder.")
        v_out = path_row(4, "Output folder:", "output_dir", "dir", "Where folder pickers start when saving. Empty = last used.")

        tk.Label(win, text="People-detection models (name | checkpoint file | config)",
                 bg=AppTheme.BG_PANEL, fg=AppTheme.ACCENT, font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=20, pady=(12, 3))
        lst = tk.Listbox(win, height=7, bg=AppTheme.ENTRY_BG, fg=AppTheme.TEXT,
                         selectbackground=AppTheme.ACCENT, font=("Consolas", 9),
                         exportselection=False)
        lst.pack(fill=tk.X, padx=20)

        def refill():
            lst.delete(0, tk.END)
            for k, (c, y) in models.items():
                lst.insert(tk.END, f"{k}  |  {c}  |  {y}")

        def add_model():
            name = simpledialog.askstring("Model", "Display name (e.g. Large):", parent=win)
            if not name:
                return
            ckpt = simpledialog.askstring("Model", "Checkpoint file name (inside the SAM 2 folder):", parent=win)
            cfg = simpledialog.askstring("Model", "Config file name (e.g. sam2_hiera_l.yaml):", parent=win)
            if ckpt and cfg:
                models[name.strip()] = [ckpt.strip(), cfg.strip()]
                refill()

        def remove_model():
            sel = lst.curselection()
            if not sel:
                return
            if len(models) <= 1:
                messagebox.showwarning("Models", "Keep at least one model.", parent=win)
                return
            models.pop(list(models)[sel[0]])
            refill()
            combo["values"] = list(models)

        refill()
        mb = tk.Frame(win, bg=AppTheme.BG_PANEL)
        mb.pack(fill=tk.X, padx=20, pady=5)
        ActionButton(mb, "  Add  ", add_model).pack(side=tk.LEFT)
        ActionButton(mb, "  Remove  ", remove_model).pack(side=tk.LEFT, padx=6)

        dr = tk.Frame(win, bg=AppTheme.BG_PANEL)
        dr.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(dr, text="Default model:", bg=AppTheme.BG_PANEL, fg=AppTheme.TEXT,
                 font=font).pack(side=tk.LEFT)
        v_def = tk.StringVar(value=SETTINGS["sam2_default"])
        combo = ttk.Combobox(dr, textvariable=v_def, values=list(models), state="readonly", width=14)
        combo.pack(side=tk.LEFT, padx=8)

        def save():
            if v_def.get() not in models:
                v_def.set(next(iter(models)))
            SETTINGS.update({
                "sam2_dir": v_dir.get().strip(),
                "yolo_person": v_person.get().strip(),
                "yolo_face": v_face.get().strip(),
                "east_model": v_east.get().strip(),
                "output_dir": v_out.get().strip(),
                "sam2_models": models,
                "sam2_default": v_def.get(),
            })
            try:
                save_settings()
            except OSError as e:
                messagebox.showerror("Settings", f"Could not write {SETTINGS_FILE}:\n{e}", parent=win)
                return
            # drop cached models so new paths take effect
            ImageProcessor._yolo_face = None
            ImageProcessor.face_note = ""
            self._yolo_person = None
            self._sam2_cache = {}
            self._update_status(f"Settings saved to {SETTINGS_FILE.name}")
            win.destroy()

        bf = tk.Frame(win, bg=AppTheme.BG_PANEL)
        bf.pack(fill=tk.X, padx=20, pady=15, side=tk.BOTTOM)
        ActionButton(bf, "Cancel", win.destroy).pack(side=tk.LEFT)
        ActionButton(bf, "  Save  ", save, accent=True).pack(side=tk.RIGHT)

    def _load_sam2(self, key):
        """Load (and cache) a SAM2 image predictor."""
        cache = getattr(self, "_sam2_cache", None)
        if cache is None:
            cache = {}
            self._sam2_cache = cache
        if key in cache:
            return cache[key]

        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        ckpt_name, cfg_name = SETTINGS["sam2_models"][key]
        ckpt_path = os.path.join(SETTINGS["sam2_dir"], ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{ckpt_path}\nSet the model folder in File > Settings.")

        device = "cuda"
        if not torch.cuda.is_available():
            device = "cpu"

        last_err = None
        candidates = (
            f"configs/sam2/{cfg_name}",
            cfg_name,
            f"configs/sam2.1/{cfg_name}",
        )
        for cfg in candidates:
            try:
                model = build_sam2(cfg, ckpt_path, device=device)
                predictor = SAM2ImagePredictor(model)
                cache[key] = predictor
                return predictor
            except Exception as e:
                last_err = e
        raise last_err

    def _segment_people(self, key, conf):
        """YOLO person boxes -> SAM2 mask per person. Returns list of bool masks (left to right)."""
        import torch
        from ultralytics import YOLO

        yolo = getattr(self, "_yolo_person", None)
        if yolo is None:
            yolo = YOLO(SETTINGS["yolo_person"])
            self._yolo_person = yolo

        img = self.current_image.copy()
        # ponytail: slider scaled to half like faces (50% = 0.25), 1280 for small/occluded people in crowds
        res = yolo(img, classes=[0], conf=conf * 0.5, imgsz=1280, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return []

        confs = res.boxes.conf.cpu().numpy()
        kept = _filter_boxes(boxes, confs, img.shape[0] * img.shape[1])
        if not kept:
            return []
        boxes = boxes[kept]
        faces = ImageProcessor.detect_faces(img, confidence_threshold=conf, min_size=20)
        shown = _with_face(boxes, faces)
        if not shown:
            return []
        boxes = boxes[shown]

        predictor = self._load_sam2(key)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        use_cuda = torch.cuda.is_available()
        dev_type = "cuda"
        if not use_cuda:
            dev_type = "cpu"

        masks_out = []
        with torch.inference_mode():
            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_cuda):
                predictor.set_image(rgb)
                for box in boxes:
                    masks, scores, _ = predictor.predict(
                        box=box, multimask_output=True
                    )
                    masks_out.append(_pick_whole_person(masks, scores, box))
        if not masks_out:
            return []
        # ponytail: best-first cleanup (duplicates, fragments, shared pixels), then left to right
        masks_out, boxes = _clean_people(masks_out, boxes)
        order = np.argsort([(b[0] + b[2]) / 2 for b in boxes])
        return [masks_out[i] for i in order]

    def extract_humans(self):
        """Pick SAM2 model, detect humans, then open the results window."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        win = tk.Toplevel(self.root)
        win.title("🧍 Detect People")
        win.geometry("380x170")
        win.configure(bg=AppTheme.BG_PANEL)
        win.transient(self.root)
        win.wait_visibility()
        win.grab_set()

        tk.Label(win, text="🧍 Detect People", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.ACCENT, font=("Segoe UI", 14, "bold")).pack(pady=12)

        row = tk.Frame(win, bg=AppTheme.BG_PANEL)
        row.pack(fill=tk.X, padx=20)
        tk.Label(row, text="Model size:", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.TEXT, font=("Segoe UI", 10)).pack(side=tk.LEFT)
        names = list(SETTINGS["sam2_models"].keys())
        default = SETTINGS["sam2_default"]
        if default not in names:
            default = names[0]
        model_var = tk.StringVar(value=default)
        combo = ttk.Combobox(row, textvariable=model_var,
                             values=names,
                             state="readonly", width=12)
        combo.pack(side=tk.LEFT, padx=10)
        ToolTip(combo, "Larger = more accurate outlines but slower. Change the default in File > Settings.")

        btn_frame = tk.Frame(win, bg=AppTheme.BG_PANEL)
        btn_frame.pack(fill=tk.X, padx=20, pady=15)
        ActionButton(btn_frame, "Cancel", lambda: win.destroy()).pack(side=tk.LEFT)
        ActionButton(
            btn_frame, "  Detect  ",
            lambda: self._run_extract_humans(model_var.get(), win),
            accent=True).pack(side=tk.RIGHT)

    def _run_extract_humans(self, key, dialog):
        """Run YOLO + SAM2 in a background thread, then show results on the UI thread."""
        dialog.destroy()

        conf = self.confidence_slider.get() / 100.0
        base = "image"
        if self.file_list and self.current_file_index >= 0:
            base = os.path.splitext(os.path.basename(self.file_list[self.current_file_index]))[0]

        src = self.current_image
        rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        self._update_status("Detecting people...")

        def worker():
            try:
                masks = self._segment_people(key, conf)
                if not masks:
                    self.root.after(0, lambda: self._update_status("No people detected"))
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Detect People",
                        "No people detected.\nTry lowering the Confidence slider."))
                    return
                self.root.after(0, lambda: self._show_humans_window(rgb, masks, base, src))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: self._update_status("Extract humans failed"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Detect People", f"Failed:\n{err}"))

        threading.Thread(target=worker, daemon=True).start()

    def _fill_removed(self, img, gone):
        """Fill the masked area. LaMa if installed, else OpenCV Telea. Returns (image, engine)."""
        h, w = img.shape[:2]
        # ponytail: long side capped at 1536 for speed/VRAM; only the masked pixels are replaced
        scale = min(1.0, 1536 / max(h, w))
        sw = max(1, int(w * scale))
        sh = max(1, int(h * scale))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        # any partly covered pixel counts as masked, plus 1px grow, so no fringe of the person survives the downscale
        small_gone = (cv2.resize(gone, (sw, sh), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8) * 255
        small_gone = cv2.dilate(small_gone, np.ones((3, 3), np.uint8))

        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError:
            SimpleLama = None

        if SimpleLama is not None:
            lama = getattr(self, "_lama", None)
            if lama is None:
                lama = SimpleLama()
                self._lama = lama
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            out = lama(Image.fromarray(rgb), Image.fromarray(small_gone))
            filled = cv2.cvtColor(np.array(out.convert("RGB")), cv2.COLOR_RGB2BGR)
            engine = "LaMa"
        else:
            filled = cv2.inpaint(small, small_gone, 7, cv2.INPAINT_TELEA)
            engine = "basic fill, pip install simple-lama-inpainting for clean results"

        if filled.shape[:2] != (sh, sw):
            filled = cv2.resize(filled, (sw, sh))
        filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
        result = img.copy()
        result[gone > 0] = filled[gone > 0]
        return result, engine

    def _show_humans_window(self, rgb, masks, base, src):
        """Results window: all-humans view (cutout/silhouette) + thumbstrip with per-person extract."""
        h, w = rgb.shape[:2]
        union = np.zeros((h, w), dtype=bool)
        for m in masks:
            union |= m

        views = {
            "Cutout": Image.fromarray(np.dstack([rgb, union.astype(np.uint8) * 255])),
            "Silhouette": Image.fromarray(np.where(union, 0, 255).astype(np.uint8)).convert("RGB"),
        }

        persons = []
        for m in masks:
            ys, xs = np.where(m)
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            crop = np.dstack([
                rgb[y1:y2 + 1, x1:x2 + 1],
                m[y1:y2 + 1, x1:x2 + 1].astype(np.uint8) * 255,
            ])
            persons.append(Image.fromarray(crop))

        selected = set()
        thumbs = []
        photos = []

        win = tk.Toplevel(self.root)
        win.title(f"🧍 People found: {len(persons)}")
        win.geometry("1000x780")
        win.configure(bg=AppTheme.BG_PANEL)
        win.transient(self.root)

        view_var = tk.StringVar(value="Cutout")

        def show_view():
            img = views[view_var.get()].copy()
            img.thumbnail((960, 430), Image.LANCZOS)
            ph = ImageTk.PhotoImage(img)
            preview.config(image=ph)
            preview.image = ph

        def paint():
            for i, lbl in enumerate(thumbs):
                color = AppTheme.BG_DARK
                if i in selected:
                    color = AppTheme.ACCENT
                lbl.config(highlightbackground=color)
            sel_label.config(text=f"{len(selected)} / {len(persons)} selected")

        def toggle(i):
            if i in selected:
                selected.discard(i)
            else:
                selected.add(i)
            paint()

        def select_all():
            selected.update(range(len(persons)))
            paint()

        def pick_folder():
            return filedialog.askdirectory(parent=win, initialdir=SETTINGS["output_dir"] or None)

        def save_view():
            folder = pick_folder()
            if not folder:
                return
            name = view_var.get().lower()
            views[view_var.get()].save(os.path.join(folder, f"{base}_{name}.png"))
            self._update_status(f"Saved {name} to {folder}")

        def save_persons(indices):
            if not indices:
                messagebox.showinfo("Detect People",
                                    "Click thumbnails to select first.", parent=win)
                return
            folder = pick_folder()
            if not folder:
                return
            for i in sorted(indices):
                persons[i].save(os.path.join(folder, f"{base}_person_{i + 1}.png"))
            self._update_status(f"Saved {len(indices)} person(s) to {folder}")

        def check_ready():
            if not selected:
                messagebox.showinfo("Detect People",
                                    "Click thumbnails to select first.", parent=win)
                return False
            if self.current_image is not src:
                messagebox.showwarning(
                    "Detect People",
                    "Image changed since detection. Run Detect People again.",
                    parent=win)
                return False
            return True

        def grow(i):
            k = max(5, int(0.01 * max(h, w))) | 1  # ponytail: fixed 1% edge grow, add a slider if you need control
            return cv2.dilate(masks[i].astype(np.uint8) * 255, np.ones((k, k), np.uint8))

        def add_region(tool, points, label):
            self.mask_regions.append(MaskRegion(
                region_id=self._next_region_id(),
                tool_type=tool,
                points=points,
                mask_type=self._get_current_mask_type(),
                block_size=int(self.block_slider.get()),
                blur_strength=int(self.blur_slider.get()),
                fill_color=self.fill_color,
                opacity=self.opacity_slider.get() / 100.0,
                label=label,
            ))

        def finish(msg):
            self._update_mask_list()
            self._refresh_canvas()
            self._update_status(msg)
            win.destroy()

        def redact_body():
            if not check_ready():
                return
            self.history.push(self.mask_regions[:])
            for i in sorted(selected):
                contours, _ = cv2.findContours(grow(i), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if len(cnt) < 3:
                        continue
                    add_region(ToolType.POLYGON,
                               [(int(p[0][0]), int(p[0][1])) for p in cnt],
                               f"Person {i + 1}")
            finish(f"Redacting body of {len(selected)} person(s)")

        def redact_face():
            if not check_ready():
                return
            faces = ImageProcessor.detect_faces(
                self.current_image,
                confidence_threshold=self.confidence_slider.get() / 100.0,
                min_size=int(self.min_face_slider.get()))
            new = []
            missing = []
            for i in sorted(selected):
                grown = grow(i)
                found = False
                for (x, y, fw, fh) in faces:
                    if grown[y + fh // 2, x + fw // 2] == 0:
                        continue
                    pad = int(fw * 0.1)
                    new.append(([(max(0, x - pad), max(0, y - pad)),
                                 (min(w, x + fw + pad), min(h, y + fh + pad))],
                                f"Person {i + 1} face"))
                    found = True
                if not found:
                    missing.append(str(i + 1))
            if missing:
                messagebox.showwarning(
                    "Detect People",
                    f"No face found for Person {', '.join(missing)}.\n"
                    "Lower Confidence / Min Face Size, or use Redact Body.",
                    parent=win)
            if not new:
                return
            self.history.push(self.mask_regions[:])
            for points, label in new:
                add_region(ToolType.AUTO_DETECT, points, label)
            finish(f"Redacting {len(new)} face(s) [{ImageProcessor.face_engine}] {ImageProcessor.face_note}")

        def remove_people():
            if not check_ready():
                return
            gone = np.zeros((h, w), np.uint8)
            for i in selected:
                gone |= grow(i)
            count = len(selected)
            base_img = self.current_image
            win.destroy()
            self._update_status("Removing people...")

            def on_removed(r):
                pts = r.points
                if len(pts) == 2:
                    pts = [((pts[0][0] + pts[1][0]) // 2, (pts[0][1] + pts[1][1]) // 2)]
                hit = sum(1 for x, y in pts if 0 <= y < h and 0 <= x < w and gone[y, x] > 0)
                return hit * 2 >= len(pts)

            def done(result, engine):
                if self.current_image is not base_img:
                    self._update_status("Image changed during removal, result discarded")
                    return
                self.current_image = result
                # masks over the erased people (body/face redactions) would pixelate the fill, drop them
                self.history.push(self.mask_regions[:])
                self.mask_regions = [r for r in self.mask_regions if not on_removed(r)]
                self._update_mask_list()
                self._refresh_canvas()
                self._update_status(f"Removed {count} person(s) ({engine})")

            def worker():
                try:
                    result, engine = self._fill_removed(base_img, gone)
                except Exception as e:
                    err = str(e)
                    self.root.after(0, lambda: self._update_status("Remove failed"))
                    self.root.after(0, lambda: messagebox.showerror(
                        "Detect People", f"Failed:\n{err}"))
                    return
                self.root.after(0, lambda: done(result, engine))

            threading.Thread(target=worker, daemon=True).start()

        # top bar: view switch + save current view
        top = tk.Frame(win, bg=AppTheme.BG_PANEL)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)
        for name in views:
            ttk.Radiobutton(top, text=name, variable=view_var, value=name,
                            style="Dark.TRadiobutton",
                            command=show_view).pack(side=tk.LEFT, padx=6)
        ActionButton(top, "  Save View  ", save_view, accent=True,
                     tip="Save the image shown below (Cutout or Silhouette) as a PNG.").pack(side=tk.RIGHT)

        # bottom bar: thumbstrip actions
        bar = tk.Frame(win, bg=AppTheme.BG_PANEL)
        bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)
        sel_label = tk.Label(bar, text="", bg=AppTheme.BG_PANEL,
                             fg=AppTheme.TEXT_DIM, font=("Segoe UI", 9))
        sel_label.pack(side=tk.LEFT)
        ActionButton(bar, "  Save All  ",
                     lambda: save_persons(range(len(persons))),
                     tip="Save every person as a separate transparent PNG.").pack(side=tk.RIGHT, padx=4)
        ActionButton(bar, "  Save Selected  ",
                     lambda: save_persons(selected), accent=True,
                     tip="Save only the selected people as separate transparent PNGs.").pack(side=tk.RIGHT, padx=4)
        ActionButton(bar, "  Select All  ", select_all,
                     tip="Select every person.").pack(side=tk.RIGHT, padx=4)

        # thumbstrip (ponytail: scrollbar only, no wheel binding)
        strip = tk.Frame(win, bg=AppTheme.BG_DARK)
        strip.pack(side=tk.BOTTOM, fill=tk.X, padx=10)
        strip_canvas = tk.Canvas(strip, bg=AppTheme.BG_DARK, height=190,
                                 highlightthickness=0)
        xbar = tk.Scrollbar(strip, orient=tk.HORIZONTAL, command=strip_canvas.xview)
        strip_canvas.configure(xscrollcommand=xbar.set)
        xbar.pack(side=tk.BOTTOM, fill=tk.X)
        strip_canvas.pack(fill=tk.X)
        inner = tk.Frame(strip_canvas, bg=AppTheme.BG_DARK)
        strip_canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: strip_canvas.configure(scrollregion=strip_canvas.bbox("all")))

        for i, p in enumerate(persons):
            t = p.copy()
            t.thumbnail((150, 150), Image.LANCZOS)
            ph = ImageTk.PhotoImage(t)
            photos.append(ph)
            lbl = tk.Label(inner, image=ph, text=f"Person {i + 1}", compound="top",
                           bg=AppTheme.BG_MEDIUM, fg=AppTheme.TEXT,
                           font=("Segoe UI", 9), highlightthickness=3,
                           highlightbackground=AppTheme.BG_DARK, cursor="hand2")
            lbl.pack(side=tk.LEFT, padx=4, pady=4)
            lbl.bind("<Button-1>", lambda e, i=i: toggle(i))
            ToolTip(lbl, "Click to select or deselect.")
            thumbs.append(lbl)

        # actions on selected people (packed after the strip so it sits just above it)
        act = tk.Frame(win, bg=AppTheme.BG_PANEL)
        act.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(6, 0))
        tk.Label(act, text="Selected people:", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.TEXT_DIM, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        ActionButton(act, "  Redact Face  ", redact_face, accent=True,
                     tip="Mask the face of each selected person using the current Mosaic Type and settings.").pack(side=tk.LEFT, padx=4)
        ActionButton(act, "  Redact Body  ", redact_body, accent=True,
                     tip="Mask the whole silhouette of each selected person.").pack(side=tk.LEFT, padx=4)
        ActionButton(act, "  Remove Person  ", remove_people, accent=True,
                     tip="Erase the selected people and fill in the background. Not undoable with Ctrl+Z; use Reset Image to go back.").pack(side=tk.LEFT, padx=4)

        # main preview
        preview = tk.Label(win, bg=AppTheme.CANVAS_BG)
        preview.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10)

        win.photos = photos
        show_view()
        paint()
        self._update_status(f"Found {len(persons)} person(s)")

    def delete_selected_mask(self):
        if self.selected_region:
            self._delete_region(self.selected_region)

    # ── Property Change Handlers ────────────────────────────────────────

    def _on_block_size_change(self, val):
        if self.selected_region:
            self.selected_region.block_size = int(val)
            self._refresh_canvas()

    # ADD this new method to IdentityShieldApp class:
    def _on_block_entry(self, event=None):
        """Handle custom block size typed into entry."""
        try:
            val = int(self.block_entry.get())
            val = max(3, min(val, 999))
            self.block_entry.delete(0, tk.END)
            self.block_entry.insert(0, str(val))
            # Update slider only if within its range
            if val <= 200:
                self.block_slider.set(val)
            if self.selected_region:
                self.selected_region.block_size = val
                self._refresh_canvas()
        except ValueError:
            self.block_entry.delete(0, tk.END)
            self.block_entry.insert(0, str(int(self.block_slider.get())))

    # MODIFY _on_block_size_change to sync the entry:
    def _on_block_size_change(self, val):
        int_val = int(val)
        self.block_entry.delete(0, tk.END)
        self.block_entry.insert(0, str(int_val))
        if self.selected_region:
            self.selected_region.block_size = int_val
            self._refresh_canvas()

    def _on_blur_change(self, val):
        if self.selected_region:
            self.selected_region.blur_strength = int(val)
            self._refresh_canvas()

    def _on_opacity_change(self, val):
        if self.selected_region:
            self.selected_region.opacity = float(val) / 100.0
            self._refresh_canvas()

    def _on_mask_type_change(self):
        if self.selected_region:
            self.selected_region.mask_type = self._get_current_mask_type()
            self._refresh_canvas()

    def _pick_color(self, event=None):
        color = colorchooser.askcolor(
            initialcolor=f"#{self.fill_color[0]:02x}{self.fill_color[1]:02x}{self.fill_color[2]:02x}",
            title="Select Fill Color"
        )
        if color[0]:
            self.fill_color = tuple(int(c) for c in color[0])
            hex_color = color[1]
            self.color_swatch.config(bg=hex_color)
            if self.selected_region:
                self.selected_region.fill_color = self.fill_color
                self._refresh_canvas()

    # ── Batch Operations ────────────────────────────────────────────────

    def batch_detect(self):
        if not self.file_list:
            messagebox.showwarning("No Files", "Please open a folder first.")
            return

        progress_win = tk.Toplevel(self.root)
        progress_win.title("Batch Detection")
        progress_win.geometry("400x150")
        progress_win.configure(bg=AppTheme.BG_PANEL)
        progress_win.transient(self.root)

        tk.Label(progress_win, text="Processing...", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.TEXT, font=("Segoe UI", 12)).pack(pady=15)

        progress_var = tk.DoubleVar()
        progress = ttk.Progressbar(progress_win, variable=progress_var,
                                     maximum=len(self.file_list))
        progress.pack(fill=tk.X, padx=30, pady=10)

        status_lbl = tk.Label(progress_win, text="", bg=AppTheme.BG_PANEL,
                              fg=AppTheme.TEXT_DIM, font=("Segoe UI", 9))
        status_lbl.pack()

        def process():
            total_faces = 0
            for i, fpath in enumerate(self.file_list):
                img = cv2.imread(fpath)
                if img is not None:
                    faces = ImageProcessor.detect_faces(img)
                    total_faces += len(faces)
                    progress_var.set(i + 1)
                    status_lbl.config(
                        text=f"{os.path.basename(fpath)} - {len(faces)} faces")
                    progress_win.update()

            status_lbl.config(text=f"Done! {total_faces} faces found in {len(self.file_list)} images")
            self.root.after(2000, progress_win.destroy)

        threading.Thread(target=process, daemon=True).start()

    def batch_export(self):
        if not self.file_list:
            messagebox.showwarning("No Files", "Please open a folder first.")
            return

        output_dir = filedialog.askdirectory(title="Select Output Folder",
                                           initialdir=SETTINGS["output_dir"] or None)
        if not output_dir:
            return

        confidence = self.confidence_slider.get() / 100.0
        min_size = int(self.min_face_slider.get())
        mask_type = self._get_current_mask_type()
        block_size = int(self.block_slider.get())
        blur_strength = int(self.blur_slider.get())

        progress_win = tk.Toplevel(self.root)
        progress_win.title("Batch Export")
        progress_win.geometry("400x150")
        progress_win.configure(bg=AppTheme.BG_PANEL)
        progress_win.transient(self.root)

        tk.Label(progress_win, text="Exporting...", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.TEXT, font=("Segoe UI", 12)).pack(pady=15)

        progress_var = tk.DoubleVar()
        progress = ttk.Progressbar(progress_win, variable=progress_var,
                                     maximum=len(self.file_list))
        progress.pack(fill=tk.X, padx=30, pady=10)

        status_lbl = tk.Label(progress_win, text="", bg=AppTheme.BG_PANEL,
                              fg=AppTheme.TEXT_DIM, font=("Segoe UI", 9))
        status_lbl.pack()

        def process():
            for i, fpath in enumerate(self.file_list):
                img = cv2.imread(fpath)
                if img is not None:
                    faces = ImageProcessor.detect_faces(
                        img, confidence_threshold=confidence, min_size=min_size)

                    for (x, y, w, h) in faces:
                        pad = int(w * 0.1)
                        x1 = max(0, x - pad)
                        y1 = max(0, y - pad)
                        x2 = min(img.shape[1], x + w + pad)
                        y2 = min(img.shape[0], y + h + pad)

                        region = MaskRegion(
                            region_id="batch",
                            tool_type=ToolType.AUTO_DETECT,
                            points=[(x1, y1), (x2, y2)],
                            mask_type=mask_type,
                            block_size=block_size,
                            blur_strength=blur_strength,
                            fill_color=self.fill_color,
                        )
                        img = ImageProcessor.apply_mask_region(img, region)

                    base = os.path.splitext(os.path.basename(fpath))[0]
                    out_path = os.path.join(output_dir, f"{base}_redacted.png")

                    if self.strip_exif_var.get():
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil_img = ImageProcessor.strip_exif(Image.fromarray(rgb))
                        pil_img.save(out_path, quality=95)
                    else:
                        cv2.imwrite(out_path, img)

                progress_var.set(i + 1)
                status_lbl.config(text=f"Exported: {os.path.basename(fpath)}")
                progress_win.update()

            status_lbl.config(text=f"Done! {len(self.file_list)} images exported")
            self._update_status(f"Batch export complete: {output_dir}")
            self.root.after(2000, progress_win.destroy)

        threading.Thread(target=process, daemon=True).start()

    # ── Dialogs ─────────────────────────────────────────────────────────

    def show_shortcuts(self):
        win = tk.Toplevel(self.root)
        win.title("⌨ Keyboard Shortcuts")
        win.geometry("400x500")
        win.configure(bg=AppTheme.BG_PANEL)
        win.transient(self.root)

        tk.Label(win, text="⌨ Keyboard Shortcuts", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.ACCENT, font=("Segoe UI", 14, "bold")).pack(pady=15)

        shortcuts = [
            ("Ctrl+O", "Open Image"),
            ("Ctrl+Shift+O", "Open Folder"),
            ("Ctrl+S", "Save"),
            ("Ctrl+Shift+S", "Save As"),
            ("Ctrl+E", "Export"),
            ("Ctrl+D", "Auto-Detect Faces"),
            ("Ctrl+Z", "Undo"),
            ("Ctrl+Y", "Redo"),
            # In show_shortcuts(), ADD to shortcuts list:
            ("C", "Crop Tool"),
            # In show_shortcuts(), REPLACE "RMB Drag" entry with:
            ("LMB+RMB Drag", "Pan Canvas"),
            ("MMB Drag", "Pan Canvas"),
            ("Right Click", "Context Menu"),
            # In show_shortcuts(), ADD to the shortcuts list after ("Ctrl+Y", "Redo"):
            ("Ctrl+V", "Paste from Clipboard"),
            ("", ""),
            ("V", "Select Tool"),
            ("R", "Rectangle Tool"),
            ("P", "Polygon Tool"),
            ("F", "Freehand Tool"),
            ("B", "Brush Tool"),
            ("", ""),
            ("Enter", "Finish Polygon"),
            ("Escape", "Cancel Drawing"),
            ("Delete", "Delete Selected"),
            ("+/-", "Zoom In/Out"),
            ("Ctrl+0", "Fit to Window"),
            ("Ctrl+1", "Zoom 100%"),
            ("Middle Drag", "Pan Canvas"),
            ("Scroll", "Zoom at Cursor"),
            ("Right Click", "Context Menu"),
            ("Double Click", "Finish Polygon"),
        ]

        frame = tk.Frame(win, bg=AppTheme.BG_PANEL)
        frame.pack(fill=tk.BOTH, expand=True, padx=20)

        for key, desc in shortcuts:
            if not key:
                tk.Frame(frame, bg=AppTheme.BORDER, height=1).pack(
                    fill=tk.X, pady=3)
                continue
            row = tk.Frame(frame, bg=AppTheme.BG_PANEL)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=key, bg=AppTheme.BG_DARK, fg=AppTheme.ACCENT,
                     font=("Consolas", 9), width=18, anchor="e",
                     padx=5, pady=2).pack(side=tk.LEFT)
            tk.Label(row, text=desc, bg=AppTheme.BG_PANEL,
                     fg=AppTheme.TEXT, font=("Segoe UI", 9),
                     anchor="w", padx=10).pack(side=tk.LEFT, fill=tk.X)

    def show_about(self):
        win = tk.Toplevel(self.root)
        win.title("About IdentityShield Pro")
        win.geometry("400x300")
        win.configure(bg=AppTheme.BG_PANEL)
        win.transient(self.root)

        tk.Label(win, text="🛡️", bg=AppTheme.BG_PANEL,
                 font=("Segoe UI", 48)).pack(pady=(20, 5))
        tk.Label(win, text="IdentityShield Pro", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.ACCENT, font=("Segoe UI", 18, "bold")).pack()
        tk.Label(win, text="v1.0.0", bg=AppTheme.BG_PANEL,
                 fg=AppTheme.TEXT_DIM, font=("Segoe UI", 10)).pack()
        tk.Label(win, text="\nImage Privacy Protection Tool\n"
                          "Protect identities with face detection,\n"
                          "mosaic, blur, and manual masking tools.\n\n"
                          "Supports batch processing & EXIF stripping.",
                 bg=AppTheme.BG_PANEL, fg=AppTheme.TEXT,
                 font=("Segoe UI", 10), justify="center").pack()

    # ── Utility ─────────────────────────────────────────────────────────

    def _update_status(self, msg):
        self.status_label.config(text=msg)


# ─── Entry Point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()

    # Set icon if possible
    try:
        root.iconbitmap(default="")
    except:
        pass

    app = IdentityShieldApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
