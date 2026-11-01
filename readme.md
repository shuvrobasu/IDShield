# 🛡️ IdentityShield Pro

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A desktop privacy tool for redacting faces, bodies and text in images and PDFs — with AI person detection, one-click selective redaction, and clean person removal.**

Built with Python and Tkinter. Runs entirely on your own machine: nothing is uploaded anywhere.

<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/eb71b01b-e40a-4d74-9d5e-438d27ad56ff" />


---

## Why

Sharing a photo or document often means hiding someone: a bystander's face, a colleague in a group shot, a name in a PDF. Doing that by hand in an editor is slow, and a lazy blur is easy to get wrong.

IdentityShield Pro finds the people for you, lets you choose exactly who gets redacted and how, and can erase a person from the picture altogether.

## Highlights

- **Auto-Detect faces** — including tilted and profile faces, with an adjustable sensitivity slider.
- **Detect People** — finds every person in the photo and shows each one as a thumbnail.
  - Redact the **face**, the whole **body**, or **remove the person** and fill in the background.
  - Save every person as a transparent PNG cut-out, or save all of them as a single humans-only image.
- **Manual masks** — rectangle, polygon, freehand and brush tools for anything the detector misses.
- **Four mask styles** — Pixelate, Gaussian Blur, Solid Fill, Noise Fill, each with adjustable strength and opacity.
- **PDF redaction** — permanent redaction of masked areas, plus search and regex redaction across every page.
- **Remove Background** and **Remove Text** tools.
- **Batch mode** — detect and export a whole folder.
- **Everything configurable** — model files and folders are set in a Settings dialog, nothing is hard-coded.
- **Recent Files** menu and **hover tooltips** on every control.

<img width="1002" height="812" alt="image" src="https://github.com/user-attachments/assets/62e068f9-401d-477d-b7ba-3ab6357bfa35" />
<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/f625e29f-c098-4869-9f81-d5fcddf965e0" />
<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/e0c0da5d-c7de-4367-b25d-db8f322ad1c9" />
<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/5f7b492c-b196-4d1f-90fb-5502aa78185a" />

<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/a899217d-a61c-429a-9dc9-98dd0846861f" />
<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/00c29058-ae9e-45b7-a79b-9a94c0f9915b" />
(Yes face redact works on imgs in PDF too!)






---

## Features in detail

### Face redaction

| | |
|---|---|
| **Auto-Detect** | YOLO face model finds frontal, tilted and profile faces. Falls back to a basic OpenCV detector if the model is not configured. The status bar shows which engine was used. |
| **Confidence** | Lower finds more faces (and more false hits). Default 25%. |
| **Min Face Size** | Ignore faces smaller than a pixel size you choose. |

### Detect People

1. Click **Detect People** (left panel or *Tools* menu) and pick a model size.
2. A results window shows everyone found:
   - **Cutout** view: the people on a transparent background.
   - **Silhouette** view: everyone as one solid shape.
   - A **thumbnail strip** with one entry per person. Click to select.
3. With people selected:
   - **Redact Face** — masks the face using your current mask style.
   - **Redact Body** — masks the whole outline.
   - **Remove Person** — erases the person and fills in the background.
4. **Save Selected / Save All** writes each person as a transparent PNG. **Save View** saves the Cutout or Silhouette image.

Notes on detection:

- A person is only listed if a **face is visible**. Arms, clothing and people turned away are not counted.
- Duplicates and fragments are removed, overlapping people are separated, and small gaps inside an outline are filled.
- Masks you create appear in the **Mask List** and can be edited, hidden or deleted like any other mask. Removing a person also removes any masks that covered them.

### Remove Person

Uses **LaMa** inpainting when installed for natural-looking fills. Without it the tool falls back to a basic OpenCV fill, which is fine for small areas but smears on large ones. The status bar tells you which engine was used.

> Remove Person changes the image pixels, so **Ctrl+Z does not undo it**. Use **Edit → Reset Image** to go back to the original.

### Manual masks

Rectangle, Polygon (click points, double-click or Enter to finish), Freehand, and Brush. Select, move and resize masks; right-click a mask for delete, visibility, change type and duplicate. Undo/redo covers up to 50 steps of mask changes.

### PDF

- Open a PDF and add masks on any page.
- **Save Redacted PDF** burns the redactions in permanently.
- **Search/Regex Redact** finds text across all pages and redacts every match.
- Works on the PDF's text layer; scanned pages need masks drawn by hand.

### Other tools

- **Remove Background** — uses `rembg` if installed, otherwise OpenCV GrabCut.
- **Remove Text** — detects text with the EAST model, MSER or EasyOCR, then removes it (Telea or Navier-Stokes inpainting) or just creates masks for it.
- **Crop**, **Paste from clipboard**, **Zoom/pan**.
- **Export** to PNG, JPEG, BMP, TIFF or WEBP, with an option to **strip EXIF/metadata**.
- **Batch** — *Detect All in Folder* and *Export All Processed*.
- **Recent Files** under the File menu remembers your last 10 files.

---

## Installation

**Requirements:** Python 3.10 or newer. Developed on Windows; a CUDA GPU is strongly recommended for Detect People.

```bash
git clone https://github.com/shuvrobasu/idshield.git
cd idshield

pip install opencv-python numpy pillow pymupdf
pip install torch torchvision          # pick the build matching your CUDA version: https://pytorch.org
pip install ultralytics
```

Install **SAM 2** (needed for Detect People):

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

Optional extras:

```bash
pip install rembg                                  # better Remove Background
pip install easyocr                                # alternative text detector
pip install simple-lama-inpainting --no-deps       # clean Remove Person fills
```

> Install `simple-lama-inpainting` with `--no-deps` so it does not replace your existing PyTorch build.

Run it:

```bash
python all_mosaic.py
```

### Model files

| Model | Used for | Where to get it |
|---|---|---|
| SAM 2 checkpoints (`sam2_hiera_large.pt`, `sam2_hiera_base_plus.pt`, `sam2_hiera_small.pt`, `sam2_hiera_tiny.pt`) | Person outlines | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| `yolov8x.pt` | Finding people | Downloaded automatically by Ultralytics on first use |
| `yolov8l-face.pt` | Face detection | [akanametov/yolo-face](https://github.com/akanametov/yolo-face) |
| `frozen_east_text_detection.pb` | Remove Text (optional) | EAST text detector for OpenCV |

SAM 2 model sizes trade speed for accuracy. **Base+** is the default and works well on most photos.

### Settings

Open **⚙ Settings** (left panel, or *File → Settings*) and point the app at your files:

- SAM 2 folder and the list of SAM 2 models (add or remove entries, choose the default)
- Person model, face model and EAST text model
- Default output folder

Everything is saved to `settings.sec` (plain JSON) next to the script. Nothing is hard-coded, and missing values fall back to safe defaults. Changes apply immediately, with no restart.

Add `settings.sec` to your `.gitignore` — it contains paths from your machine:

```
settings.sec
```

---

## Keyboard shortcuts

| Action | Keys |
|---|---|
| Open image / folder | `Ctrl+O` / `Ctrl+Shift+O` |
| Save / Save As | `Ctrl+S` / `Ctrl+Shift+S` |
| Export | `Ctrl+E` |
| Undo / Redo | `Ctrl+Z` / `Ctrl+Y` |
| Paste from clipboard | `Ctrl+V` |
| Auto-detect faces | `Ctrl+D` or `D` |
| Zoom in / out | `+` / `-` |
| Fit to window / 100% | `Ctrl+0` / `Ctrl+1` |
| Select / Rectangle / Polygon / Freehand / Brush / Crop | `V` / `R` / `P` / `F` / `B` / `C` |
| Finish polygon | `Enter` or double-click |
| Cancel drawing | `Esc` |
| Delete selected mask | `Delete` |

Mouse: wheel to zoom, middle-button drag (or left+right) to pan, right-click a mask for its menu.

---

## How Detect People works

```
photo ──► YOLOv8 finds person boxes
          └─► YOLO face model: keep only boxes that own a visible face
              └─► SAM 2 draws an exact outline inside each box
                  └─► clean-up: remove duplicates, separate overlaps, fill gaps
                      └─► Cutout / Silhouette / thumbnails / redact / remove
```

---

## Limitations and responsible use

- **Always review the result before sharing.** Automatic detection can miss faces, especially very small, heavily covered or low-resolution ones.
- For anything sensitive, prefer **Solid Fill**. Pixelation and blur are weaker and can sometimes be partly reversed.
- Inpainting (Remove Person, Remove Text, Remove Background) produces a plausible fill, not a forensic-grade erasure.
- People without a visible face are not detected by Detect People; use the manual tools for them.
- Remove Person cannot be undone with Ctrl+Z (use Reset Image).
- PDF redaction works on the text layer; scanned pages need manual masks.
- Without the YOLO face model configured, face detection falls back to a basic frontal-only detector.
- Use the tool only on images you have the right to edit.

---

## Third-party components

| Component | Licence |
|---|---|
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| [PyMuPDF](https://github.com/pymupdf/PyMuPDF) | AGPL-3.0 |
| [SAM 2](https://github.com/facebookresearch/sam2) | Apache-2.0 |
| [LaMa / simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) | Apache-2.0 |
| [OpenCV](https://opencv.org) | Apache-2.0 |
| [yolo-face weights](https://github.com/akanametov/yolo-face) | Check the repository's licence before redistributing |

These components keep their own licences. The licence below covers only this project's own code and documentation, not them.

## License

This work is licensed under the [Creative Commons Attribution-NoDerivatives 4.0 International License (CC BY-ND 4.0)](https://creativecommons.org/licenses/by-nd/4.0/).

You are free to **share** — copy and redistribute the material in any medium or format, for any purpose, even commercially — under these terms:

- **Attribution** — you must give appropriate credit, provide a link to the licence, and indicate if changes were made.
- **NoDerivatives** — if you remix, transform, or build upon the material, you may not distribute the modified material.

© Shuvro Basu
