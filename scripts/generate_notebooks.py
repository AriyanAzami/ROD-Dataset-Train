"""
Generate clean, consistent Kaggle training notebooks for the ROD-Dataset.

One notebook per YOLO nano-scale variant. Every notebook is identical in
structure and only differs by the model weights it loads + a short note, so a
viewer who reads one understands all of them.

Run:  python scripts/generate_notebooks.py
Output: notebooks/<variant>-rod.ipynb
"""
import json
from pathlib import Path

# ---------------------------------------------------------------- variants ---
# (variant_stem, pretty_title, version_note)
VARIANTS = [
    ("yolov8n",  "YOLOv8n",  "Stable baseline. Works on any recent `ultralytics`."),
    ("yolov9t",  "YOLOv9t",  "YOLOv9 'tiny' is the nano-equivalent (there is no `yolov9n`)."),
    ("yolov10n", "YOLOv10n", "End-to-end NMS-free head; great latency/accuracy trade-off."),
    ("yolov11n", "YOLO11n",  "Ultralytics' current default family. File stem is `yolo11n`."),
    ("yolov12n", "YOLO12n",  "Attention-centric family. File stem is `yolo12n`; needs a fresh `ultralytics`."),
    ("yolo26n",  "YOLO26n",  "Newest family — needs a fresh `ultralytics` (the `-U` install below handles it)."),
]

# yolov11n's actual weights stem is "yolo11n" (no 'v'); map stem -> weights file.
WEIGHTS_STEM = {
    "yolov8n":  "yolov8n",
    "yolov9t":  "yolov9t",
    "yolov10n": "yolov10n",
    "yolov11n": "yolo11n",
    "yolov12n": "yolo12n",
    "yolo26n":  "yolo26n",
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.rstrip("\n").splitlines(keepends=True)}


def build_cells(stem, title, note):
    weights = WEIGHTS_STEM[stem]
    cells = []

    cells.append(md(f"""# Train {title} on the ROD-Dataset

**Real-Time Obstacle Detection** — 25 classes, ~24,326 images.
Dataset: https://www.kaggle.com/datasets/abtinzandi/obstacle-detection-dataset

**Code by Ariyan Azami**

> {note}

## Before you run (one-time Kaggle setup)
1. **Settings (right sidebar) → Accelerator → `GPU T4 x2`.**
2. **Settings → Internet → ON** (so `ultralytics` installs and pretrained weights download).
3. **+ Add Input → Datasets →** search `obstacle-detection-dataset` by `abtinzandi` and attach it.
   It mounts read-only under `/kaggle/input/...`; the cells below locate it automatically.

Everything is driven by the **Config** cell — edit it and *Run All*.
"""))

    cells.append(md("## 1 · Config — edit me, then Run All"))
    cells.append(code(f"""# ============================ CONFIG ============================
MODEL_VARIANT = "{stem}"           # which model this notebook trains
WEIGHTS       = "{weights}.pt"     # pretrained checkpoint to fine-tune from

EPOCHS   = 100        # training length (lower for a quick smoke test)
BATCH    = 128        # TOTAL images per step, split across GPUs (64/GPU on 2x T4)
IMGSZ    = 640        # train/val image size
DEVICE   = "0,1"      # use BOTH T4 GPUs for training (DDP). Set to 0 for one GPU.
EVAL_DEVICE = 0       # val/predict/benchmark always run on a single GPU
PATIENCE = 20         # early-stop if val mAP stalls this many epochs
COS_LR   = True       # cosine LR schedule
SEED     = 42

CONF     = 0.25       # confidence threshold for eval/predict/timing
IOU      = 0.7        # NMS IoU threshold

PROJECT  = "/kaggle/working/runs"
RUN_NAME = f"rod_{{MODEL_VARIANT}}"
# ===============================================================
print(f"Training {{MODEL_VARIANT}} from {{WEIGHTS}} | {{EPOCHS}} epochs | batch {{BATCH}} | imgsz {{IMGSZ}} | device {{DEVICE}}")"""))

    cells.append(md("## 2 · Install & environment check"))
    cells.append(code("""!pip install -q -U ultralytics"""))
    cells.append(code("""import torch, ultralytics
print("ultralytics:", ultralytics.__version__)
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({total/1e9:.0f} GB)")

# Auto-correct DEVICE if fewer GPUs are available than requested (e.g. P100 x1 instead of T4 x2)
if isinstance(DEVICE, str) and "," in DEVICE:
    requested = len(DEVICE.split(","))
    available = torch.cuda.device_count()
    if available < requested:
        DEVICE = ",".join(str(i) for i in range(available)) if available > 0 else "cpu"
        print(f"WARNING: Only {available} GPU(s) available — DEVICE overridden to '{DEVICE}'")"""))

    cells.append(md("""## 3 · Locate the dataset & write a fresh `data.yaml`

`/kaggle/input/` is read-only and the mount path differs per dataset version, so
we search for the dataset's own `data.yaml`, then write a corrected copy into
`/kaggle/working/` with absolute split paths."""))
    cells.append(code("""import yaml
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
src = next((p for p in INPUT_ROOT.rglob("data.yaml")), None)
assert src is not None, "data.yaml not found under /kaggle/input — did you Add the dataset?"
data_root = src.parent
print("Dataset root:", data_root)

with open(src) as f:
    orig = yaml.safe_load(f)

data_yaml = {
    "path":  str(data_root),
    "train": "train/images",
    "val":   "valid/images",
    "test":  "test/images",
    "nc":    orig["nc"],
    "names": orig["names"],
}
DATA = "/kaggle/working/data.yaml"
with open(DATA, "w") as f:
    yaml.safe_dump(data_yaml, f, sort_keys=False)

print("Wrote", DATA, "|", data_yaml["nc"], "classes")
print(data_yaml["names"])"""))

    cells.append(md("""## 4 · Train

By default `DEVICE = "0,1"` trains on **both** T4s via PyTorch DDP (`BATCH` is the
total, split across GPUs). If DDP misbehaves in the notebook, set `DEVICE = 0` to
fall back to a single GPU. The dataset is a *detect-segment mixed* set —
Ultralytics keeps the boxes and drops segments automatically (you'll see a
one-time warning, which is expected)."""))
    cells.append(code("""from ultralytics import YOLO

model = YOLO(WEIGHTS)
results = model.train(
    data     = DATA,
    task     = "detect",
    epochs   = EPOCHS,
    imgsz    = IMGSZ,
    batch    = BATCH,
    device   = DEVICE,
    amp      = True,
    cache    = False,
    workers  = 4,
    project  = PROJECT,
    name     = RUN_NAME,
    patience = PATIENCE,
    cos_lr   = COS_LR,
    plots    = True,
    seed     = SEED,
)
# Multi-GPU DDP training in a notebook relaunches as a subprocess and returns
# None in the parent process, so `results` may be None even though training
# finished. Fall back to locating the run directory under PROJECT/RUN_NAME.
if results is not None and getattr(results, "save_dir", None):
    RUN_DIR = Path(results.save_dir)
else:
    candidates = sorted(Path(PROJECT).glob(RUN_NAME + "*"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"No run directory found under {PROJECT}/{RUN_NAME}*")
    RUN_DIR = candidates[-1]
print("Run dir:", RUN_DIR)"""))

    cells.append(md("""## 5 · Validate on the held-out **test** split

We reload `best.pt` (more reliable than reusing the live training object) and
run `val()` on the test split — this reports mAP50, mAP50-95, precision, recall
and writes confusion matrices + PR/F1 curves."""))
    cells.append(code("""best = RUN_DIR / "weights" / "best.pt"
eval_model = YOLO(str(best))

metrics = eval_model.val(
    data   = DATA,
    task   = "detect",
    split  = "test",
    imgsz  = IMGSZ,
    batch  = BATCH,
    device = EVAL_DEVICE,   # validate on a single GPU (DDP val in notebooks is flaky)
    conf   = CONF,
    iou    = IOU,
    plots  = True,
)
VAL_DIR = Path(metrics.save_dir)
print(metrics.results_dict)"""))

    cells.append(md("""## 6 · Inference speed benchmark (latency / FPS)

Warm up, then time `predict()` over the whole test set — the same methodology as
the project's `time_inference.py`."""))
    cells.append(code("""import time, glob

def benchmark(m, image_paths, imgsz, device=0, conf=0.25, iou=0.7, batch=8, warmup=5):
    for p in image_paths[:warmup]:
        _ = m.predict(p, imgsz=imgsz, device=device, conf=conf, iou=iou, verbose=False)
    t0 = time.perf_counter(); n = 0
    for i in range(0, len(image_paths), batch):
        chunk = image_paths[i:i + batch]
        _ = m.predict(chunk, imgsz=imgsz, device=device, conf=conf, iou=iou, verbose=False)
        n += len(chunk)
    dt = time.perf_counter() - t0
    return (dt / n) * 1000.0, n / dt, dt

test_imgs = sorted(glob.glob(str(data_root / "test/images/*.jpg")))
avg_ms, fps, total_s = benchmark(eval_model, test_imgs, IMGSZ, device=EVAL_DEVICE,
                                 conf=CONF, iou=IOU, batch=BATCH)
print(f"{avg_ms:.2f} ms/img | {fps:.1f} FPS | {total_s:.1f}s over {len(test_imgs)} images")"""))

    cells.append(md("""## 7 · Save a machine-readable summary

Writes `<variant>_summary.json` and `<variant>_results.txt` to
`/kaggle/working/` so the **compare** notebook can chart every model side by
side, and viewers can download the numbers."""))
    cells.append(code("""import json
from datetime import datetime

rd = metrics.results_dict
summary = {
    "model":          MODEL_VARIANT,
    "weights":        str(best),
    "timestamp":      datetime.now().isoformat(timespec="seconds"),
    "epochs":         EPOCHS,
    "imgsz":          IMGSZ,
    "batch":          BATCH,
    "params":         int(sum(p.numel() for p in eval_model.model.parameters())),
    "mAP50_95":       rd.get("metrics/mAP50-95(B)"),
    "mAP50":          rd.get("metrics/mAP50(B)"),
    "precision":      rd.get("metrics/precision(B)"),
    "recall":         rd.get("metrics/recall(B)"),
    "avg_ms_per_img": round(avg_ms, 3),
    "fps":            round(fps, 2),
    "test_images":    len(test_imgs),
}
out_json = Path("/kaggle/working") / f"{MODEL_VARIANT}_summary.json"
out_json.write_text(json.dumps(summary, indent=2))
out_txt = Path("/kaggle/working") / f"{MODEL_VARIANT}_results.txt"
out_txt.write_text("\\n".join(f"{k}: {v}" for k, v in summary.items()))
print(json.dumps(summary, indent=2))"""))

    cells.append(md("## 8 · Training & evaluation graphs"))
    cells.append(code("""import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show(path, title=""):
    path = Path(path)
    if not path.exists():
        print("[skip]", path.name, "not found"); return
    plt.figure(figsize=(16, 8))
    plt.imshow(mpimg.imread(str(path)))
    plt.axis("off")
    if title:
        plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()

show(RUN_DIR / "results.png",                     "Training curves")
show(VAL_DIR / "confusion_matrix_normalized.png", "Confusion matrix (normalised)")
show(VAL_DIR / "confusion_matrix.png",            "Confusion matrix (raw)")
show(VAL_DIR / "PR_curve.png",                    "Precision-Recall curve")
show(VAL_DIR / "F1_curve.png",                    "F1-Confidence curve")
show(VAL_DIR / "P_curve.png",                     "Precision-Confidence curve")
show(VAL_DIR / "R_curve.png",                     "Recall-Confidence curve")"""))

    cells.append(md("## 9 · Predictions on sample test images"))
    cells.append(code("""import random
random.seed(SEED)
samples = random.sample(test_imgs, min(6, len(test_imgs)))

eval_model.predict(samples, imgsz=IMGSZ, conf=CONF, device=EVAL_DEVICE, save=True,
                   project=PROJECT, name="predict", exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
pred_imgs = sorted(glob.glob(f"{PROJECT}/predict/*.jpg"))[:6]
for ax, p in zip(axes.flat, pred_imgs):
    ax.imshow(mpimg.imread(p)); ax.axis("off")
for ax in axes.flat[len(pred_imgs):]:
    ax.axis("off")
plt.suptitle(f"{MODEL_VARIANT} predictions (conf >= {CONF})", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()"""))

    cells.append(md("## 10 · Save weights for download"))
    cells.append(code("""import shutil
for w in (RUN_DIR / "weights" / "best.pt", RUN_DIR / "weights" / "last.pt"):
    if w.exists():
        dst = Path("/kaggle/working") / f"{MODEL_VARIANT}_{w.name}"
        shutil.copy(str(w), str(dst))
        print("Saved", dst)"""))

    cells.append(md("""## 11 · (Optional) Export for mobile / edge

Uncomment what you need. Mirrors the deployment formats used by the ROD project
(ONNX for desktop, TFLite/NCNN for Android)."""))
    cells.append(code("""# eval_model.export(format="onnx",  imgsz=IMGSZ, opset=12)
# eval_model.export(format="tflite", imgsz=320, int8=True)
# eval_model.export(format="ncnn",  imgsz=320)"""))

    return cells


def build_compare_cells():
    cells = []
    cells.append(md("""# Compare all ROD models

**Code by Ariyan Azami**

Loads every `*_summary.json` produced by the training notebooks and builds a
side-by-side table + charts of accuracy vs. speed vs. size.

**How to use on Kaggle:** after running the training notebooks, download each
`<variant>_summary.json` from their Output and drop them into a `results/`
folder in this repo (or add them as an input dataset), then point `SEARCH_DIRS`
at that folder. Locally, this just reads `results/`."""))
    cells.append(code("""from pathlib import Path
import json, glob

# Where to look for <variant>_summary.json files.
SEARCH_DIRS = ["results", "/kaggle/working", "/kaggle/input"]

rows = []
seen = set()
for d in SEARCH_DIRS:
    for fp in glob.glob(str(Path(d) / "**" / "*_summary.json"), recursive=True):
        data = json.loads(Path(fp).read_text())
        key = data.get("model")
        if key and key not in seen:
            seen.add(key); rows.append(data)

assert rows, "No *_summary.json found — run the training notebooks first."
print(f"Loaded {len(rows)} model summaries:", [r['model'] for r in rows])"""))
    cells.append(code("""import pandas as pd

df = pd.DataFrame(rows)
order = ["model", "params", "mAP50_95", "mAP50", "precision", "recall",
         "avg_ms_per_img", "fps", "epochs", "imgsz"]
df = df[[c for c in order if c in df.columns]].sort_values("mAP50_95", ascending=False)
df["params (M)"] = (df["params"] / 1e6).round(2)
df"""))
    cells.append(code("""import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
d = df.set_index("model")

d["mAP50_95"].plot.bar(ax=axes[0], color="#3b7dd8")
axes[0].set_title("mAP50-95 (higher = better)"); axes[0].set_ylabel("mAP50-95")

d["fps"].plot.bar(ax=axes[1], color="#2ca02c")
axes[1].set_title("Throughput FPS (higher = better)"); axes[1].set_ylabel("FPS")

d["params (M)"].plot.bar(ax=axes[2], color="#d8743b")
axes[2].set_title("Parameters (lower = smaller)"); axes[2].set_ylabel("Million params")

for ax in axes:
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()"""))
    cells.append(code("""# Accuracy vs. speed trade-off — the money chart
plt.figure(figsize=(9, 7))
for _, r in df.iterrows():
    plt.scatter(r["avg_ms_per_img"], r["mAP50_95"], s=120)
    plt.annotate(r["model"], (r["avg_ms_per_img"], r["mAP50_95"]),
                 textcoords="offset points", xytext=(8, 4))
plt.xlabel("Latency (ms / image)  — lower is better")
plt.ylabel("mAP50-95  — higher is better")
plt.title("ROD models: accuracy vs. latency")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()"""))
    cells.append(code("""# Save a combined CSV for the repo / dataset page
out = Path("results"); out.mkdir(exist_ok=True)
df.to_csv(out / "comparison.csv", index=False)
print("Wrote", out / "comparison.csv")"""))
    return cells


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    out_dir = Path(__file__).resolve().parent.parent / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, title, note in VARIANTS:
        nb = notebook(build_cells(stem, title, note))
        path = out_dir / f"{stem}-rod.ipynb"
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("Wrote", path)

    compare = notebook(build_compare_cells())
    cpath = out_dir / "compare-models.ipynb"
    cpath.write_text(json.dumps(compare, indent=1), encoding="utf-8")
    print("Wrote", cpath)


if __name__ == "__main__":
    main()
