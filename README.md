# ROD-Dataset-Train

Training and benchmarking pipeline for compact, real-time object detectors on the
**ROD-Dataset** (Real-Time Obstacle Detection). This repository provides a set of
self-contained, reproducible Kaggle notebooks that fine-tune several *nano-scale*
YOLO architectures on the dataset and report a complete suite of accuracy and
inference-speed metrics.

The work targets resource-constrained, real-time deployment (mobile and edge
devices), where small models with low latency are required without an
unacceptable loss in detection accuracy.

**Dataset:** [`abtinzandi/obstacle-detection-dataset`](https://www.kaggle.com/datasets/abtinzandi/obstacle-detection-dataset)
— 25 obstacle classes, ≈ 24,326 images, split into train / validation / test.

## Models

We benchmark one nano-scale detector per generation of the YOLO family so that
accuracy, latency, and parameter count can be compared on a common dataset and
evaluation protocol.

| Notebook | Model | Notes |
|---|---|---|
| [`notebooks/yolov8n-rod.ipynb`](notebooks/yolov8n-rod.ipynb)   | YOLOv8n  | Stable baseline |
| [`notebooks/yolov9t-rod.ipynb`](notebooks/yolov9t-rod.ipynb)   | YOLOv9t  | v9 nano-equivalent ("tiny") |
| [`notebooks/yolov10n-rod.ipynb`](notebooks/yolov10n-rod.ipynb) | YOLOv10n | NMS-free end-to-end detection head |
| [`notebooks/yolov11n-rod.ipynb`](notebooks/yolov11n-rod.ipynb) | YOLO11n  | Ultralytics default family |
| [`notebooks/yolov12n-rod.ipynb`](notebooks/yolov12n-rod.ipynb) | YOLO12n  | Attention-centric architecture |
| [`notebooks/yolo26n-rod.ipynb`](notebooks/yolo26n-rod.ipynb)   | YOLO26n  | Newest YOLO family |
| [`notebooks/compare-models.ipynb`](notebooks/compare-models.ipynb) | — | Aggregates all results into comparison tables + charts |

## Methodology

Each model is initialised from publicly available COCO-pretrained weights and
fine-tuned on the ROD-Dataset under an identical protocol so that differences in
the reported metrics reflect the architectures rather than the training setup.

**Training configuration (default):**

- Epochs: 100, with early stopping (patience 20 on validation mAP)
- Batch size: 128 (total, split across GPUs)
- Image size: 640 × 640
- Optimiser schedule: cosine learning-rate decay
- Mixed-precision (AMP) training
- Hardware: 2 × NVIDIA Tesla T4 via PyTorch Distributed Data Parallel (DDP)
- Fixed random seed for reproducibility

The dataset contains a mixture of detection and segmentation annotations; the
training pipeline retains the bounding boxes and discards segmentation masks, so
all models are trained and evaluated as pure object detectors.

## Evaluation

Every model is evaluated on the **held-out test split** using a single,
consistent protocol:

- **Accuracy:** mAP@0.50, mAP@0.50–0.95, precision, recall, together with
  confusion matrices and precision–recall / F1 / precision / recall curves.
- **Inference speed:** average latency (ms / image) and throughput (FPS),
  measured with warm-up followed by timed prediction over the full test set.
- **Model size:** total number of parameters.

Each training notebook writes a machine-readable `<model>_summary.json` and a
human-readable `<model>_results.txt`. The comparison notebook collects these
summaries and produces side-by-side tables and accuracy-versus-latency plots.

## Running on Kaggle

1. Create a new Kaggle notebook and **import** one of `notebooks/*.ipynb`, or add
   this repository as a GitHub input.
2. **Settings → Accelerator → `GPU T4 x2`** and **Settings → Internet → ON**
   (required for installing dependencies and downloading pretrained weights).
3. **+ Add Input → Datasets →** search for `obstacle-detection-dataset` by
   `abtinzandi` and attach it. The dataset mounts read-only under
   `/kaggle/input/`; the notebooks locate it automatically.
4. Optionally edit the **Config** cell (epochs, batch, image size, device), then
   **Run All**.

The dataset mount path varies between dataset versions, so the notebooks search
for the dataset's `data.yaml`, then write a corrected copy with absolute split
paths to `/kaggle/working/data.yaml`.

Outputs are written to `/kaggle/working/` and are available under the notebook's
Output tab: the metrics summary, the results text file, the best/last weights,
and all training and evaluation graphs.

To train on a single GPU, set `DEVICE = 0` in the Config cell. Validation and the
speed benchmark always run on a single GPU.

## Repository layout

```
notebooks/                       generated training notebooks + comparison notebook
scripts/generate_notebooks.py    single source of truth for all notebooks
src/rod_eval/                    reusable accuracy + inference-speed helpers
src/evaluate.py                  headless CLI evaluation entry point
results/                         model summaries consumed by the comparison notebook
requirements.txt                 dependencies for local / headless evaluation
```

## Regenerating the notebooks

The training notebooks are generated programmatically to guarantee that every
variant follows an identical structure. Do not hand-edit the notebooks; edit the
generator and regenerate:

```bash
python scripts/generate_notebooks.py
```

## Local / headless evaluation

The evaluation logic is also packaged under `src/` for use outside Kaggle:

```bash
pip install -r requirements.txt
python src/evaluate.py --model path/to/best.pt --data_yaml path/to/data.yaml --test_dir path/to/test/images
```
