"""Headless evaluation CLI — accuracy + speed for a trained .pt on the test split.

Mirrors the reference project's `test_and_validation.py`. Example:

    python src/evaluate.py \
        --model runs/rod_yolov8n/weights/best.pt \
        --data_yaml data.yaml \
        --test_dir /path/to/dataset/test \
        --imgsz 640 --device 0 --batch 8
"""
import argparse
import glob
import sys
from pathlib import Path

from ultralytics import YOLO

from rod_eval import (
    calculate_accuracy_metrics,
    calculate_time_inference,
    print_and_save_results,
)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def collect_images(test_dir: Path):
    img_dir = test_dir / "images"
    files = sorted(f for ext in IMG_EXTS for f in glob.glob(str(img_dir / f"*{ext}")))
    if not files:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return files


def main():
    ap = argparse.ArgumentParser(description="Evaluate a YOLO .pt on a YOLO-format test set.")
    ap.add_argument("--model", required=True, help="Path to .pt weights.")
    ap.add_argument("--data_yaml", required=True, help="Path to data.yaml.")
    ap.add_argument("--test_dir", required=True, help="Folder containing test/images and test/labels.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="'cpu', '0', or 'cuda:0'.")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="eval_run")
    ap.add_argument("--project", default="runs/eval")
    args = ap.parse_args()

    model_path, test_dir = Path(args.model), Path(args.test_dir)
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        sys.exit(f"ERROR: model not found: {model_path}")
    if not (test_dir / "images").exists() or not (test_dir / "labels").exists():
        sys.exit(f"ERROR: {test_dir} must contain images/ and labels/ subfolders")

    model = YOLO(str(model_path))

    map_50_95, map_50, precision, recall = calculate_accuracy_metrics(
        model, args.data_yaml, split="test", imgsz=args.imgsz, batch=args.batch,
        device=args.device, conf=args.conf, iou=args.iou,
    )
    image_paths = collect_images(test_dir)
    avg_ms, fps, total_s = calculate_time_inference(
        model, image_paths, imgsz=args.imgsz, device=args.device,
        conf=args.conf, iou=args.iou, batch=args.batch,
    )
    print_and_save_results(
        model_path, test_dir, image_paths, args.imgsz, args.device, args.batch,
        map_50_95, map_50, precision, recall, avg_ms, fps, total_s,
        out_dir / "results.txt", out_dir.resolve(),
    )


if __name__ == "__main__":
    main()
