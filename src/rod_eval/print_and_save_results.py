from datetime import datetime


def print_and_save_results(model_path, test_dir, image_paths, imgsz, device, batch,
                           map_50_95, map_50, precision, recall, avg_ms, fps,
                           total_s, results_txt, out_dir):
    """Format accuracy + speed metrics, print them, and write to ``results_txt``."""
    lines = [
        f"YOLO Evaluation — {datetime.now().isoformat(timespec='seconds')}",
        f"Model: {model_path}",
        f"Test dir: {test_dir}  (#images: {len(image_paths)})",
        f"Image size: {imgsz} | Device: {device or 'auto'} | Batch: {batch}",
        "",
        "== Accuracy (Ultralytics val on test split) ==",
        f"mAP50-95: {map_50_95:.4f}",
        f"mAP50:    {map_50:.4f}",
    ]
    if precision == precision:  # not NaN
        lines.append(f"Precision (mean): {precision:.4f}")
    if recall == recall:        # not NaN
        lines.append(f"Recall (mean):    {recall:.4f}")
    lines += [
        "",
        "== Inference speed ==",
        f"Average inference time per image: {avg_ms:.2f} ms",
        f"Throughput (FPS): {fps:.2f}",
        f"Total wall time on {len(image_paths)} images: {total_s:.2f} s",
        "",
    ]
    report = "\n".join(lines)
    print(report)
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved results to: {results_txt}")
    print(f"(Project outputs in: {out_dir})")
