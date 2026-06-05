import time


def calculate_time_inference(model, image_paths, imgsz=640, device=None,
                             conf=0.25, iou=0.7, batch=1, warmup=5):
    """Measure inference latency and throughput over a list of images.

    Warms up on the first ``warmup`` images (to stabilise GPU timing), then times
    ``predict()`` over the whole list in batches.

    Returns:
        tuple[float, float, float]: (avg_ms_per_image, fps, total_seconds).
    """
    if not image_paths:
        raise ValueError("image_paths is empty — cannot measure inference time.")

    for p in image_paths[:warmup]:
        _ = model.predict(p, imgsz=imgsz, device=device, conf=conf, iou=iou, verbose=False)

    t0 = time.perf_counter()
    count = 0
    for i in range(0, len(image_paths), batch):
        chunk = image_paths[i:i + batch]
        _ = model.predict(chunk, imgsz=imgsz, device=device, conf=conf, iou=iou, verbose=False)
        count += len(chunk)
    total_s = time.perf_counter() - t0

    avg_ms = (total_s / count) * 1000.0
    fps = count / total_s if total_s > 0 else 0.0
    return avg_ms, fps, total_s
