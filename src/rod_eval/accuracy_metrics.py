def calculate_accuracy_metrics(model, data_yaml, split="test", imgsz=640, batch=8,
                               device=None, conf=0.25, iou=0.7, verbose=False, plots=False):
    """Run Ultralytics `val()` and return (map50_95, map50, precision, recall).

    Args:
        model: a loaded Ultralytics ``YOLO`` model.
        data_yaml (str | Path): path to the dataset ``data.yaml``.
        split (str): which split to evaluate ('test' or 'val').
        imgsz (int): inference image size.
        batch (int): validation batch size.
        device (str | int | None): 'cpu', '0', 'cuda:0', or None for auto.
        conf (float): confidence threshold.
        iou (float): IoU threshold for NMS.
        verbose (bool): verbose Ultralytics logging.
        plots (bool): write PR/F1/confusion-matrix plots.

    Returns:
        tuple[float, float, float, float]: (mAP50-95, mAP50, precision, recall).
    """
    metrics = model.val(
        data=str(data_yaml), split=split, imgsz=imgsz, batch=batch,
        device=device, conf=conf, iou=iou, verbose=verbose, plots=plots,
    )
    try:
        return (
            float(metrics.box.map),
            float(metrics.box.map50),
            float(getattr(metrics.box, "mp", float("nan"))),
            float(getattr(metrics.box, "mr", float("nan"))),
        )
    except Exception:
        rd = getattr(metrics, "results_dict", {})
        return (
            float(rd.get("metrics/mAP50-95(B)", float("nan"))),
            float(rd.get("metrics/mAP50(B)", float("nan"))),
            float(rd.get("metrics/precision(B)", float("nan"))),
            float(rd.get("metrics/recall(B)", float("nan"))),
        )
