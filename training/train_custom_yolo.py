"""Train the site-specific wildlife detector (Ultralytics YOLO).

    # 1) Small dataset (first ~300-1500 labelled frames): fine-tune, tiles, native pixels
    python training/train_custom_yolo.py --data dataset/data.yaml --preset small

    # 2) Bigger dataset / GPU: more capacity
    python training/train_custom_yolo.py --data dataset/data.yaml --preset large --device 0

Presets
-------
small : yolo11s, imgsz 640 (use with build_dataset.py --tile), first 10 layers frozen
        for the first epochs' stability, strong scale/colour augmentation.
large : yolo11m, imgsz 960, nothing frozen.
cpu   : yolo11n, imgsz 640, backbone frozen, max 14 h - for a PC without GPU.

Result: runs/wildlife/<name>/weights/best.pt  -> copy to models/melia_wildlife.pt and set
  WILDLIFE_CUSTOM_YOLO_ENABLED=true  WILDLIFE_CUSTOM_YOLO_MODEL_PATH=models/melia_wildlife.pt
Then run it on full frames at native size (CUSTOM_YOLO_IMGSZ=1920 in config).

Do NOT trust the final mAP alone - run training/eval_by_size.py on the validation
frames: it shows recall for tiny / small / medium animals separately, which is
where birds fail.
"""

import argparse

PRESETS = {
    # CPU-only PC (i5-7200U): small model, frozen backbone, capped wall-clock time.
    "cpu": dict(model="yolo11n.pt", imgsz=640, batch=8, epochs=30, freeze=10, patience=10, workers=2,
                time=14.0, scale=0.6, mosaic=1.0, mixup=0.0, close_mosaic=5, hsv_h=0.02, hsv_s=0.6,
                hsv_v=0.45, degrees=0.0, fliplr=0.5, flipud=0.0, translate=0.12),
    # Fine-tuning the site model we already have (round 4, 25/10).  Round 3 ran with the
    # Ultralytics "auto" choice AdamW lr 0.001667 + 3 warm-up epochs of 7 (made for training
    # from COCO weights): mAP50 was best after epoch 1 (0.759) and fell to 0.698-0.736.
    # Gentle: lr 0.0003, short warm-up, weights saved every epoch so the best epoch by
    # mAP50 (the acceptance measure) can be chosen, not the last one.
    "cpu_finetune": dict(model="yolo11n.pt", imgsz=640, batch=8, epochs=12, freeze=10, patience=10, workers=2,
                         time=14.0, scale=0.6, mosaic=1.0, mixup=0.0, close_mosaic=2, hsv_h=0.02, hsv_s=0.6,
                         hsv_v=0.45, degrees=0.0, fliplr=0.5, flipud=0.0, translate=0.12,
                         optimizer="AdamW", lr0=0.0003, lrf=0.1, momentum=0.9, warmup_epochs=0.5,
                         warmup_bias_lr=0.0, save_period=1),
    "small": dict(model="yolo11s.pt", imgsz=640, batch=16, epochs=120, freeze=10, patience=30,
                  scale=0.6, mosaic=1.0, mixup=0.1, close_mosaic=15, hsv_h=0.02, hsv_s=0.6, hsv_v=0.45,
                  degrees=0.0, fliplr=0.5, flipud=0.0, translate=0.12),
    "large": dict(model="yolo11m.pt", imgsz=960, batch=8, epochs=150, freeze=None, patience=40,
                  scale=0.5, mosaic=1.0, mixup=0.1, close_mosaic=20, hsv_h=0.02, hsv_s=0.6, hsv_v=0.45,
                  degrees=0.0, fliplr=0.5, flipud=0.0, translate=0.1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset/data.yaml from build_dataset.py")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="small")
    ap.add_argument("--time", type=float, default=None, help="max training hours")
    ap.add_argument("--model", default=None, help="override the preset's starting weights")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default="0", help="'0' for the first GPU, 'cpu' for CPU (slow)")
    ap.add_argument("--project", default="runs/wildlife")
    ap.add_argument("--name", default="melia")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("pip install -r requirements-custom-yolo.txt first") from exc

    cfg = dict(PRESETS[a.preset])
    if a.model:
        cfg["model"] = a.model
    for key in ("epochs", "imgsz", "batch", "time"):
        if getattr(a, key) is not None:
            cfg[key] = getattr(a, key)
    model_name = cfg.pop("model")
    if cfg.get("freeze") is None:
        cfg.pop("freeze", None)
    model = YOLO(model_name)
    model.train(data=a.data, device=a.device, project=a.project, name=a.name, pretrained=True,
                cos_lr=True, resume=a.resume, plots=True, **cfg)
    metrics = model.val(data=a.data, imgsz=cfg["imgsz"] if a.preset != "small" else 1920, device=a.device)
    print("\nValidation on FULL frames:", getattr(metrics, "results_dict", metrics))
    print(f"Weights: {a.project}/{a.name}/weights/best.pt")


if __name__ == "__main__":
    main()
