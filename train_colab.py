from ultralytics import YOLO

DATA_YAML = "data.yaml"

models_to_train = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov11n": "yolo11n.pt",
}

EPOCHS = 60
IMG_SIZE = 640
BATCH = 16

results_summary = {}

for name, weights in models_to_train.items():
    print("=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    model = YOLO(weights)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project="helmet_research",
        name=name,
        patience=20,
        save=True,
        plots=True,
        device=0,
        exist_ok=True,
    )

    metrics = results.results_dict
    results_summary[name] = {
        "mAP50": round(metrics.get("metrics/mAP50(B)", 0), 4),
        "mAP50-95": round(metrics.get("metrics/mAP50-95(B)", 0), 4),
        "precision": round(metrics.get("metrics/precision(B)", 0), 4),
        "recall": round(metrics.get("metrics/recall(B)", 0), 4),
    }

    print(f"Result {name}: {results_summary[name]}")

print("=" * 65)
print(f"{'Model':<12} {'mAP50':<10} {'mAP50-95':<12} {'Precision':<12} {'Recall':<10}")
print("-" * 65)
for name, m in results_summary.items():
    print(f"{name:<12} {m['mAP50']:<10} {m['mAP50-95']:<12} {m['precision']:<12} {m['recall']:<10}")
