from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import time


MODELS = {
    "YOLOv8n":  r"helmet_models/yolov8n_best.pt",
    "YOLOv8s":  r"helmet_models/yolov8s_best.pt",
    "YOLOv11n": r"helmet_models/yolov11n_best.pt",
}

IMAGE_PATH = r"test_image.png"
CONFIDENCE = 0.25
OUTPUT_DIR = "comparison_results"


def test_single_model(model_path: str, image_path: str, conf: float = 0.25):
    model = YOLO(model_path)

    start = time.time()
    results = model.predict(
        source=image_path,
        conf=conf,
        imgsz=640,
        verbose=False
    )[0]
    elapsed = time.time() - start

    counts = {}
    if results.boxes is not None and len(results.boxes) > 0:
        names = results.names
        for cls_id in results.boxes.cls.cpu().numpy().astype(int):
            name = names[int(cls_id)]
            counts[name] = counts.get(name, 0) + 1

    annotated = results.plot(line_width=2, conf=True)
    return annotated, counts, elapsed, results


def main():
    img_path = Path(IMAGE_PATH)
    if not img_path.exists():
        print(f"[Error] Image not found: {IMAGE_PATH}")
        return

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Helmet Detection Model Comparison")
    print("=" * 60)
    print(f"Test image: {IMAGE_PATH}\n")

    all_results = {}

    for model_name, model_path in MODELS.items():
        path = Path(model_path)
        if not path.exists():
            print(f"[Warning] Model not found: {model_path}")
            continue

        print(f"Testing model: {model_name} ...")
        annotated, counts, elapsed, _ = test_single_model(
            model_path, str(img_path), conf=CONFIDENCE
        )

        all_results[model_name] = {
            "counts": counts,
            "time": elapsed,
            "annotated": annotated
        }

        save_path = out_dir / f"result_{model_name}.jpg"
        cv2.imwrite(str(save_path), annotated)
        print(f"  -> Saved: {save_path}")
        print(f"  -> Time: {elapsed:.3f} s")
        print(f"  -> Detections: {counts if counts else 'None'}")
        print()

    if not all_results:
        print("No models were tested. Check model paths.")
        return

    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))

    if n == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, all_results.items()):
        img_rgb = cv2.cvtColor(data["annotated"], cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.axis("off")

        title = f"{name}\n"
        title += f"Time: {data['time']:.2f}s\n"
        if data["counts"]:
            title += " | ".join([f"{k}:{v}" for k, v in data["counts"].items()])
        else:
            title += "No detection"
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    comparison_path = out_dir / "comparison_all.jpg"
    plt.savefig(str(comparison_path), dpi=150, bbox_inches="tight")
    plt.show()

    print("=" * 60)
    print("Summary:")
    print("-" * 60)
    print(f"{'Model':<12} {'Time (s)':<10} {'Detections'}")
    print("-" * 60)
    for name, data in all_results.items():
        det_str = str(data["counts"]) if data["counts"] else "None"
        print(f"{name:<12} {data['time']:<10.3f} {det_str}")
    print("=" * 60)
    print(f"\nResults saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
