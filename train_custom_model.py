"""
Train a football-specific YOLO model that has real "player" / "referee" /
"goalkeeper" / "ball" classes — instead of the generic pretrained model's
"person" class, which matches literally anyone in frame.

This uses the public "football-players-detection" dataset on Roboflow
Universe (the same one used by the original tutorial this project follows).

SETUP (one-time):
    1. Create a free account at https://roboflow.com
    2. Go to your account settings and copy your Private API Key
    3. Run: python train_custom_model.py YOUR_API_KEY

⚠️ IMPORTANT — TRAINING SPEED:
    Training YOLO on CPU (a normal laptop) is extremely slow — this could
    take many hours to days. Strongly recommended: run this script on
    Google Colab instead, which gives you a free GPU:
        1. Go to https://colab.research.google.com, new notebook
        2. Runtime > Change runtime type > select a GPU (T4 is fine, free)
        3. Upload this script or paste its contents into a cell
        4. Run it there — should take well under an hour instead of days

Output:
    runs/detect/train/weights/best.pt  <- your custom-trained model

Once training finishes, use it like this:
    python main.py "input_videos/match1.mp4" runs/detect/train/weights/best.pt 0.15 1280
"""
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_custom_model.py YOUR_ROBOFLOW_API_KEY")
        print("Get a free API key at https://roboflow.com (account settings)")
        sys.exit(1)

    api_key = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Missing dependency. Run: pip install roboflow")
        sys.exit(1)

    print("Downloading dataset (football-players-detection)...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    version = project.version(1)
    dataset = version.download("yolov8")

    print(f"\nDataset downloaded to: {dataset.location}")
    print(f"Starting training for {epochs} epochs — this is the slow part.\n")

    from ultralytics import YOLO
    model = YOLO("yolov8s.pt")  # start from pretrained weights, fine-tune on football data
    model.train(
        data=os.path.join(dataset.location, "data.yaml"),
        epochs=epochs,
        imgsz=1280,
        patience=15,   # stop early if it stops improving
    )

    print("\nDone! Your custom model is at: runs/detect/train/weights/best.pt")
    print('Use it with: python main.py "input_videos/match1.mp4" '
          'runs/detect/train/weights/best.pt 0.15 1280')


if __name__ == "__main__":
    main()