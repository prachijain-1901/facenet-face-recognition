"""
FaceNet webcam demo for saved facenet_model.pt.
No training is performed. Keep this file beside facenet_model.pt.
"""
import argparse
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face, fixed_image_standardization

def nearest_centroid(embedding, centroids):
    distances = torch.cdist(embedding, centroids)
    min_distance, index = distances.min(dim=1)
    return index, min_distance

def main():
    parser = argparse.ArgumentParser(description="FaceNet webcam demo")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    model_path = project_dir / "facenet_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Loading checkpoint:", model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Could not find {model_path}. Keep demo.py beside facenet_model.pt."
        )

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    required_keys = {"state", "classes", "centroids", "threshold"}
    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise KeyError(
            f"Checkpoint is missing keys {sorted(missing_keys)}. "
            "Confirm this is the trained facenet_model.pt from your notebook."
        )

    classes = checkpoint["classes"]
    centroids = torch.as_tensor(checkpoint["centroids"], dtype=torch.float32, device=device)
    threshold = args.threshold if args.threshold is not None else float(checkpoint["threshold"])

    # Recreate the trained FaceNet architecture and load its saved weights.
    backbone = InceptionResnetV1(pretrained=None).to(device)
    backbone.load_state_dict(checkpoint["state"], strict=False)
    backbone.eval()

    mtcnn = MTCNN(device=device, keep_all=False)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam {args.camera}. Close apps using it or try --camera 1."
        )

    print("Loaded member classes:", classes)
    print(f"Recognition threshold: {threshold:.3f}")
    print("Webcam running. Press Q in the webcam window to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a webcam frame.")
                break

            rgb_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, probabilities = mtcnn.detect(rgb_image)

            if boxes is not None and probabilities is not None:
                for box, probability in zip(boxes, probabilities):
                    if probability is None or probability < 0.90:
                        continue
                    face = extract_face(rgb_image, box, image_size=160, margin=20).float()
                    face = fixed_image_standardization(face).unsqueeze(0).to(device)

                    with torch.no_grad():
                        embedding = F.normalize(backbone(face), dim=1)

                    index, distance = nearest_centroid(embedding, centroids)
                    distance_value = float(distance.item())
                    name = str(classes[int(index.item())]) if distance_value <= threshold else "unidentified"

                    x1, y1, x2, y2 = [int(value) for value in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    color = (0, 200, 0) if name != "unidentified" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, f"{name} (d={distance_value:.2f})",
                        (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2, cv2.LINE_AA
                    )

            cv2.imshow("FaceNet Webcam Recognition", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
