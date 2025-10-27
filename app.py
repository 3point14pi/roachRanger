# =========================
# STEP 0: Installs
# =========================
!pip -q install ultralytics opencv-python

# =========================
# STEP 1: Imports
# =========================
import os, csv, zipfile, time
import cv2
import numpy as np
from google.colab import files
from ultralytics import YOLO

# =========================
# STEP 2: Upload video
# =========================
print("📤 Upload your video file")
uploaded = files.upload()
video_path = list(uploaded.keys())[0]
print(f"✅ Uploaded: {video_path}")

# =========================
# STEP 3: Settings
# =========================
# Change weights if you have your own .pt (e.g., "best.pt" or a roach model)
WEIGHTS = "yolov8n.pt"    # small & fast default
CONF    = 0.35            # confidence threshold
IMGSZ   = 640             # inference size
SAVE_FRAMES = False       # set True to also save annotated JPG frames
FRAME_STRIDE = 1          # process every Nth frame (e.g., 2 = half speed)

# Output paths
stem = os.path.splitext(os.path.basename(video_path))[0]
out_dir = f"{stem}_yolo_out"
os.makedirs(out_dir, exist_ok=True)
frames_dir = os.path.join(out_dir, "frames")
if SAVE_FRAMES:
    os.makedirs(frames_dir, exist_ok=True)
csv_path = os.path.join(out_dir, f"{stem}_detections.csv")
out_video_path = os.path.join(out_dir, f"{stem}_annotated.mp4")
zip_name = f"{stem}_yolo_outputs.zip"

# =========================
# STEP 4: Load model (GPU if available)
# =========================
model = YOLO(WEIGHTS)
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    device = "cpu"
print(f"🧠 Using device: {device}")

# =========================
# STEP 5: Open video
# =========================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("❌ Error: Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

# Prepare video writer (mp4v is broadly compatible in Colab)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

# CSV header
with open(csv_path, "w", newline="") as fcsv:
    wcsv = csv.writer(fcsv)
    wcsv.writerow(["frame_idx", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])

# =========================
# STEP 6: Process loop
# =========================
frame_idx = 0
kept = 0
t0 = time.time()

print(f"▶️ Processing {total if total>0 else '?'} frames @ {fps:.2f} FPS, {w}x{h}")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames for speed if requested
    if FRAME_STRIDE > 1 and (frame_idx % FRAME_STRIDE) != 0:
        frame_idx += 1
        continue

    # Predict
    results = model.predict(
        frame,
        conf=CONF,
        imgsz=IMGSZ,
        device=device,
        verbose=False
    )

    r = results[0]
    boxes = getattr(r, "boxes", None)

    # Draw and log
    if boxes is not None and len(boxes) > 0:
        # Access tensors safely
        xyxy = boxes.xyxy
        cls  = boxes.cls
        conf = boxes.conf

        # Normalize class names list/dict
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            try:
                max_k = max(names.keys())
                if all(k in names for k in range(max_k + 1)):
                    names = [names[k] for k in range(max_k + 1)]
            except Exception:
                pass

        # to CPU numpy
        xyxy_np = xyxy.detach().cpu().numpy()
        cls_np  = cls.detach().cpu().numpy() if cls is not None else np.array([])
        conf_np = conf.detach().cpu().numpy() if conf is not None else np.array([])

        # Draw
        for i in range(xyxy_np.shape[0]):
            x1, y1, x2, y2 = map(int, xyxy_np[i].tolist())
            c  = int(cls_np[i]) if i < len(cls_np) else -1
            p  = float(conf_np[i]) if i < len(conf_np) else 0.0
            label = None
            if isinstance(names, (list, tuple)) and 0 <= c < len(names):
                label = f"{names[c]} {p:.2f}"
            elif isinstance(names, dict) and c in names:
                label = f"{names[c]} {p:.2f}"
            else:
                label = f"id{c} {p:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 220, 50), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # write to CSV
            with open(csv_path, "a", newline="") as fcsv:
                wcsv = csv.writer(fcsv)
                wcsv.writerow([frame_idx, c, label.split()[0] if label else "", f"{p:.4f}", x1, y1, x2, y2])

    # Save annotated frame to video
    writer.write(frame)
    kept += 1

    # Optional: save individual frames
    if SAVE_FRAMES:
        cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:05d}.jpg"), frame)

    if frame_idx % 50 == 0:
        elapsed = time.time() - t0
        print(f"Processed {frame_idx} frames... ({elapsed:.1f}s)")
    frame_idx += 1

cap.release()
writer.release()

print(f"✅ Done! Wrote {kept} processed frames to video: {out_video_path}")

# =========================
# STEP 7: Zip & download
# =========================
with zipfile.ZipFile(zip_name, 'w') as zipf:
    # video
    zipf.write(out_video_path)
    # CSV
    zipf.write(csv_path)
    # frames (optional)
    if SAVE_FRAMES:
        for root, _, files_list in os.walk(frames_dir):
            for fn in files_list:
                zipf.write(os.path.join(root, fn))

print(f"📦 Zipped outputs into {zip_name}")
files.download(zip_name)
