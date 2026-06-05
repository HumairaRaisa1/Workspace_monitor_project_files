import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ─── CONFIG ───────────────────────────────────────────
MODEL_PATH = "gaze_model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── LOAD MODEL ───────────────────────────────────────
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded ✅")

# ─── TRANSFORM ────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── GAZE DIRECTION LABEL ─────────────────────────────
def get_gaze_label(x, y):
    """Convert gaze vector to human-readable direction"""
    if y < -0.1:
        vertical = "Top"
    elif y > 0.1:
        vertical = "Bottom"
    else:
        vertical = "Middle"

    if x < -0.1:
        horizontal = "Left"
    elif x > 0.1:
        horizontal = "Right"
    else:
        horizontal = "Center"

    return f"{vertical}{horizontal}"

# ─── DRAW GAZE ARROW ──────────────────────────────────
def draw_gaze(frame, gaze_vec, face_center):
    x, y, z = gaze_vec
    cx, cy = face_center
    # Scale arrow length
    scale = 150
    end_x = int(cx + x * scale)
    end_y = int(cy - y * scale)  # flip Y for screen coords
    cv2.arrowedLine(frame, (cx, cy), (end_x, end_y),
                    (0, 255, 0), 2, tipLength=0.3)
    return frame

# ─── LIVE DETECTION ───────────────────────────────────
def run_detector():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print("Starting live detection... Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Crop face region
            face_img = frame[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

            # Predict gaze
            input_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                gaze_vec = model(input_tensor).cpu().numpy()[0]

            # Get label
            label = get_gaze_label(gaze_vec[0], gaze_vec[1])

            # Draw face box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Draw gaze arrow
            face_center = (x + w//2, y + h//2)
            frame = draw_gaze(frame, gaze_vec, face_center)

            # Display label
            cv2.putText(frame, f"Gaze: {label}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)

            # Display gaze vector values
            cv2.putText(frame,
                        f"Vec: ({gaze_vec[0]:.2f}, {gaze_vec[1]:.2f}, {gaze_vec[2]:.2f})",
                        (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1)

        cv2.imshow("Workspace Gaze Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detector stopped.")

# ─── RUN ──────────────────────────────────────────────
if __name__ == "__main__":
    run_detector()