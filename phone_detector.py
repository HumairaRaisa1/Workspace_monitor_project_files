from ultralytics import YOLO

class PhoneDetector:
    CELL_PHONE_CLASS = 67  # COCO class id for cell phone

    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)  # pretrained on COCO — no extra training needed

    def detect(self, frame):
        """Returns True if a phone is detected anywhere in frame."""
        results = self.model(frame, verbose=False)[0]
        for box in results.boxes:
            if int(box.cls) == self.CELL_PHONE_CLASS and float(box.conf) > 0.4:
                return True, box.xyxy[0].tolist()
        return False, None