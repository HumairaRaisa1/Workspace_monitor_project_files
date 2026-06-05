import mediapipe as mp
import numpy as np
import cv2
import time

mp_face_mesh = mp.solutions.face_mesh

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    # Vertical distances
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # Horizontal distance
    h1 = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (v1 + v2) / (2.0 * h1 + 1e-6)

# MediaPipe FaceMesh eye landmark indices
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

class SleepDetector:
    EAR_THRESHOLD = 0.22
    CONSEC_FRAMES = 20   # ~0.7s at 30fps

    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.counters = {}  # track_id -> consecutive closed frames

    def analyze(self, frame, track_id=0):
        """Returns: 'sleeping', 'drowsy', 'awake', or None."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
        right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        if track_id not in self.counters:
            self.counters[track_id] = 0

        if avg_ear < self.EAR_THRESHOLD:
            self.counters[track_id] += 1
        else:
            self.counters[track_id] = 0

        if self.counters[track_id] >= self.CONSEC_FRAMES * 3:
            return 'sleeping'
        elif self.counters[track_id] >= self.CONSEC_FRAMES:
            return 'drowsy'
        return 'awake'