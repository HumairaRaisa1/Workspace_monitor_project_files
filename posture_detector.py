import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def get_angle(a, b, c):
    """Angle at joint b between points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle

class PostureDetector:
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def analyze(self, frame):
        """Returns: 'upright', 'slouching', or None if not detected."""
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return None, None

        lm = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        def pt(idx):
            return [lm[idx].x * w, lm[idx].y * h]

        # Shoulder midpoint and hip midpoint
        l_shoulder = pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        r_shoulder = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        l_hip = pt(mp_pose.PoseLandmark.LEFT_HIP.value)
        r_hip = pt(mp_pose.PoseLandmark.RIGHT_HIP.value)
        nose = pt(mp_pose.PoseLandmark.NOSE.value)

        shoulder_mid = [(l_shoulder[0]+r_shoulder[0])/2, (l_shoulder[1]+r_shoulder[1])/2]
        hip_mid = [(l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2]

        # Spine tilt angle (vertical = 90 degrees)
        spine_angle = get_angle(
            [hip_mid[0], hip_mid[1] - 100],  # virtual vertical
            hip_mid,
            shoulder_mid
        )

        # Head tilt: nose should be roughly above shoulder midpoint
        head_forward = abs(nose[0] - shoulder_mid[0]) > 60  # pixels

        if spine_angle < 130 or head_forward: # was 160
            return 'slouching', results.pose_landmarks
        return 'upright', results.pose_landmarks