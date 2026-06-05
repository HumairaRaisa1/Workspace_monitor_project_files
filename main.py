import cv2
import traceback

try:
    from ultralytics import YOLO
    from posture_detector import PostureDetector
    from gaze_detector import GazeDetector
    from sleep_detector import SleepDetector
    from phone_detector import PhoneDetector
    from productivity_scorer import ProductivityScorer

    print("All imports OK")

    tracker   = YOLO('yolov8n.pt')
    print("YOLO loaded")

    posture_d = PostureDetector()
    print("PostureDetector OK")

    gaze_d    = GazeDetector()
    print("GazeDetector OK")

    sleep_d   = SleepDetector()
    print("SleepDetector OK")

    phone_d   = PhoneDetector()
    print("PhoneDetector OK")

    scorer    = ProductivityScorer()
    print("Scorer OK")

    CAMERA_INDEX = 0
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera opened, starting loop... Press Q to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        phone_found, phone_box = phone_d.detect(frame)

        results = tracker.track(frame, persist=True, classes=[0], verbose=False)[0]

        for box in results.boxes:
            if box.id is None:
                continue
            pid = int(box.id)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            posture, _  = posture_d.analyze(person_crop)
            gaze_poses  = gaze_d.get_head_pose(person_crop)
            gaze_away   = any(gaze_d.is_looking_away(y, p) for y, p in gaze_poses) if gaze_poses else False
            sleep_state = sleep_d.analyze(person_crop, pid)

            score, states = scorer.update(pid, posture, gaze_away, phone_found, sleep_state)
            label, color  = scorer.get_label(pid)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"P{pid}: {label} ({int(score)})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            flags = []
            if states['phone']:     flags.append('PHONE')
            if states['sleeping']:  flags.append('SLEEPING')
            if states['drowsy']:    flags.append('DROWSY')
            if states['slouching']: flags.append('SLOUCH')
            if states['away_gaze']: flags.append('NOT LOOKING')

            for i, f in enumerate(flags):
                cv2.putText(frame, f, (x1, y1 + 20 + i*18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)

        if phone_found and phone_box:
            px1, py1, px2, py2 = map(int, phone_box)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 165, 255), 2)
            cv2.putText(frame, 'PHONE', (px1, py1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        cv2.imshow('Workspace Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print("\n--- ERROR ---")
    traceback.print_exc()
    input("\nPress Enter to close...")