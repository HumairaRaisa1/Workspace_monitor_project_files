import traceback

try:
    print("Testing imports...")
    from posture_detector import PostureDetector
    print("posture_detector OK")
    from gaze_detector import GazeDetector
    print("gaze_detector OK")
    from sleep_detector import SleepDetector
    print("sleep_detector OK")
    from phone_detector import PhoneDetector
    print("phone_detector OK")
    from productivity_scorer import ProductivityScorer
    print("productivity_scorer OK")
    print("\nAll imports successful!")
except Exception as e:
    traceback.print_exc()

input("\nPress Enter to close...")