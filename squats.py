import cv2
from squat_modules.thresholds import get_thresholds_beginner
from squat_modules.process_frame import ProcessFrame
from squat_modules.utils import get_mediapipe_pose

# Set up thresholds and pose detector
thresholds = get_thresholds_beginner()
pose = get_mediapipe_pose()

# Create frame processor
frame_processor = ProcessFrame(thresholds=thresholds, flip_frame=True)

def squat_detection(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and convert frame to RGB
        frame = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        # processed_frame, _ = frame_processor.process(frame_rgb, pose)
        processed_frame, play_sound = frame_processor.process(frame_rgb, pose)

        # Convert back to BGR for display
        display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()

        # Stream the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
