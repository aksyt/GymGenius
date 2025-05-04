import cv2
import mediapipe as mp
import numpy as np
import time

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (in degrees)
    
    Args:
        a: First point [x, y]
        b: Mid point [x, y]
        c: End point [x, y]
    
    Returns:
        Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_visibility(landmarks, keypoints):
    """
    Calculate the average visibility of specified keypoints
    
    Args:
        landmarks: MediaPipe pose landmarks
        keypoints: List of landmark indices to check
    
    Returns:
        Average visibility score (0-1)
    """
    visibility_sum = sum(landmarks[i].visibility for i in keypoints)
    return visibility_sum / len(keypoints)

def push_up_detection(cap):
    """
    Generator function for push-up detection that yields frame data for Flask streaming
    """
    # Reset cap to ensure fresh start
    if cap.isOpened():
        cap.release()
    cap = cv2.VideoCapture(0)
    
    # Set width and height
    cap.set(3, 640)
    cap.set(4, 480)
    
    # MediaPipe Pose setup
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Push-up variables
    counter = 0
    stage = None
    form_feedback = "Position yourself sideways"
    feedback_color = (245, 117, 16)  # Orange
    
    acceptable_form_count = 0
    
    # Angle thresholds
    GOOD_ELBOW_ANGLE = 90  # Deeper is better
    MIN_ELBOW_ANGLE = 70   # Minimum to count a rep
    MIN_BODY_ANGLE = 160   # For straight body alignment
    
    # Previous angles for movement detection
    prev_elbow_angle = None
    
    # Rep timing
    rep_start_time = None
    min_rep_angle = 180
    
    # Countdown setup
    countdown_start = time.time()
    countdown_duration = 5
    countdown_active = True
    
    # Movement detection to prevent false positives
    movement_buffer = []
    MOVEMENT_BUFFER_SIZE = 10
    MIN_ANGLE_CHANGE = 15  # Minimum angle change to register as a movement
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process the image with MediaPipe Pose
        results = pose.process(image)
        
        # Convert back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Handle countdown
        if countdown_active:
            elapsed = time.time() - countdown_start
            if elapsed < countdown_duration:
                remaining = countdown_duration - elapsed
                countdown_text = str(int(remaining) + 1)
                font_scale = 7
                text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 25)[0]
                text_x = int((w - text_size[0]) / 2)
                text_y = int((h + text_size[1]) / 2)
                cv2.putText(image, countdown_text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 25, cv2.LINE_AA)
                
                # Draw pose landmarks during countdown too
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )
                
                # Yield frame for Flask streaming
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue
            else:
                go_text = "GO!"
                font_scale = 7
                text_size = cv2.getTextSize(go_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 25)[0]
                text_x = int((w - text_size[0]) / 2)
                text_y = int((h + text_size[1]) / 2)
                cv2.putText(image, go_text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 25, cv2.LINE_AA)
                
                # Yield frame for Flask streaming
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    
                countdown_active = False
                # Reset start time for rep timing
                rep_start_time = time.time()
                continue
        
        # Extract landmarks
        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Check visibility of left vs right side
                left_side_keypoints = [
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_ELBOW.value,
                    mp_pose.PoseLandmark.LEFT_WRIST.value,
                    mp_pose.PoseLandmark.LEFT_HIP.value,
                    mp_pose.PoseLandmark.LEFT_KNEE.value,
                    mp_pose.PoseLandmark.LEFT_ANKLE.value
                ]
                
                right_side_keypoints = [
                    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                    mp_pose.PoseLandmark.RIGHT_WRIST.value,
                    mp_pose.PoseLandmark.RIGHT_HIP.value,
                    mp_pose.PoseLandmark.RIGHT_KNEE.value,
                    mp_pose.PoseLandmark.RIGHT_ANKLE.value
                ]
                
                left_visibility = calculate_visibility(landmarks, left_side_keypoints)
                right_visibility = calculate_visibility(landmarks, right_side_keypoints)
                
                # Determine which side to use based on visibility
                use_left = left_visibility > right_visibility
                
                # Get coordinates based on the more visible side
                if use_left:
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                           landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                else:
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                           landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                # Calculate angles
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                body_angle = calculate_angle(shoulder, hip, knee)
                leg_angle = calculate_angle(hip, knee, ankle)
                
                # Display angle at elbow
                elbow_coords = tuple(np.multiply(elbow, [w, h]).astype(int))
                cv2.putText(image, str(round(elbow_angle, 2)), elbow_coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Body angle at hip
                hip_coords = tuple(np.multiply(hip, [w, h]).astype(int))
                cv2.putText(image, str(round(body_angle, 2)), hip_coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Track movement for false positive prevention
                if prev_elbow_angle is not None:
                    angle_change = abs(elbow_angle - prev_elbow_angle)
                    movement_buffer.append(angle_change)
                    if len(movement_buffer) > MOVEMENT_BUFFER_SIZE:
                        movement_buffer.pop(0)
                
                # Significant movement detected
                significant_movement = sum(movement_buffer) > MIN_ANGLE_CHANGE if movement_buffer else False
                
                # Push-up logic
                if prev_elbow_angle is not None:
                    
                    # Body alignment check
                    body_aligned = body_angle > MIN_BODY_ANGLE
                    
                    if not body_aligned:
                        form_feedback = "Keep your body straight!"
                        feedback_color = (0, 0, 255)  # Red
                    else:
                        # Going down - arms bending
                        if elbow_angle < 120 and (stage == "up" or stage is None) and significant_movement:
                            stage = "down"
                            form_feedback = "Going down..."
                            feedback_color = (245, 117, 16)  # Orange
                            rep_start_time = time.time()
                            min_rep_angle = elbow_angle
                            
                        # In down position - track minimum angle
                        elif stage == "down":
                            min_rep_angle = min(min_rep_angle, elbow_angle)
                            
                            # Only provide feedback if still in down position
                            if elbow_angle < 120:
                                # Feedback on depth
                                if elbow_angle > 90:
                                    form_feedback = "Go lower for better form"
                                    feedback_color = (0, 165, 255)  # Orange
                                else:
                                    form_feedback = "Good depth!"
                                    feedback_color = (0, 255, 0)  # Green
                        
                        # Going up - completed rep
                        if elbow_angle > 160 and stage == "down" and significant_movement:
                            # Only count if there was a significant bend
                            if min_rep_angle < 120:
                                # Evaluate form based on depth
                                if min_rep_angle > MIN_ELBOW_ANGLE and min_rep_angle <= GOOD_ELBOW_ANGLE:
                                    # Acceptable form but not great
                                    counter += 1
                                    acceptable_form_count += 1
                                    form_feedback = "Push-up counted! Go deeper next time"
                                    feedback_color = (0, 165, 255)  # Orange
                                elif min_rep_angle <= MIN_ELBOW_ANGLE:
                                    # Not deep enough
                                    form_feedback = "Too shallow! Not counted"
                                    feedback_color = (0, 0, 255)  # Red
                                else:
                                    # Good form
                                    counter += 1
                                    form_feedback = "Great push-up!"
                                    feedback_color = (0, 255, 0)  # Green
                                    
                            stage = "up"
                            rep_start_time = None
                            # Reset movement buffer
                            movement_buffer = []
                
                # Update previous angle
                prev_elbow_angle = elbow_angle
                
        except Exception as e:
            print(f"Error processing landmarks: {e}")
            form_feedback = "Position yourself sideways to the camera"
            feedback_color = (0, 0, 255)  # Red
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        # ====== IMPROVED UI LAYOUT ======
        
        # Calculate widths for UI elements with better spacing
        counter_width = 150
        feedback_width = w - counter_width - 300
        acceptable_width = 150
        
        # Total push-up counter (left side)
        cv2.rectangle(image, (10, 10), (10 + counter_width, 60), (245, 117, 16), -1)
        cv2.putText(image, 'TOTAL REPS', (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Acceptable form counter (right side)
        cv2.rectangle(image, (w - 10 - acceptable_width, 10), (w - 10, 60), (0, 165, 255), -1)
        cv2.putText(image, 'ACCEPTABLE', (w - acceptable_width + 5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(acceptable_form_count), (w - acceptable_width + 50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Form feedback (center)
        feedback_x = 10 + counter_width + 10
        feedback_width = w - feedback_x - acceptable_width - 20
        
        cv2.rectangle(image, (feedback_x, 10), (feedback_x + feedback_width, 60), feedback_color, -1)
        
        # Ensure form feedback text fits in the box
        form_text = form_feedback
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Calculate text width to ensure it fits
        text_size = cv2.getTextSize(form_text, font, font_scale, thickness)[0]
        
        # If text is too long, reduce font size
        if text_size[0] > feedback_width - 20:
            font_scale = 0.5
            text_size = cv2.getTextSize(form_text, font, font_scale, thickness)[0]
        
        # Calculate text position to center it
        text_x = feedback_x + (feedback_width - text_size[0]) // 2
        text_y = 40
        
        cv2.putText(image, 'FORM', (feedback_x + 5, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, form_text, (text_x, text_y), 
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Convert to JPEG for Flask streaming
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')