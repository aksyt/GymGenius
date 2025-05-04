# bicep_curl.py

import cv2
import mediapipe as mp
import numpy as np
import time

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def bicep_curl_detection(cap):
    # Reset variables for each session
    counter = 0
    correct_reps = 0  # Track correct reps separately
    stage = None
    form_feedback = "Good form"
    feedback_color = (0, 255, 0)

    rep_start_time = None
    rep_start_angle = None
    rep_min_angle = 180
    rep_max_angle = 0

    lock_start_time = None
    is_locked = False
    lock_duration = 0

    countdown_start = time.time()
    countdown_duration = 3
    countdown_active = True

    MIN_CURL_ANGLE = 30
    MAX_EXTENSION_ANGLE = 170
    HALF_REP_THRESHOLD = 80
    LOCK_ANGLE_THRESHOLD = 175
    LOCK_TIME_THRESHOLD = 0.5
    MIN_REP_DURATION = 0.7

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Get frame dimensions
            h, w, _ = image.shape

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
                else:
                    h, w, _ = image.shape
                    go_text = "GO!"
                    font_scale = 7
                    text_size = cv2.getTextSize(go_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 25)[0]
                    text_x = int((w - text_size[0]) / 2)
                    text_y = int((h + text_size[1]) / 2)
                    cv2.putText(image, go_text, (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 25, cv2.LINE_AA)
                    countdown_active = False
                    continue

            try:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = calculate_angle(shoulder, elbow, wrist)

                elbow_coords = tuple(np.multiply(elbow, [w, h]).astype(int))
                cv2.putText(image, str(round(angle, 2)), elbow_coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if angle > LOCK_ANGLE_THRESHOLD and not is_locked:
                    if lock_start_time is None:
                        lock_start_time = time.time()
                    else:
                        lock_duration = time.time() - lock_start_time
                        if lock_duration > LOCK_TIME_THRESHOLD:
                            is_locked = True
                            form_feedback = "Avoid locking joints"
                            feedback_color = (0, 0, 255)
                elif angle < LOCK_ANGLE_THRESHOLD:
                    lock_start_time = None
                    lock_duration = 0
                    if is_locked:
                        is_locked = False
                        if form_feedback == "Avoid locking joints":
                            form_feedback = "Good form"
                            feedback_color = (0, 255, 0)

                if stage == "down":
                    rep_min_angle = min(rep_min_angle, angle)
                    rep_max_angle = max(rep_max_angle, angle)

                if angle > 160 and stage != "down":
                    stage = "down"
                    rep_start_time = time.time()
                    rep_start_angle = angle
                    rep_min_angle = angle
                    rep_max_angle = angle
                    if is_locked:
                        form_feedback = "Avoid locking joints"
                        feedback_color = (0, 0, 255)

                if angle < MIN_CURL_ANGLE and stage == "down":
                    stage = "up"
                    rep_end_time = time.time()
                    rep_duration = rep_end_time - rep_start_time if rep_start_time else 0
                    counter += 1
                    
                    # Track correct reps
                    is_correct_rep = True
                    
                    if rep_duration < MIN_REP_DURATION:
                        form_feedback = "Slow down"
                        feedback_color = (0, 165, 255)
                        is_correct_rep = False
                    elif is_locked or form_feedback == "Avoid locking joints":
                        is_correct_rep = False
                    else:
                        form_feedback = "Good form"
                        feedback_color = (0, 255, 0)
                        is_correct_rep = True
                    
                    if is_correct_rep:
                        correct_reps += 1
                        
                    print(f"Rep {counter} duration: {rep_duration:.2f} seconds")
                    rep_start_time = None
                    rep_start_angle = None
                    rep_min_angle = 180
                    rep_max_angle = 0

                if stage == "down" and angle > HALF_REP_THRESHOLD and rep_min_angle < angle and rep_min_angle > MIN_CURL_ANGLE:
                    if angle - rep_min_angle > 20:
                        form_feedback = "Half rep detected"
                        feedback_color = (0, 0, 255)
                        stage = None
                        rep_start_time = None
                        rep_min_angle = 180
                        rep_max_angle = 0

            except:
                pass

            # ====== IMPROVED UI LAYOUT ======
            
            # Calculate widths for UI elements with better spacing
            counter_width = 150
            feedback_width = w - counter_width - 300
            correct_width = 150
            
            # Total rep counter (left side)
            cv2.rectangle(image, (10, 10), (10 + counter_width, 60), (245, 117, 16), -1)
            cv2.putText(image, 'TOTAL REPS', (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Correct form counter (right side)
            cv2.rectangle(image, (w - 10 - correct_width, 10), (w - 10, 60), (0, 255, 0), -1)
            cv2.putText(image, 'CORRECT', (w - correct_width + 5, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(correct_reps), (w - correct_width + 50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Form feedback (center)
            feedback_x = 10 + counter_width + 10
            feedback_width = w - feedback_x - correct_width - 20
            
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

            if is_locked:
                cv2.circle(image, (w - 30, 100), 15, (0, 0, 255), -1)
                cv2.putText(image, "LOCKED", (w - 100, 105), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # bicep_curl.py

# import cv2
# import mediapipe as mp
# import numpy as np
# import time

# def calculate_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)

#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#     if angle > 180.0:
#         angle = 360 - angle
#     return angle

# def bicep_curl_detection(cap):
#     # Reset variables for each session
#     counter = 0
#     stage = None
#     form_feedback = "Good form"
#     feedback_color = (0, 255, 0)

#     rep_start_time = None
#     rep_start_angle = None
#     rep_min_angle = 180
#     rep_max_angle = 0

#     lock_start_time = None
#     is_locked = False
#     lock_duration = 0

#     countdown_start = time.time()
#     countdown_duration = 3
#     countdown_active = True

#     MIN_CURL_ANGLE = 30
#     MAX_EXTENSION_ANGLE = 170
#     HALF_REP_THRESHOLD = 80
#     LOCK_ANGLE_THRESHOLD = 175
#     LOCK_TIME_THRESHOLD = 0.5
#     MIN_REP_DURATION = 0.7

#     mp_drawing = mp.solutions.drawing_utils
#     mp_pose = mp.solutions.pose

#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False
#             results = pose.process(image)
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             if countdown_active:
#                 elapsed = time.time() - countdown_start
#                 if elapsed < countdown_duration:
#                     h, w, _ = image.shape
#                     remaining = countdown_duration - elapsed
#                     countdown_text = str(int(remaining) + 1)
#                     font_scale = 7
#                     text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 25)[0]
#                     text_x = int((w - text_size[0]) / 2)
#                     text_y = int((h + text_size[1]) / 2)
#                     cv2.putText(image, countdown_text, (text_x, text_y), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 25, cv2.LINE_AA)
#                 else:
#                     h, w, _ = image.shape
#                     go_text = "GO!"
#                     font_scale = 7
#                     text_size = cv2.getTextSize(go_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 25)[0]
#                     text_x = int((w - text_size[0]) / 2)
#                     text_y = int((h + text_size[1]) / 2)
#                     cv2.putText(image, go_text, (text_x, text_y), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 25, cv2.LINE_AA)
#                     countdown_active = False
#                     continue

#             try:
#                 landmarks = results.pose_landmarks.landmark
#                 shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#                 elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#                 wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
#                 angle = calculate_angle(shoulder, elbow, wrist)

#                 elbow_coords = tuple(np.multiply(elbow, [640, 480]).astype(int))
#                 cv2.putText(image, str(round(angle, 2)), elbow_coords, 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

#                 if angle > LOCK_ANGLE_THRESHOLD and not is_locked:
#                     if lock_start_time is None:
#                         lock_start_time = time.time()
#                     else:
#                         lock_duration = time.time() - lock_start_time
#                         if lock_duration > LOCK_TIME_THRESHOLD:
#                             is_locked = True
#                             form_feedback = "Avoid locking joints"
#                             feedback_color = (0, 0, 255)
#                 elif angle < LOCK_ANGLE_THRESHOLD:
#                     lock_start_time = None
#                     lock_duration = 0
#                     if is_locked:
#                         is_locked = False
#                         if form_feedback == "Avoid locking joints":
#                             form_feedback = "Good form"
#                             feedback_color = (0, 255, 0)

#                 if stage == "down":
#                     rep_min_angle = min(rep_min_angle, angle)
#                     rep_max_angle = max(rep_max_angle, angle)

#                 if angle > 160 and stage != "down":
#                     stage = "down"
#                     rep_start_time = time.time()
#                     rep_start_angle = angle
#                     rep_min_angle = angle
#                     rep_max_angle = angle
#                     if is_locked:
#                         form_feedback = "Avoid locking joints"
#                         feedback_color = (0, 0, 255)

#                 if angle < MIN_CURL_ANGLE and stage == "down":
#                     stage = "up"
#                     rep_end_time = time.time()
#                     rep_duration = rep_end_time - rep_start_time if rep_start_time else 0
#                     counter += 1
#                     if rep_duration < MIN_REP_DURATION:
#                         form_feedback = "Slow down"
#                         feedback_color = (0, 165, 255)
#                     elif is_locked or form_feedback == "Avoid locking joints":
#                         pass
#                     else:
#                         form_feedback = "Good form"
#                         feedback_color = (0, 255, 0)
#                     print(f"Rep {counter} duration: {rep_duration:.2f} seconds")
#                     rep_start_time = None
#                     rep_start_angle = None
#                     rep_min_angle = 180
#                     rep_max_angle = 0

#                 if stage == "down" and angle > HALF_REP_THRESHOLD and rep_min_angle < angle and rep_min_angle > MIN_CURL_ANGLE:
#                     if angle - rep_min_angle > 20:
#                         form_feedback = "Half rep detected"
#                         feedback_color = (0, 0, 255)
#                         stage = None
#                         rep_start_time = None
#                         rep_min_angle = 180
#                         rep_max_angle = 0

#             except:
#                 pass

#             # Draw overlays
#             cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
#             cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#             cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

#             h, w, _ = image.shape
#             cv2.rectangle(image, (w - 225, 0), (w, 73), feedback_color, -1)
#             cv2.putText(image, 'FORM', (w - 210, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#             cv2.putText(image, form_feedback, (w - 210, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                       mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#                                       mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

#             if is_locked:
#                 cv2.circle(image, (w - 30, 100), 15, (0, 0, 255), -1)
#                 cv2.putText(image, "LOCKED", (w - 100, 105), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

#             ret, buffer = cv2.imencode('.jpg', image)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
