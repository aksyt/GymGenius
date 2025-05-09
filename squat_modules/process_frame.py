# import time
# import cv2
# import numpy as np
# from .utils import find_angle, get_landmark_features, draw_text


# class ProcessFrame:
#     def __init__(self, thresholds, flip_frame = False):
        
#         # Set if frame should be flipped or not.
#         self.flip_frame = flip_frame

#         # self.thresholds
#         self.thresholds = thresholds

#         # Font type.
#         self.font = cv2.FONT_HERSHEY_SIMPLEX

#         # line type
#         self.linetype = cv2.LINE_AA

#         # set radius to draw arc
#         self.radius = 20

#         # Colors in BGR format.
#         self.COLORS = {
#                         'blue'       : (0, 127, 255),
#                         'red'        : (255, 50, 50),
#                         'green'      : (0, 255, 127),
#                         'light_green': (100, 233, 127),
#                         'yellow'     : (255, 255, 0),
#                         'magenta'    : (255, 0, 255),
#                         'white'      : (255,255,255),
#                         'cyan'       : (0, 255, 255),
#                         'light_blue' : (102, 204, 255)
#                       }



#         # Dictionary to maintain the various landmark features.
#         self.dict_features = {}
#         self.left_features = {
#                                 'shoulder': 11,
#                                 'elbow'   : 13,
#                                 'wrist'   : 15,                    
#                                 'hip'     : 23,
#                                 'knee'    : 25,
#                                 'ankle'   : 27,
#                                 'foot'    : 31
#                              }

#         self.right_features = {
#                                 'shoulder': 12,
#                                 'elbow'   : 14,
#                                 'wrist'   : 16,
#                                 'hip'     : 24,
#                                 'knee'    : 26,
#                                 'ankle'   : 28,
#                                 'foot'    : 32
#                               }

#         self.dict_features['left'] = self.left_features
#         self.dict_features['right'] = self.right_features
#         self.dict_features['nose'] = 0

        
#         # For tracking counters and sharing states in and out of callbacks.
#         self.state_tracker = {
#             'state_seq': [],

#             'start_inactive_time': time.perf_counter(),
#             'start_inactive_time_front': time.perf_counter(),
#             'INACTIVE_TIME': 0.0,
#             'INACTIVE_TIME_FRONT': 0.0,

#             # 0 --> Bend Backwards, 1 --> Bend Forward, 2 --> Keep shin straight, 3 --> Deep squat
#             'DISPLAY_TEXT' : np.full((4,), False),
#             'COUNT_FRAMES' : np.zeros((4,), dtype=np.int64),

#             'LOWER_HIPS': False,

#             'INCORRECT_POSTURE': False,

#             'prev_state': None,
#             'curr_state':None,

#             'SQUAT_COUNT': 0,
#             'IMPROPER_SQUAT':0
            
#         }
        
#         self.FEEDBACK_ID_MAP = {
#                                 0: ('BEND BACKWARDS', 215, (0, 153, 255)),
#                                 1: ('BEND FORWARD', 215, (0, 153, 255)),
#                                 2: ('KNEE FALLING OVER TOE', 170, (255, 80, 80)),
#                                 3: ('SQUAT TOO DEEP', 125, (255, 80, 80))
#                                }

        


#     def _get_state(self, knee_angle):
        
#         knee = None        

#         if self.thresholds['HIP_KNEE_VERT']['NORMAL'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['NORMAL'][1]:
#             knee = 1
#         elif self.thresholds['HIP_KNEE_VERT']['TRANS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['TRANS'][1]:
#             knee = 2
#         elif self.thresholds['HIP_KNEE_VERT']['PASS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['PASS'][1]:
#             knee = 3

#         return f's{knee}' if knee else None



    
#     def _update_state_sequence(self, state):

#         if state == 's2':
#             if (('s3' not in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2'))==0) or \
#                     (('s3' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2')==1)):
#                         self.state_tracker['state_seq'].append(state)
            

#         elif state == 's3':
#             if (state not in self.state_tracker['state_seq']) and 's2' in self.state_tracker['state_seq']: 
#                 self.state_tracker['state_seq'].append(state)

            


#     def _show_feedback(self, frame, c_frame, dict_maps, lower_hips_disp):


#         if lower_hips_disp:
#             draw_text(
#                     frame, 
#                     'LOWER YOUR HIPS', 
#                     pos=(30, 80),
#                     text_color=(0, 0, 0),
#                     font_scale=0.6,
#                     text_color_bg=(255, 255, 0)
#                 )  

#         for idx in np.where(c_frame)[0]:
#             draw_text(
#                     frame, 
#                     dict_maps[idx][0], 
#                     pos=(30, dict_maps[idx][1]),
#                     text_color=(255, 255, 230),
#                     font_scale=0.6,
#                     text_color_bg=dict_maps[idx][2]
#                 )

#         return frame



#     def process(self, frame: np.array, pose):
#         play_sound = None
       

#         frame_height, frame_width, _ = frame.shape

#         # Process the image.
#         keypoints = pose.process(frame)

#         if keypoints.pose_landmarks:
#             ps_lm = keypoints.pose_landmarks

#             nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
#             left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
#                                 get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
#             right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
#                                 get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)

#             offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)

#             if offset_angle > self.thresholds['OFFSET_THRESH']:
                
#                 display_inactivity = False

#                 end_time = time.perf_counter()
#                 self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - self.state_tracker['start_inactive_time_front']
#                 self.state_tracker['start_inactive_time_front'] = end_time

#                 if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']:
#                     self.state_tracker['SQUAT_COUNT'] = 0
#                     self.state_tracker['IMPROPER_SQUAT'] = 0
#                     display_inactivity = True

#                 cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
#                 cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1)
#                 cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)

#                 if self.flip_frame:
#                     frame = cv2.flip(frame, 1)

#                 if display_inactivity:
#                     # cv2.putText(frame, 'Resetting SQUAT_COUNT due to inactivity!!!', (10, frame_height - 90), 
#                     #             self.font, 0.5, self.COLORS['blue'], 2, lineType=self.linetype)
#                     play_sound = 'reset_counters'
#                     self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
#                     self.state_tracker['start_inactive_time_front'] = time.perf_counter()

#                 draw_text(
#                     frame, 
#                     "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
#                     pos=(int(frame_width*0.75), 30),
#                     text_color=(255, 255, 230),
#                     font_scale=0.7,
#                     text_color_bg=(18, 185, 0)
#                 )  
                

#                 draw_text(
#                     frame, 
#                     "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
#                     pos=(int(frame_width*0.75), 80),
#                     text_color=(255, 255, 230),
#                     font_scale=0.7,
#                     text_color_bg=(221, 0, 0),
                    
#                 )  
                
                
#                 draw_text(
#                     frame, 
#                     'CAMERA NOT ALIGNED PROPERLY!!!', 
#                     pos=(30, frame_height-60),
#                     text_color=(255, 255, 230),
#                     font_scale=0.65,
#                     text_color_bg=(255, 153, 0),
#                 ) 
                
                
#                 draw_text(
#                     frame, 
#                     'OFFSET ANGLE: '+str(offset_angle), 
#                     pos=(30, frame_height-30),
#                     text_color=(255, 255, 230),
#                     font_scale=0.65,
#                     text_color_bg=(255, 153, 0),
#                 ) 

#                 # Reset inactive times for side view.
#                 self.state_tracker['start_inactive_time'] = time.perf_counter()
#                 self.state_tracker['INACTIVE_TIME'] = 0.0
#                 self.state_tracker['prev_state'] =  None
#                 self.state_tracker['curr_state'] = None
            
#             # Camera is aligned properly.
#             else:

#                 self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
#                 self.state_tracker['start_inactive_time_front'] = time.perf_counter()


#                 dist_l_sh_hip = abs(left_foot_coord[1]- left_shldr_coord[1])
#                 dist_r_sh_hip = abs(right_foot_coord[1] - right_shldr_coord)[1]

#                 shldr_coord = None
#                 elbow_coord = None
#                 wrist_coord = None
#                 hip_coord = None
#                 knee_coord = None
#                 ankle_coord = None
#                 foot_coord = None

#                 if dist_l_sh_hip > dist_r_sh_hip:
#                     shldr_coord = left_shldr_coord
#                     elbow_coord = left_elbow_coord
#                     wrist_coord = left_wrist_coord
#                     hip_coord = left_hip_coord
#                     knee_coord = left_knee_coord
#                     ankle_coord = left_ankle_coord
#                     foot_coord = left_foot_coord

#                     multiplier = -1
                                     
                
#                 else:
#                     shldr_coord = right_shldr_coord
#                     elbow_coord = right_elbow_coord
#                     wrist_coord = right_wrist_coord
#                     hip_coord = right_hip_coord
#                     knee_coord = right_knee_coord
#                     ankle_coord = right_ankle_coord
#                     foot_coord = right_foot_coord

#                     multiplier = 1
                    

#                 # ------------------- Verical Angle calculation --------------
                
#                 hip_vertical_angle = find_angle(shldr_coord, np.array([hip_coord[0], 0]), hip_coord)
#                 cv2.ellipse(frame, hip_coord, (30, 30), 
#                             angle = 0, startAngle = -90, endAngle = -90+multiplier*hip_vertical_angle, 
#                             color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

#                 cv2.line(frame, (hip_coord[0], hip_coord[1] + 20), (hip_coord[0], hip_coord[1] - 80), self.COLORS['blue'], 4, lineType=self.linetype)




#                 knee_vertical_angle = find_angle(hip_coord, np.array([knee_coord[0], 0]), knee_coord)
#                 cv2.ellipse(frame, knee_coord, (20, 20), 
#                             angle = 0, startAngle = -90, endAngle = -90-multiplier*knee_vertical_angle, 
#                             color = self.COLORS['white'], thickness = 3,  lineType = self.linetype)

#                 cv2.line(frame, (knee_coord[0], knee_coord[1] + 20), (knee_coord[0], knee_coord[1] - 50), self.COLORS['blue'], 4, lineType=self.linetype)



#                 ankle_vertical_angle = find_angle(knee_coord, np.array([ankle_coord[0], 0]), ankle_coord)
#                 cv2.ellipse(frame, ankle_coord, (30, 30),
#                             angle = 0, startAngle = -90, endAngle = -90 + multiplier*ankle_vertical_angle,
#                             color = self.COLORS['white'], thickness = 3,  lineType=self.linetype)

#                 cv2.line(frame, (ankle_coord[0], ankle_coord[1] + 20), (ankle_coord[0], ankle_coord[1] - 50), self.COLORS['blue'], 4, lineType=self.linetype)

#                 # ------------------------------------------------------------
        
                
#                 # Join landmarks.
#                 cv2.line(frame, shldr_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
#                 cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
#                 cv2.line(frame, shldr_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
#                 cv2.line(frame, knee_coord, hip_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
#                 cv2.line(frame, ankle_coord, knee_coord,self.COLORS['light_blue'], 4,  lineType=self.linetype)
#                 cv2.line(frame, ankle_coord, foot_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                
#                 # Plot landmark points
#                 cv2.circle(frame, shldr_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
#                 cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
#                 cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
#                 cv2.circle(frame, hip_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
#                 cv2.circle(frame, knee_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
#                 cv2.circle(frame, ankle_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
#                 cv2.circle(frame, foot_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)

                

#                 current_state = self._get_state(int(knee_vertical_angle))
#                 self.state_tracker['curr_state'] = current_state
#                 self._update_state_sequence(current_state)



#                 # -------------------------------------- COMPUTE COUNTERS --------------------------------------

#                 if current_state == 's1':

#                     if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
#                         self.state_tracker['SQUAT_COUNT']+=1
#                         play_sound = str(self.state_tracker['SQUAT_COUNT'])
                        
#                     elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq'])==1:
#                         self.state_tracker['IMPROPER_SQUAT']+=1
#                         play_sound = 'incorrect'

#                     elif self.state_tracker['INCORRECT_POSTURE']:
#                         self.state_tracker['IMPROPER_SQUAT']+=1
#                         play_sound = 'incorrect'
                        
                    
#                     self.state_tracker['state_seq'] = []
#                     self.state_tracker['INCORRECT_POSTURE'] = False


#                 # ----------------------------------------------------------------------------------------------------




#                 # -------------------------------------- PERFORM FEEDBACK ACTIONS --------------------------------------

#                 else:
#                     if hip_vertical_angle > self.thresholds['HIP_THRESH'][1]:
#                         self.state_tracker['DISPLAY_TEXT'][0] = True
                        

#                     elif hip_vertical_angle < self.thresholds['HIP_THRESH'][0] and \
#                          self.state_tracker['state_seq'].count('s2')==1:
#                             self.state_tracker['DISPLAY_TEXT'][1] = True
                        
                                        
                    
#                     if self.thresholds['KNEE_THRESH'][0] < knee_vertical_angle < self.thresholds['KNEE_THRESH'][1] and \
#                        self.state_tracker['state_seq'].count('s2')==1:
#                         self.state_tracker['LOWER_HIPS'] = True


#                     elif knee_vertical_angle > self.thresholds['KNEE_THRESH'][2]:
#                         self.state_tracker['DISPLAY_TEXT'][3] = True
#                         self.state_tracker['INCORRECT_POSTURE'] = True

                    
#                     if (ankle_vertical_angle > self.thresholds['ANKLE_THRESH']):
#                         self.state_tracker['DISPLAY_TEXT'][2] = True
#                         self.state_tracker['INCORRECT_POSTURE'] = True


#                 # ----------------------------------------------------------------------------------------------------


                
                
#                 # ----------------------------------- COMPUTE INACTIVITY ---------------------------------------------

#                 display_inactivity = False
                
#                 if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

#                     end_time = time.perf_counter()
#                     self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
#                     self.state_tracker['start_inactive_time'] = end_time

#                     if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
#                         self.state_tracker['SQUAT_COUNT'] = 0
#                         self.state_tracker['IMPROPER_SQUAT'] = 0
#                         display_inactivity = True

                
#                 else:
                    
#                     self.state_tracker['start_inactive_time'] = time.perf_counter()
#                     self.state_tracker['INACTIVE_TIME'] = 0.0

#                 # -------------------------------------------------------------------------------------------------------
              


#                 hip_text_coord_x = hip_coord[0] + 10
#                 knee_text_coord_x = knee_coord[0] + 15
#                 ankle_text_coord_x = ankle_coord[0] + 10

#                 if self.flip_frame:
#                     frame = cv2.flip(frame, 1)
#                     hip_text_coord_x = frame_width - hip_coord[0] + 10
#                     knee_text_coord_x = frame_width - knee_coord[0] + 15
#                     ankle_text_coord_x = frame_width - ankle_coord[0] + 10

                
                
#                 if 's3' in self.state_tracker['state_seq']:
#                     self.state_tracker['LOWER_HIPS'] = False

#                 self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']]+=1

#                 frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP, self.state_tracker['LOWER_HIPS'])



#                 if display_inactivity:
#                     # cv2.putText(frame, 'Resetting COUNTERS due to inactivity!!!', (10, frame_height - 20), self.font, 0.5, self.COLORS['blue'], 2, lineType=self.linetype)
#                     play_sound = 'reset_counters'
#                     self.state_tracker['start_inactive_time'] = time.perf_counter()
#                     self.state_tracker['INACTIVE_TIME'] = 0.0

                
#                 cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x, hip_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
#                 cv2.putText(frame, str(int(knee_vertical_angle)), (knee_text_coord_x, knee_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
#                 cv2.putText(frame, str(int(ankle_vertical_angle)), (ankle_text_coord_x, ankle_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)

                 
#                 draw_text(
#                     frame, 
#                     "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
#                     pos=(int(frame_width*0.75), 30),
#                     text_color=(255, 255, 230),
#                     font_scale=0.7,
#                     text_color_bg=(18, 185, 0)
#                 )  
                

#                 draw_text(
#                     frame, 
#                     "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
#                     pos=(int(frame_width*0.75), 80),
#                     text_color=(255, 255, 230),
#                     font_scale=0.7,
#                     text_color_bg=(221, 0, 0),
                    
#                 )  
                
                
#                 self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
#                 self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0    
#                 self.state_tracker['prev_state'] = current_state
                                  

       
        
#         else:

#             if self.flip_frame:
#                 frame = cv2.flip(frame, 1)

#             end_time = time.perf_counter()
#             self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']

#             display_inactivity = False

#             if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
#                 self.state_tracker['SQUAT_COUNT'] = 0
#                 self.state_tracker['IMPROPER_SQUAT'] = 0
#                 # cv2.putText(frame, 'Resetting SQUAT_COUNT due to inactivity!!!', (10, frame_height - 25), self.font, 0.7, self.COLORS['blue'], 2)
#                 display_inactivity = True

#             self.state_tracker['start_inactive_time'] = end_time

#             draw_text(
#                     frame, 
#                     "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
#                     pos=(int(frame_width*0.75), 30),
#                     text_color=(255, 255, 230),
#                     font_scale=0.7,
#                     text_color_bg=(18, 185, 0)
#                 )  
                

#             draw_text(
#                     frame, 
#                     "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
#                     pos=(int(frame_width*0.75), 80),
#                     text_color=(255, 255, 230),
#                     font_scale=0.7,
#                     text_color_bg=(221, 0, 0),
                    
#                 )  

#             if display_inactivity:
#                 play_sound = 'reset_counters'
#                 self.state_tracker['start_inactive_time'] = time.perf_counter()
#                 self.state_tracker['INACTIVE_TIME'] = 0.0
            
            
#             # Reset all other state variables
            
#             self.state_tracker['prev_state'] =  None
#             self.state_tracker['curr_state'] = None
#             self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
#             self.state_tracker['INCORRECT_POSTURE'] = False
#             self.state_tracker['DISPLAY_TEXT'] = np.full((5,), False)
#             self.state_tracker['COUNT_FRAMES'] = np.zeros((5,), dtype=np.int64)
#             self.state_tracker['start_inactive_time_front'] = time.perf_counter()
            
            
            
#         return frame, play_sound

                    




import time
import cv2
import numpy as np
from .utils import find_angle, get_landmark_features, draw_text


class ProcessFrame:
    def __init__(self, thresholds, flip_frame = False):
        
        # Set if frame should be flipped or not.
        self.flip_frame = flip_frame

        # self.thresholds
        self.thresholds = thresholds

        # Font type.
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # line type
        self.linetype = cv2.LINE_AA

        # set radius to draw arc
        self.radius = 20

        # Colors in BGR format.
        self.COLORS = {
                        'blue'       : (0, 127, 255),
                        'red'        : (255, 50, 50),
                        'green'      : (0, 255, 127),
                        'light_green': (100, 233, 127),
                        'yellow'     : (255, 255, 0),
                        'magenta'    : (255, 0, 255),
                        'white'      : (255,255,255),
                        'cyan'       : (0, 255, 255),
                        'light_blue' : (102, 204, 255),
                        'orange'     : (16, 117, 245)  # BGR format for orange
                      }



        # Dictionary to maintain the various landmark features.
        self.dict_features = {}
        self.left_features = {
                                'shoulder': 11,
                                'elbow'   : 13,
                                'wrist'   : 15,                    
                                'hip'     : 23,
                                'knee'    : 25,
                                'ankle'   : 27,
                                'foot'    : 31
                             }

        self.right_features = {
                                'shoulder': 12,
                                'elbow'   : 14,
                                'wrist'   : 16,
                                'hip'     : 24,
                                'knee'    : 26,
                                'ankle'   : 28,
                                'foot'    : 32
                              }

        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0

        
        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            'state_seq': [],

            'start_inactive_time': time.perf_counter(),
            'start_inactive_time_front': time.perf_counter(),
            'INACTIVE_TIME': 0.0,
            'INACTIVE_TIME_FRONT': 0.0,

            # 0 --> Bend Backwards, 1 --> Bend Forward, 2 --> Keep shin straight, 3 --> Deep squat
            'DISPLAY_TEXT' : np.full((4,), False),
            'COUNT_FRAMES' : np.zeros((4,), dtype=np.int64),

            'LOWER_HIPS': False,

            'INCORRECT_POSTURE': False,

            'prev_state': None,
            'curr_state':None,

            'SQUAT_COUNT': 0,
            'IMPROPER_SQUAT':0,
            
            'FORM_FEEDBACK': 'STAND STRAIGHT',
            'FEEDBACK_COLOR': (0, 255, 0)  # Default green
        }
        
        self.FEEDBACK_ID_MAP = {
                                0: ('BEND BACKWARDS', 215, (0, 153, 255)),
                                1: ('BEND FORWARD', 215, (0, 153, 255)),
                                2: ('KNEE FALLING OVER TOE', 170, (255, 80, 80)),
                                3: ('SQUAT TOO DEEP', 125, (255, 80, 80))
                               }

        


    def _get_state(self, knee_angle):
        
        knee = None        

        if self.thresholds['HIP_KNEE_VERT']['NORMAL'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['NORMAL'][1]:
            knee = 1
        elif self.thresholds['HIP_KNEE_VERT']['TRANS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['TRANS'][1]:
            knee = 2
        elif self.thresholds['HIP_KNEE_VERT']['PASS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['PASS'][1]:
            knee = 3

        return f's{knee}' if knee else None



    
    def _update_state_sequence(self, state):

        if state == 's2':
            if (('s3' not in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2'))==0) or \
                    (('s3' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2')==1)):
                        self.state_tracker['state_seq'].append(state)
            

        elif state == 's3':
            if (state not in self.state_tracker['state_seq']) and 's2' in self.state_tracker['state_seq']: 
                self.state_tracker['state_seq'].append(state)

            


    def _show_feedback(self, frame, c_frame, dict_maps, lower_hips_disp):
        # Update form feedback based on current issues
        feedback_text = 'GOOD FORM'
        feedback_color = self.COLORS['green']

        if lower_hips_disp:
            feedback_text = 'LOWER YOUR HIPS'
            feedback_color = self.COLORS['yellow']

        for idx in np.where(c_frame)[0]:
            # Set the most important feedback to display in UI
            feedback_text = dict_maps[idx][0]
            feedback_color = dict_maps[idx][2]
            
            # Still show all individual feedback messages in their original positions
            draw_text(
                    frame, 
                    dict_maps[idx][0], 
                    pos=(30, dict_maps[idx][1]),
                    text_color=(255, 255, 230),
                    font_scale=0.6,
                    text_color_bg=dict_maps[idx][2]
                )

        # Update the state tracker with current feedback
        self.state_tracker['FORM_FEEDBACK'] = feedback_text
        self.state_tracker['FEEDBACK_COLOR'] = feedback_color

        return frame



    def process(self, frame: np.array, pose):
        play_sound = None
    
        frame_height, frame_width, _ = frame.shape

        # Process the image.
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks

            nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
            left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
            right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)

            if offset_angle > self.thresholds['OFFSET_THRESH']:
                
                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']:
                    self.state_tracker['SQUAT_COUNT'] = 0
                    self.state_tracker['IMPROPER_SQUAT'] = 0
                    display_inactivity = True

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                # Update form feedback
                self.state_tracker['FORM_FEEDBACK'] = 'CAMERA NOT ALIGNED'
                self.state_tracker['FEEDBACK_COLOR'] = self.COLORS['orange']
                
                # Draw new UI elements
                # Calculate widths for UI elements
                counter_width = 150
                feedback_width = frame_width - counter_width - 300
                correct_width = 150
                
                # Total squat counter (left side)
                cv2.rectangle(frame, (10, 10), (10 + counter_width, 60), self.COLORS['orange'], -1)
                cv2.putText(frame, 'TOTAL REPS', (15, 30), self.font, 0.5, (0, 0, 0), 1, self.linetype)
                cv2.putText(frame, str(self.state_tracker['SQUAT_COUNT']), (50, 50), self.font, 1.0, self.COLORS['white'], 2, self.linetype)
                
                # Correct form counter (right side)
                cv2.rectangle(frame, (frame_width - 10 - correct_width, 10), (frame_width - 10, 60), self.COLORS['green'], -1)
                cv2.putText(frame, 'CORRECT', (frame_width - correct_width + 5, 30), self.font, 0.5, (0, 0, 0), 1, self.linetype)
                cv2.putText(frame, str(self.state_tracker['SQUAT_COUNT']), (frame_width - correct_width + 50, 50), self.font, 1.0, self.COLORS['white'], 2, self.linetype)
                
                # Form feedback (center)
                feedback_x = 10 + counter_width + 10
                feedback_width = frame_width - feedback_x - correct_width - 20
                cv2.rectangle(frame, (feedback_x, 10), (feedback_x + feedback_width, 60), self.state_tracker['FEEDBACK_COLOR'], -1)
                
                # Ensure form feedback text fits in the box
                form_text = self.state_tracker['FORM_FEEDBACK']
                font_scale = 0.7
                thickness = 2
                
                # Calculate text width to ensure it fits
                text_size = cv2.getTextSize(form_text, self.font, font_scale, thickness)[0]
                
                # If text is too long, reduce font size
                if text_size[0] > feedback_width - 20:
                    font_scale = 0.5
                    text_size = cv2.getTextSize(form_text, self.font, font_scale, thickness)[0]
                
                # Calculate text position to center it
                text_x = feedback_x + (feedback_width - text_size[0]) // 2
                text_y = 40
                cv2.putText(frame, 'FORM', (feedback_x + 5, 30), self.font, 0.5, (0, 0, 0), 1, self.linetype)
                cv2.putText(frame, form_text, (text_x, text_y), self.font, font_scale, self.COLORS['white'], thickness, self.linetype)

                # draw_text(
                #     frame, 
                #     'CAMERA NOT ALIGNED PROPERLY!!!', 
                #     pos=(30, frame_height-60),
                #     text_color=(255, 255, 230),
                #     font_scale=0.65,
                #     text_color_bg=(255, 153, 0),
                # ) 
                
                
                # draw_text(
                #     frame, 
                #     'OFFSET ANGLE: '+str(offset_angle), 
                #     pos=(30, frame_height-30),
                #     text_color=(255, 255, 230),
                #     font_scale=0.65,
                #     text_color_bg=(255, 153, 0),
                # ) 

                # Reset inactive times for side view.
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0
                self.state_tracker['prev_state'] =  None
                self.state_tracker['curr_state'] = None
            
            # Camera is aligned properly.
            else:

                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                # FIX: Correct distance calculation for right side
                dist_l_sh_hip = abs(left_foot_coord[1] - left_shldr_coord[1])
                dist_r_sh_hip = abs(right_foot_coord[1] - right_shldr_coord[1])  # Fixed this line

                shldr_coord = None
                elbow_coord = None
                wrist_coord = None
                hip_coord = None
                knee_coord = None
                ankle_coord = None
                foot_coord = None

                if dist_l_sh_hip > dist_r_sh_hip:
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord
                    hip_coord = left_hip_coord
                    knee_coord = left_knee_coord
                    ankle_coord = left_ankle_coord
                    foot_coord = left_foot_coord

                    multiplier = -1
                                    
                
                else:
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord
                    hip_coord = right_hip_coord
                    knee_coord = right_knee_coord
                    ankle_coord = right_ankle_coord
                    foot_coord = right_foot_coord

                    multiplier = 1
                    

                # ------------------- Verical Angle calculation --------------
                
                hip_vertical_angle = find_angle(shldr_coord, np.array([hip_coord[0], 0]), hip_coord)
                cv2.ellipse(frame, hip_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*hip_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

                cv2.line(frame, (hip_coord[0], hip_coord[1] + 20), (hip_coord[0], hip_coord[1] - 80), self.COLORS['blue'], 4, lineType=self.linetype)




                knee_vertical_angle = find_angle(hip_coord, np.array([knee_coord[0], 0]), knee_coord)
                cv2.ellipse(frame, knee_coord, (20, 20), 
                            angle = 0, startAngle = -90, endAngle = -90-multiplier*knee_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3,  lineType = self.linetype)

                cv2.line(frame, (knee_coord[0], knee_coord[1] + 20), (knee_coord[0], knee_coord[1] - 50), self.COLORS['blue'], 4, lineType=self.linetype)



                ankle_vertical_angle = find_angle(knee_coord, np.array([ankle_coord[0], 0]), ankle_coord)
                cv2.ellipse(frame, ankle_coord, (30, 30),
                            angle = 0, startAngle = -90, endAngle = -90 + multiplier*ankle_vertical_angle,
                            color = self.COLORS['white'], thickness = 3,  lineType=self.linetype)

                cv2.line(frame, (ankle_coord[0], ankle_coord[1] + 20), (ankle_coord[0], ankle_coord[1] - 50), self.COLORS['blue'], 4, lineType=self.linetype)

                # ------------------------------------------------------------
        
                
                # Join landmarks.
                cv2.line(frame, shldr_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, shldr_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, knee_coord, hip_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, knee_coord,self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, foot_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                
                # Plot landmark points
                cv2.circle(frame, shldr_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, hip_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, knee_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, ankle_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, foot_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)

                

                current_state = self._get_state(int(knee_vertical_angle))
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)



                # -------------------------------------- COMPUTE COUNTERS --------------------------------------

                if current_state == 's1':
                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['SQUAT_COUNT']+=1
                        play_sound = str(self.state_tracker['SQUAT_COUNT'])
                        # Update feedback for good squat
                        self.state_tracker['FORM_FEEDBACK'] = 'GOOD FORM'
                        self.state_tracker['FEEDBACK_COLOR'] = self.COLORS['green']
                        
                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq'])==1:
                        self.state_tracker['IMPROPER_SQUAT']+=1
                        play_sound = 'incorrect'
                        # Update feedback for improper squat
                        self.state_tracker['FORM_FEEDBACK'] = 'INCOMPLETE SQUAT'
                        self.state_tracker['FEEDBACK_COLOR'] = self.COLORS['red']

                    elif self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['IMPROPER_SQUAT']+=1
                        play_sound = 'incorrect'
                        # Form feedback is already set by _show_feedback
                        
                    
                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False


                # ----------------------------------------------------------------------------------------------------




                # -------------------------------------- PERFORM FEEDBACK ACTIONS --------------------------------------

                else:
                    if hip_vertical_angle > self.thresholds['HIP_THRESH'][1]:
                        self.state_tracker['DISPLAY_TEXT'][0] = True
                        

                    elif hip_vertical_angle < self.thresholds['HIP_THRESH'][0] and \
                        self.state_tracker['state_seq'].count('s2')==1:
                            self.state_tracker['DISPLAY_TEXT'][1] = True
                        
                                        
                    
                    if self.thresholds['KNEE_THRESH'][0] < knee_vertical_angle < self.thresholds['KNEE_THRESH'][1] and \
                    self.state_tracker['state_seq'].count('s2')==1:
                        self.state_tracker['LOWER_HIPS'] = True


                    elif knee_vertical_angle > self.thresholds['KNEE_THRESH'][2]:
                        self.state_tracker['DISPLAY_TEXT'][3] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True

                    
                    if (ankle_vertical_angle > self.thresholds['ANKLE_THRESH']):
                        self.state_tracker['DISPLAY_TEXT'][2] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True


                # ----------------------------------------------------------------------------------------------------


                
                
                # ----------------------------------- COMPUTE INACTIVITY ---------------------------------------------

                display_inactivity = False
                
                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
                    self.state_tracker['start_inactive_time'] = end_time

                    if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                        self.state_tracker['SQUAT_COUNT'] = 0
                        self.state_tracker['IMPROPER_SQUAT'] = 0
                        display_inactivity = True

                
                else:
                    
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                # -------------------------------------------------------------------------------------------------------
            


                hip_text_coord_x = hip_coord[0] + 10
                knee_text_coord_x = knee_coord[0] + 15
                ankle_text_coord_x = ankle_coord[0] + 10

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    hip_text_coord_x = frame_width - hip_coord[0] + 10
                    knee_text_coord_x = frame_width - knee_coord[0] + 15
                    ankle_text_coord_x = frame_width - ankle_coord[0] + 10

                
                
                if 's3' in self.state_tracker['state_seq']:
                    self.state_tracker['LOWER_HIPS'] = False

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']]+=1

                frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP, self.state_tracker['LOWER_HIPS'])



                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0
                    self.state_tracker['FORM_FEEDBACK'] = 'RESET DUE TO INACTIVITY'
                    self.state_tracker['FEEDBACK_COLOR'] = self.COLORS['orange']

                
                cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x, hip_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(knee_vertical_angle)), (knee_text_coord_x, knee_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(ankle_vertical_angle)), (ankle_text_coord_x, ankle_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)

                # Draw new UI elements
                # Calculate widths for UI elements
                counter_width = 150
                feedback_width = frame_width - counter_width - 300
                correct_width = 150
                
                # Total squat counter (left side)
                cv2.rectangle(frame, (10, 10), (10 + counter_width, 60), self.COLORS['orange'], -1)
                cv2.putText(frame, 'TOTAL REPS', (15, 30), self.font, 0.5, (0, 0, 0), 1, self.linetype)
                cv2.putText(frame, str(self.state_tracker['SQUAT_COUNT'] + self.state_tracker['IMPROPER_SQUAT']), (50, 50), self.font, 1.0, self.COLORS['white'], 2, self.linetype)
                
                # Correct form counter (right side)
                cv2.rectangle(frame, (frame_width - 10 - correct_width, 10), (frame_width - 10, 60), self.COLORS['green'], -1)
                cv2.putText(frame, 'CORRECT', (frame_width - correct_width + 5, 30), self.font, 0.5, (0, 0, 0), 1, self.linetype)
                cv2.putText(frame, str(self.state_tracker['SQUAT_COUNT']), (frame_width - correct_width + 50, 50), self.font, 1.0, self.COLORS['white'], 2, self.linetype)
                
                # Form feedback (center)
                feedback_x = 10 + counter_width + 10
                feedback_width = frame_width - feedback_x - correct_width - 20
                cv2.rectangle(frame, (feedback_x, 10), (feedback_x + feedback_width, 60), self.state_tracker['FEEDBACK_COLOR'], -1)
                
                # Ensure form feedback text fits in the box
                form_text = self.state_tracker['FORM_FEEDBACK']
                font_scale = 0.7
                thickness = 2
                
                # Calculate text width to ensure it fits
                text_size = cv2.getTextSize(form_text, self.font, font_scale, thickness)[0]
                
                # If text is too long, reduce font size
                if text_size[0] > feedback_width - 20:
                    font_scale = 0.5
                    text_size = cv2.getTextSize(form_text, self.font, font_scale, thickness)[0]
                
                # Calculate text position to center it
                text_x = feedback_x + (feedback_width - text_size[0]) // 2
                text_y = 40
                cv2.putText(frame, 'FORM', (feedback_x + 5, 30), self.font, 0.5, (0, 0, 0), 1, self.linetype)
                cv2.putText(frame, form_text, (text_x, text_y), self.font, font_scale, self.COLORS['white'], thickness, self.linetype)
                
                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0    
                self.state_tracker['prev_state'] = current_state
                                

    
        
        else:
            # When no pose landmarks are detected
            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                self.state_tracker['SQUAT_COUNT'] = 0
                self.state_tracker['IMPROPER_SQUAT'] = 0
                display_inactivity = True
                self.state_tracker['FORM_FEEDBACK'] = 'NO PERSON DETECTED'
                self.state_tracker['FEEDBACK_COLOR'] = self.COLORS['orange']

            self.state_tracker['start_inactive_time'] = end_time

            # Draw new UI elements
            # Calculate widths for UI elements
            counter_width = 150
            feedback_width = frame_width - counter_width - 300
            correct_width = 150
            
            # Total squat counter (left side)
            cv2.rectangle(frame, (10, 10), (10 + counter_width, 60), self.COLORS['orange'], -1)
            cv2.putText(frame, 'TOTAL REPS', (15, 30), self.font, 0.5, (0, 0, 0), 1, self.linetype)
            cv2.putText(frame, str(self.state_tracker['SQUAT_COUNT'] + self.state_tracker['IMPROPER_SQUAT']), (50, 50), self.font, 1.0, self.COLORS['white'], 2, self.linetype)
            
            # Correct form counter (right side)
            cv2.rectangle(frame, (frame_width - 10 - correct_width, 10), (frame_width - 10, 60), self.COLORS['green'], -1)
            cv2.putText(frame, 'CORRECT', (frame_width - correct_width + 5, 30), self.font, 0.5, (0, 0, 0), 1, self.linetype)
            cv2.putText(frame, str(self.state_tracker['SQUAT_COUNT']), (frame_width - correct_width + 50, 50), self.font, 1.0, self.COLORS['white'], 2, self.linetype)
            
            # Form feedback (center)
            feedback_x = 10 + counter_width + 10
            feedback_width = frame_width - feedback_x - correct_width - 20
            cv2.rectangle(frame, (feedback_x, 10), (feedback_x + feedback_width, 60), self.state_tracker['FEEDBACK_COLOR'], -1)
            
            # Ensure form feedback text fits in the box
            form_text = self.state_tracker['FORM_FEEDBACK']
            font_scale = 0.7
            thickness = 2
            
            # Calculate text width to ensure it fits
            text_size = cv2.getTextSize(form_text, self.font, font_scale, thickness)[0]
            
            # If text is too long, reduce font size
            if text_size[0] > feedback_width - 20:
                font_scale = 0.5
                text_size = cv2.getTextSize(form_text, self.font, font_scale, thickness)[0]
            
            # Calculate text position to center it
            text_x = feedback_x + (feedback_width - text_size[0]) // 2
            text_y = 40
            cv2.putText(frame, 'FORM', (feedback_x + 5, 30), self.font, 0.5, (0, 0, 0), 1, self.linetype)
            cv2.putText(frame, form_text, (text_x, text_y), self.font, font_scale, self.COLORS['white'], thickness, self.linetype)
            
        return frame, play_sound