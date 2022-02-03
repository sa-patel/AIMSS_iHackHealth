import cv2
import mediapipe as mp
import numpy as np
import time 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Note: This is a for demonstration purposes only. It is not mature enough for testing and needs improvements before it is used for any medical application
# Some of the major steps needed to commercialize would be to streamline the calculation process, reduce the bugs and choose appropriate thresholds for the stroke scale

# Initializing variables for the analysis
stroke_scale_gaze = None
stroke_scale_palsy = None
stroke_scale_arms = None
stroke_scale_legs = None

# Eye Gaze Stroke Scale Calculation
# Returns the gaze component of stroke scale
def eye_gaze_test():
  eye_score = None
  print('Point your head directly in front of the camera, move your eyes to the LEFT')

  # For webcam input:
  cap = cv2.VideoCapture(0)
  start_time = time.time()
  end_time = time.time()
  switched_eye = 0

  max_right_eye_dist = -float("inf")
  min_right_eye_dist = float("inf")

  max_left_eye_dist = -float("inf")
  min_left_eye_dist = float("inf")

  with mp_holistic.Holistic(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5,
      refine_face_landmarks=True) as holistic:

    while cap.isOpened() and ((end_time - start_time) < 10):
      success, image = cap.read()
      # print(end_time-start_time)
      if not (switched_eye) and ((end_time - start_time) > 5):
        print('Point your head directly in front of the camera, move your eyes to the RIGHT')
        switched_eye = 1 

      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = holistic.process(image)

      # for pose_landmarks in pose_landmarks:
      #     #Landmark Map - see https://google.github.io/mediapipe/solutions/pose.html
      #     results.pose_landmarks[0]

      # Draw landmark annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      mp_drawing.draw_landmarks(
          image=image,
          landmark_list= results.face_landmarks,
          connections=mp_holistic.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())

      mp_drawing.draw_landmarks(
          image,
          results.face_landmarks,
          mp_holistic.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())

      mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_holistic.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles
          .get_default_pose_landmarks_style())

      # Wait for 10 seconds and switch eyes halfway through
      if (results.face_landmarks is not None):
        left_eye_inner = results.face_landmarks.landmark[362]
        left_eyeball = results.face_landmarks.landmark[473]
        left_eye_outer = results.face_landmarks.landmark[263]

        right_eye_inner = results.face_landmarks.landmark[133]
        right_eyeball = results.face_landmarks.landmark[468]
        right_eye_outer = results.face_landmarks.landmark[33]

        left_length = np.sqrt( ((left_eye_inner.x - left_eye_outer.x) **2) + ((left_eye_inner.y - left_eye_outer.y) **2))
        #expressed as a proportion of eye length
        left_eye_dist = np.sqrt(((left_eyeball.x - left_eye_outer.x) **2) + ((left_eyeball.y - left_eye_outer.y) **2)) / left_length
        
        right_length = np.sqrt( ((right_eye_inner.x - right_eye_outer.x) **2) + ((right_eye_inner.y - right_eye_outer.y) **2))   
        #expressed as a proportion of eye length
        right_eye_dist = np.sqrt(((right_eyeball.x - right_eye_outer.x) **2) + ((right_eyeball.y - right_eye_outer.y) **2)) / right_length

        # print('right_eye: ', right_eye_dist)
        # print('left_eye: ', left_eye_dist)
        max_right_eye_dist = max(right_eye_dist, max_right_eye_dist)
        min_right_eye_dist = min(right_eye_dist, min_right_eye_dist)

        max_left_eye_dist = max(left_eye_dist, max_left_eye_dist)
        min_left_eye_dist = min(left_eye_dist, min_left_eye_dist)

        end_time = time.time()
      
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

      # hit the key to break the video recording sequence
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()

  print('max_right_eye_dist: ', max_right_eye_dist)
  print('min_right_eye_dist: ', min_right_eye_dist)

  print('max_left_eye_dist: ', max_left_eye_dist)
  print('min_left_eye_dist: ', min_left_eye_dist)

  # lack of movement in both eyes
  if (abs(max_left_eye_dist - min_left_eye_dist) < 0.1) and (abs(max_right_eye_dist - min_right_eye_dist) < 0.1):
    eye_score = 2
  # lack of movement in both eyes
  elif (abs(max_left_eye_dist - min_left_eye_dist) < 0.1) or (abs(max_right_eye_dist - min_right_eye_dist)< 0.1):
    eye_score = 1
  else:
    eye_score = 0

  return eye_score

# Facial Palsy Stroke Scale Calculation
# A rough attempt at characterizing 
# Facial Palsy
# Grading
# 0 = Normal symmetrical movements. 
# 1 = Minor paralysis (flattened nasolabial fold, asymmetry on smiling). 
# 2 = Partial paralysis (total or near-total paralysis of lower face). 
# 3 = Complete paralysis of one or both sides (absence of facial movement in the upper and lower face)
# Approach: Visual
# Detecting facial asymmetry (top and bottom)
# Look at smile, nasolabial fold, and forehead wrinkles
# This can be improved with a baseline and utilizes the 478 facial points from the full network to compare
def facial_palsy_test():
  facial_palsy_score = None
  print('Point your head directly in front of the camera, holding your head still, then alternate betweening smiling and frowning')

  # For webcam input:
  cap = cv2.VideoCapture(0)
  start_time = time.time()
  end_time = time.time()

  max_right_deviation = -float("inf")
  min_right_deviation = float("inf")

  max_left_deviation = -float("inf")
  min_left_deviation = float("inf")

  with mp_holistic.Holistic(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5,
      refine_face_landmarks=True) as holistic:

    while cap.isOpened() and ((end_time - start_time) < 10):
      success, image = cap.read()

      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = holistic.process(image)

      # for pose_landmarks in pose_landmarks:
      #     #Landmark Map - see https://google.github.io/mediapipe/solutions/pose.html
      #     results.pose_landmarks[0]

      # Draw landmark annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      mp_drawing.draw_landmarks(
          image=image,
          landmark_list= results.face_landmarks,
          connections=mp_holistic.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())

      mp_drawing.draw_landmarks(
          image,
          results.face_landmarks,
          mp_holistic.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())

      mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_holistic.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles
          .get_default_pose_landmarks_style())

      # Wait for 10 seconds and switch eyes halfway through
      if (results.face_landmarks is not None):
        left_lip_corner = results.face_landmarks.landmark[291]
        right_lip_corner = results.face_landmarks.landmark[61]
        
        lip_length = np.sqrt( ((left_lip_corner.x - right_lip_corner.x) ** 2) + ((left_lip_corner.y - right_lip_corner.y) ** 2))

        right_lip_corner = results.face_landmarks.landmark[61]

        # vertical deviation expressed as a proportion of lip length
        # lip_deviation = abs(right_lip_corner.y - left_lip_corner.y) / lip_length

        # print('right_eye: ', right_eye_dist)
        # print('left_eye: ', left_eye_dist)
        max_left_deviation = max(left_lip_corner.y, max_left_deviation)
        min_left_deviation = min(left_lip_corner.y, min_left_deviation)

        max_right_deviation = max(right_lip_corner.y, max_right_deviation)
        min_right_deviation = min(right_lip_corner.y, min_right_deviation) 

        left_lip_deviation_normalized =  (max_left_deviation - min_left_deviation) / lip_length    
        right_lip_deviation_normalized =  (max_right_deviation - min_right_deviation) / lip_length

        end_time = time.time()
      
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

      # hit the key to break the video recording sequence
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()

  print('left_lip_deviation_normalized: ', left_lip_deviation_normalized)
  print('right_lip_deviation_normalized: ', right_lip_deviation_normalized)


  # A rough attempt at characterizing 
  # Facial Palsy
  # Grading
  # 0 = Normal symmetrical movements. 
  # 1 = Minor paralysis (flattened nasolabial fold, asymmetry on smiling). 
  # 2 = Partial paralysis (total or near-total paralysis of lower face). 
  # 3 = Complete paralysis of one or both sides (absence of facial movement in the upper and lower face)
  # MAKE SURE THESE buckets overlap!
  if abs(left_lip_deviation_normalized < 0.1) and abs(right_lip_deviation_normalized < 0.1):
    facial_palsy_score = 3
  elif abs(left_lip_deviation_normalized < 0.15) or abs(right_lip_deviation_normalized < 0.15):
    facial_palsy_score = 2
  elif abs(left_lip_deviation_normalized < 0.1) or abs(right_lip_deviation_normalized < 0.1):
    facial_palsy_score = 1
  else:
    facial_palsy_score = 0

  return facial_palsy_score

# stroke_scale_gaze = eye_gaze_test()
stroke_scale_palsy = facial_palsy_test()

#Return calculated stroke scales upon termination of program
print('stroke_scale_gaze: ', stroke_scale_gaze)
print('stroke_scale_palsy:', stroke_scale_palsy)
print('stroke_scale_arms: ', stroke_scale_arms)
print('stroke_scale_legs: ', stroke_scale_legs)

# # For static images: (included for reference)
# IMAGE_FILES = []
# with mp_holistic.Holistic(
#     static_image_mode=True,
#     model_complexity=2,
#     enable_segmentation=True,
#     refine_face_landmarks=True) as holistic:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     image_height, image_width, _ = image.shape
#     # Convert the BGR image to RGB before processing.
#     results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     if results.pose_landmarks:
#       print(
#           f'Nose coordinates: ('
#           f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
#           f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
#       )

#     annotated_image = image.copy()
#     # Draw segmentation on the image.
#     # To improve segmentation around boundaries, consider applying a joint
#     # bilateral filter to "results.segmentation_mask" with "image".
#     condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
#     bg_image = np.zeros(image.shape, dtype=np.uint8)
#     bg_image[:] = BG_COLOR
#     annotated_image = np.where(condition, annotated_image, bg_image)
#     # Draw pose, left and right hands, and face landmarks on the image.
#     mp_drawing.draw_landmarks(
#         annotated_image,
#         results.face_landmarks,
#         mp_holistic.FACEMESH_TESSELATION,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp_drawing_styles
#         .get_default_face_mesh_tesselation_style())
#     mp_drawing.draw_landmarks(
#         annotated_image,
#         results.pose_landmarks,
#         mp_holistic.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.
#         get_default_pose_landmarks_style())
#     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
#     # Plot pose world landmarks.
#     mp_drawing.plot_landmarks(
#         results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)