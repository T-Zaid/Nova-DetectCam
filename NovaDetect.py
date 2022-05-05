import mediapipe as mp
import cv2

# to get the holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# initializing holistic models with minimum detection attributes, these are the values we can play with
# the default values are weitten with them.

# Holistic(
#   static_image_mode=False, to work for images 
#   model_complexity=1,     high complexity leads to high accuaracy in detection but slower process
#   smooth_landmarks=True,  reduces jitter by filtering inputs accross different input frames
#   min_detection_confidence=0.5,   
#   min_tracking_confidence=0.5
# )

holistic_model = mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

# starting the camera
cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()

    # converting image to RGB so that media pipe holistic can process it
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hollistic_model_results = holistic_model.process(image)

    # coverting image to BGR because CV loves BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # drawing face landmarks, tip : FACE_CONNECTIONS changed to FACEMESH_TESSELATION
    mp_drawing.draw_landmarks(image, hollistic_model_results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

    cv2.imshow('Face Image', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()