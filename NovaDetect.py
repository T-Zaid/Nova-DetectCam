import mediapipe as mp
import numpy as np
import itertools
import cv2
import matplotlib.pyplot as plt

# to get the holistic model and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face = mp.solutions.face_mesh

# initializing FaceMesh models with minimum detection attributes, these are the values we can play with
# the default values are weitten with them.

# FaceMesh(
#   static_image_mode=False, to work for images 
#   max_num_faces,     How many faces you want to work with
#   min_detection_confidence=0.5,   
#   min_tracking_confidence=0.5
# )

face_mesh_videos = mp_face.FaceMesh(static_image_mode = False, max_num_faces = 2, min_detection_confidence = 0.5, min_tracking_confidence = 0.3)

# Function that provides the status of faceparts as open or closed
def isOpen(image, face_mesh_results, face_part, threshold=5):
    '''
    This function checks whether the an eye or mouth of the person(s) is open, 
    utilizing its facial landmarks.
    Args:
        image:             The image of person(s) whose an eye or mouth is to be checked.
        face_mesh_results: The output of the facial landmarks detection on the image.
        face_part:         The name of the face part that is required to check.
        threshold:         The threshold value used to check the isOpen condition.
        display:           A boolean value that is if set to true the function displays 
                           the output image and returns nothing.
    Returns:
        status:       A dictionary containing isOpen statuses of the face part of all the 
                      detected faces.  
    '''
    
    # Dictionary to store the isOpen status of the face part of all the detected faces.
    status={}
    
    if face_part == 'MOUTH':
        INDEXES = mp_face.FACEMESH_LIPS
           
    elif face_part == 'LEFT EYE':
        INDEXES = mp_face.FACEMESH_LEFT_EYE
       
    elif face_part == 'RIGHT EYE':
        INDEXES = mp_face.FACEMESH_RIGHT_EYE 
    
    # Otherwise return nothing.
    else:
        return
    
    # Iterate over the found faces.
    for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
        
         # Get the height of the face part.
        _, height, _ = Parts_Measurements(image, face_landmarks, INDEXES)
        
         # Get the height of the whole face.
        _, face_height, _ = Parts_Measurements(image, face_landmarks, mp_face.FACEMESH_FACE_OVAL)
        
        # Check if the face part is open.
        if (height/face_height)*100 > threshold:
            status[face_no] = 'OPEN'
        else:
            status[face_no] = 'CLOSE'
    
        return status

def Parts_Measurements(image, face_landmarks, indexes):
    
    img_height, img_width, _ = image.shape
    # Convert the indexes of the landmarks of the face part into a list.
    indexes_list = list(itertools.chain(*indexes))
    
    # Initialize a list to store the landmarks of the face part.
    landmarks = []
    
    # Iterate over the indexes of the landmarks of the face part. 
    for i in indexes_list:
        
        # Append the landmark into the list.
        landmarks.append([int(face_landmarks.landmark[i].x * img_width),
                               int(face_landmarks.landmark[i].y * img_height)])
    
    # Calculate the width and height of the face part.
    _, _, width, height = cv2.boundingRect(np.array(landmarks))
    
    # Convert the list of landmarks of the face part into a numpy array.
    landmarks = np.array(landmarks)
    
    # Retrurn the calculated width height and the landmarks of the face part.
    return width, height, landmarks



# the main driver function that will open the camera and detect
def modelDetection():
    # starting the camera
    cam = cv2.VideoCapture(1)
    while cam.isOpened():
        success, image = cam.read()

        # if the frame is not good, capture the next frame
        if not success:
            continue

        image = cv2.flip(image, 1)

        # converting image to RGB so that media pipe holistic can process it
        image.flags.writeable = False   #in docs, it claims to improve performance
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_face_results = face_mesh_videos.process(image)

        # coverting image to BGR because CV loves BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if mp_face_results.multi_face_landmarks:
            for face_landmarks in mp_face_results.multi_face_landmarks:
                # drawing face landmarks, tip : FACE_CONNECTIONS changed to FACEMESH_TESSELATION
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

        if mp_face_results.multi_face_landmarks:
            mouthStatus = isOpen(image, mp_face_results, 'MOUTH', threshold=15)
            LeyeStatus = isOpen(image, mp_face_results, 'LEFT EYE', threshold=5)
            ReyeStatus = isOpen(image, mp_face_results, 'RIGHT EYE', threshold=5)
            print(mouthStatus, LeyeStatus, ReyeStatus)

        cv2.imshow('Face Image', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

modelDetection()