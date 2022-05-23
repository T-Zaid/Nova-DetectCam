import mediapipe as mp
import numpy as np
import itertools
import cv2
import matplotlib.pyplot as plt

# to get the holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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

def isOpen(image, face_mesh_results, face_part, threshold=5, display=True):
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
        output_image: The image of the person with the face part is opened  or not status written.
        status:       A dictionary containing isOpen statuses of the face part of all the 
                      detected faces.  
    '''
    
    # Retrieve the height and width of the image.
    image_height, image_width, _ = image.shape
    
    # Create a copy of the input image to write the isOpen status.
    output_image = image.copy()
    
    # Create a dictionary to store the isOpen status of the face part of all the detected faces.
    status={}
    
    # Check if the face part is mouth.
    if face_part == 'MOUTH':
        
        # Get the indexes of the mouth.
        # INDEXES = mp_holistic.FACEMESH_LIPS
        INDEXES = mp_holistic.FACELANDMARKS.LIPS
        
        # Specify the location to write the is mouth open status.
        loc = (10, image_height - image_height//40)
        
        # Initialize a increment that will be added to the status writing location, 
        # so that the statuses of two faces donot overlap. 
        increment=-30
        
    # Check if the face part is left eye.    
    elif face_part == 'LEFT EYE':
        
        # Get the indexes of the left eye.
        INDEXES = mp_holistic.FACEMESH_LEFT_EYE
        
        # Specify the location to write the is left eye open status.
        loc = (10, 30)
        
        # Initialize a increment that will be added to the status writing location, 
        # so that the statuses of two faces donot overlap.
        increment=30
    
    # Check if the face part is right eye.    
    elif face_part == 'RIGHT EYE':
        
        # Get the indexes of the right eye.
        INDEXES = mp_holistic.FACEMESH_RIGHT_EYE 
        
        # Specify the location to write the is right eye open status.
        loc = (image_width-300, 30)
        
        # Initialize a increment that will be added to the status writing location, 
        # so that the statuses of two faces donot overlap.
        increment=30
    
    # Otherwise return nothing.
    else:
        return
    
    # Iterate over the found faces.
    for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
        
         # Get the height of the face part.
        _, height, _ = Parts_Measurements(image, face_landmarks, INDEXES)
        
         # Get the height of the whole face.
        _, face_height, _ = Parts_Measurements(image, face_landmarks, mp_holistic.FACEMESH_FACE_OVAL)
        
        # Check if the face part is open.
        if (height/face_height)*100 > threshold:
            
            # Set status of the face part to open.
            status[face_no] = 'OPEN'
            
            # Set color which will be used to write the status to green.
            color=(0,255,0)
        
        # Otherwise.
        else:
            # Set status of the face part to close.
            status[face_no] = 'CLOSE'
            
            # Set color which will be used to write the status to red.
            color=(0,0,255)
        
        # Write the face part isOpen status on the output image at the appropriate location.
        cv2.putText(output_image, f'FACE {face_no+1} {face_part} {status[face_no]}.', 
                    (loc[0],loc[1]+(face_no*increment)), cv2.FONT_HERSHEY_PLAIN, 1.4, color, 2)
                
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:
        
        # Return the output image and the isOpen statuses of the face part of each detected face.
        return output_image, status

def Parts_Measurements(image, face_landmarks, indexes):
    
    img_height, img_width = image.shape
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
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        success, image = cap.read()

        # if the frame is not good, capture the next frame
        if not success:
            continue

        image = cv2.flip(image, 1)

        # converting image to RGB so that media pipe holistic can process it
        image.flags.writeable = False   #in docs, it claims to improve performance
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hollistic_model_results = holistic_model.process(image)

        # coverting image to BGR because CV loves BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # drawing face landmarks, tip : FACE_CONNECTIONS changed to FACEMESH_TESSELATION
        mp_drawing.draw_landmarks(image, hollistic_model_results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(image, hollistic_model_results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())


        # if hollistic_model_results.face_landmarks:
        #     output_image, _ = isOpen(image, hollistic_model_results, 'MOUTH', threshold=15, display=True)
        #     output_image, _ = isOpen(output_image, hollistic_model_results, 'LEFT EYE', threshold=5, display=True)
        #     isOpen(output_image, hollistic_model_results, 'RIGHT EYE', threshold=5)

        cv2.imshow('Face Image', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

modelDetection()