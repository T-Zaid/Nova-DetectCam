import mediapipe as mp
import numpy as np
import itertools
import cv2
import matplotlib.pyplot as plt

# to get the holistic model and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face = mp.solutions.face_mesh
seet=True
# initializing FaceMesh models with minimum detection attributes, these are the values we can play with
# the default values are weitten with them.

# FaceMesh(
#   static_image_mode=False, to work for images 
#   max_num_faces,     How many faces you want to work with
#   min_detection_confidence=0.5,   
#   min_tracking_confidence=0.5
# )

face_mesh_videos = mp_face.FaceMesh(static_image_mode = False, max_num_faces = 2, min_detection_confidence = 0.7, min_tracking_confidence = 0.7)

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
    Returns:
        status:       A dictionary containing isOpen statuses of the face part of all the detected faces.  
    '''
    
    # Dictionary to store the status of the face part of all the detected faces.
    status={}
    
    if face_part == 'MOUTH':
        INDEXES = mp_face.FACEMESH_LIPS
           
    elif face_part == 'LEFT EYE':
        INDEXES = mp_face.FACEMESH_LEFT_EYE
       
    elif face_part == 'RIGHT EYE':
        INDEXES = mp_face.FACEMESH_RIGHT_EYE 

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
        landmarks.append([int(face_landmarks.landmark[i].x * img_width), int(face_landmarks.landmark[i].y * img_height)])
    
    # Calculate the width and height of the face part.
    _, _, width, height = cv2.boundingRect(np.array(landmarks))
    
    # Convert the list of landmarks of the face part into a numpy array.
    landmarks = np.array(landmarks)
    
    # Retrurn the calculated width height and the landmarks of the face part.
    return width, height, landmarks

def overlay(image, filter_img, face_landmarks, face_part, INDEXES):
    '''
    This function will overlay a filter image over a face part of a person in the image/frame.
    Args:
        image:          The image of a person on which the filter image will be overlayed.
        filter_img:     The filter image that is needed to be overlayed on the image of the person.
        face_landmarks: The facial landmarks of the person in the image.
        face_part:      The name of the face part on which the filter image will be overlayed.
        INDEXES:        The indexes of landmarks of the face part.
    Returns:
        annotated_image: The image with the overlayed filter on the top of the specified face part.
    '''
    
    annotated_image = image.copy()
    
    # Errors can come when it resizes the filter image to a too small or a too large size .
    # So use a try block to avoid application crashing.
    try:
    
        # Get the width and height of filter image.
        filter_img_height, filter_img_width, _  = filter_img.shape

        # Get the height of the face part on which we will overlay the filter image.
        _, face_part_height, landmarks = Parts_Measurements(image, face_landmarks, INDEXES)
        
        # Specify the height to which the filter image is required to be resized.
        required_height = int(face_part_height*3)
        
        # Resize the filter image to the required height, while keeping the aspect ratio constant. 
        resized_filter_img = cv2.resize(filter_img, (int(filter_img_width * (required_height / filter_img_height)), required_height))
        
        # Get the new width and height of filter image.
        filter_img_height, filter_img_width, _  = resized_filter_img.shape

        # Convert the image to grayscale and apply the threshold to get the mask image.
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY), 25, 255, cv2.THRESH_BINARY_INV)

        # Calculate the center of the face part.
        center = landmarks.mean(axis=0).astype("int")

        # Check if the face part is mouth.
        if face_part == 'MOUTH':

            # Calculate the location where the smoke filter will be placed.  
            location = (int(center[0] - filter_img_width / 3), int(center[1]))

        # Otherwise if the face part is an eye.
        else:

            # Calculate the location where the eye filter image will be placed.  
            location = (int(center[0]-filter_img_width/2), int(center[1]-filter_img_height/2))

        # Retrieve the region of interest from the image where the filter image will be placed.
        ROI = image[location[1]: location[1] + filter_img_height, location[0]: location[0] + filter_img_width]

        # Perform Bitwise-AND operation. This will set the pixel values of the region where,
        # filter image will be placed to zero.
        resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask)

        # Add the resultant image and the resized filter image.
        # This will update the pixel values of the resultant image at the indexes where 
        # pixel values are zero, to the pixel values of the filter image.
        resultant_image = cv2.add(resultant_image, resized_filter_img)

        # Update the image's region of interest with resultant image.
        annotated_image[location[1]: location[1] + filter_img_height, location[0]: location[0] + filter_img_width] = resultant_image
            
    # Catch and handle the error(s).
    except Exception as e:
        pass
        
    # Return the annotated image.
    return annotated_image


def GetRectCoords(results, image, Filtertype):

    topLeft, topRight, bottomLeft, bottomRight = [], [], [], []

    # Creating the rectangle coordinates according to the filter type
    if Filtertype == "Face":
        topLeft = [float(results[0].landmark[54].x * image.shape[1]) - (float(results[0].landmark[54].x * image.shape[1])*0.10), float(results[0].landmark[54].y * image.shape[0]) - (float(results[0].landmark[54].y * image.shape[0]))*0.10]
        topRight = [float(results[0].landmark[284].x * image.shape[1]) + (float(results[0].landmark[284].x * image.shape[1])*0.10) , float(results[0].landmark[284].y * image.shape[0]) - (float(results[0].landmark[284].y * image.shape[0])*0.10)]
        bottomRight = [float(results[0].landmark[365].x * image.shape[1])+ (float(results[0].landmark[365].x * image.shape[1])*0.10) , float(results[0].landmark[365].y * image.shape[0]) + (float(results[0].landmark[365].y * image.shape[0])*0.10)]
        bottomLeft = [float(results[0].landmark[136].x * image.shape[1]) - (float(results[0].landmark[136].x * image.shape[1])*0.10) , float(results[0].landmark[136].y * image.shape[0]) + (float(results[0].landmark[136].y * image.shape[0])*0.10)]

    elif Filtertype == "Eyes":
        topLeft = [float(results[0].landmark[21].x * image.shape[1]), float(results[0].landmark[21].y * image.shape[0])]
        topRight = [float(results[0].landmark[251].x * image.shape[1]) , float(results[0].landmark[251].y * image.shape[0])]
        bottomRight = [float(results[0].landmark[323].x * image.shape[1]) , float(results[0].landmark[323].y * image.shape[0])]
        bottomLeft = [float(results[0].landmark[93].x * image.shape[1]) , float(results[0].landmark[93].y * image.shape[0])]

    elif Filtertype == "Head":
        topLeft = [float(results[0].landmark[54].x * image.shape[1]) , float(results[0].landmark[54].y * image.shape[0]) - (float(results[0].landmark[54].y * image.shape[0]))*0.50]
        topRight = [float(results[0].landmark[251].x * image.shape[1]) , float(results[0].landmark[54].y * image.shape[0]) - (float(results[0].landmark[54].y * image.shape[0])*0.50)]
        bottomRight = [float(results[0].landmark[251].x * image.shape[1]) , float(results[0].landmark[251].y * image.shape[0])]
        bottomLeft = [float(results[0].landmark[54].x * image.shape[1]) , float(results[0].landmark[54].y * image.shape[0])]

    elif Filtertype == "Mustache":
        topLeft = [float(results[0].landmark[205].x * image.shape[1]), float(results[0].landmark[205].y * image.shape[0])]
        topRight = [float(results[0].landmark[425].x * image.shape[1]) , float(results[0].landmark[425].y * image.shape[0])]
        bottomRight = [float(results[0].landmark[436].x * image.shape[1]) , float(results[0].landmark[436].y * image.shape[0])]
        bottomLeft = [float(results[0].landmark[216].x * image.shape[1]) , float(results[0].landmark[216].y * image.shape[0])]

    elif Filtertype == "Covid":
        topLeft = [float(results[0].landmark[127].x * image.shape[1]), float(results[0].landmark[127].y * image.shape[0])]
        topRight = [float(results[0].landmark[356].x * image.shape[1]) , float(results[0].landmark[356].y * image.shape[0])]
        bottomRight = [float(results[0].landmark[365].x * image.shape[1]) , float(results[0].landmark[152].y * image.shape[0])]
        bottomLeft = [float(results[0].landmark[136].x * image.shape[1]) , float(results[0].landmark[152].y * image.shape[0])]

    dstMat = np.array([ topLeft, topRight, bottomRight, bottomLeft ])
    return dstMat

def applyFilter(source, imageFace, dstMat):
    (imgH, imgW) = imageFace.shape[:2]

    # defining the transform matrix for the source image using the height and width factor
    (srcH, srcW) = source.shape[:2]          
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

    # calculating the homography matrix and then warping the source image to the
    # destination based on the homography
    (H, _) = cv2.findHomography(srcMat, dstMat)
    warped = cv2.warpPerspective(source, H, (imgW, imgH))

    # Split out the transparency mask from the colour info
    overlay_img = warped[:,:,:3] # Grab the BRG planes
    overlay_mask = warped[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
    global seet
    frame_height, frame_width, _ = imageFace.shape
    if seet==True:
        captured_image = cv2.copyMakeBorder(src=imageFace, top=10, bottom=10, left=10, right=10,
                                                    borderType=cv2.BORDER_CONSTANT)
                
        # Capture an image and store it in the disk.
        cv2.imwrite('Captured_Image.png', captured_image)
        
        # Display a black image.
        # Play the image capture music to indicate the an image is captured and wait for 100 milliseconds.
        cv2.waitKey(100)

        # Display the captured image.
        plt.close();cv2.figure(figsize=[10,10])
        cv2.imshow(imageFace[:,:,::-1]);cv2.title("Captured Image");cv2.axis('off');
        seet=True
    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (imageFace * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
    output = np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))
    return output



# the main driver function that will open the camera and detect
def modelDetection():
    # starting the camera
    cam = cv2.VideoCapture(0)
    right_eye = cv2.imread('filters/red_eyes_right.png')
    left_eye = cv2.imread('filters/red_eyes_left.png')
    smoke = cv2.VideoCapture('filters/smoke.gif')
    filter1 = cv2.imread('filters/thug_glasses.png', cv2.IMREAD_UNCHANGED)
    smoke_counter = 0

    while cam.isOpened():
        success, image = cam.read()

        # if the frame is not good, capture the next frame
        if not success:
            continue

        image = cv2.flip(image, 1)

        _, smoke_frame = smoke.read()
        smoke_counter += 1

        # Check if the current frame is the last frame of the smoke animation video.
        if smoke_counter == smoke.get(cv2.CAP_PROP_FRAME_COUNT):
            smoke.set(cv2.CAP_PROP_POS_FRAMES, 0)
            smoke_counter = 0

        # converting image to RGB so that mediapipe can process it
        image.flags.writeable = False   #in docs, it claims to improve performance
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_face_results = face_mesh_videos.process(image)

        # coverting image to BGR because CV loves BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # if mp_face_results.multi_face_landmarks:
        #     for face_landmarks in mp_face_results.multi_face_landmarks:
        #         # drawing face landmarks, tip : FACE_CONNECTIONS changed to FACEMESH_TESSELATION
        #         mp_drawing.draw_landmarks(image, face_landmarks, mp_face.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        #         mp_drawing.draw_landmarks(image, face_landmarks, mp_face.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

        if mp_face_results.multi_face_landmarks:
            mouthStatus = isOpen(image, mp_face_results, 'MOUTH', threshold=15)
            LeyeStatus = isOpen(image, mp_face_results, 'LEFT EYE', threshold=5)
            ReyeStatus = isOpen(image, mp_face_results, 'RIGHT EYE', threshold=5)

            # Iterate over the found faces.
            for face_num, face_landmarks in enumerate(mp_face_results.multi_face_landmarks):
                dstMat = GetRectCoords(mp_face_results.multi_face_landmarks, image, "Eyes")
                image = applyFilter(filter1, image, dstMat)
                # if ReyeStatus[face_num] == 'OPEN':
                #     image = overlay(image, right_eye, face_landmarks, 'RIGHT EYE', mp_face.FACEMESH_RIGHT_EYE)
                
                # if LeyeStatus[face_num] == 'OPEN':
                #     image = overlay(image, left_eye, face_landmarks, 'LEFT EYE', mp_face.FACEMESH_LEFT_EYE)
               
                # if mouthStatus[face_num] == 'OPEN':
                #     image = overlay(image, smoke_frame, face_landmarks, 'MOUTH', mp_face.FACEMESH_LIPS)

        cv2.imshow('Nova Detect', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    

modelDetection()