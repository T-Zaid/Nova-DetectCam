import mediapipe as mp
import numpy as np
import itertools
import cv2
import matplotlib.pyplot as plt
from protobuf_to_dict import protobuf_to_dict

# to get the face and hands model and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# initializing FaceMesh and Hands models with minimum detection attributes, these are the values we can play with
# the default values are weitten with them.

# FaceMesh/Hands(
#   static_image_mode=False, to work for images 
#   max_num_faces,     How many faces you want to work with
#   min_detection_confidence=0.5,   
#   min_tracking_confidence=0.5
# )

face_mesh_videos = mp_face.FaceMesh(static_image_mode = False, max_num_faces = 1, min_detection_confidence = 0.7, min_tracking_confidence = 0.7)
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def GestureChecker():
    if(isUP("finger1") and isUP("finger2") and not isUP("finger3") and not isUP("finger4") ):
        return "Veetit"
    elif(isUP("finger1") and isUP("finger4") and not isUP("finger3") and not isUP("finger2") and hand["thumb"][3]['x']>hand["finger1"][0]['x']):
        return "spiderman"
    elif(isDown("finger1") and isDown("finger2") and isDown("finger3") and isDown("finger4") and isUP("thumb")):
        return "fist"
    elif(isUP("finger1") and isUP("finger2") and isUP("finger3") and isUP("finger4") and isUP("thumb") and ArePartsAway("finger1","finger2") and ArePartsAway("finger2","finger3") and ArePartsAway("finger3","finger4")):
        return "handUP_open"
    elif(isUP("finger1") and isUP("finger2") and isUP("finger3") and isUP("finger4") and isUP("thumb") and not ArePartsAway("finger1","finger2") and not ArePartsAway("finger2","finger3") and not ArePartsAway("finger3","finger4")):
        return "handUP_close"
    elif(isUP("thumb")  and DistanceBTW("finger1")<0.25 and not isDown("finger1",'x') and isDown("finger2",'x') and isDown("finger3",'x') and isDown("finger4",'x')):
        return "gun"
    elif(isUP("thumb")  and isDown("finger1",'x') and isDown("finger2",'x') and isDown("finger3",'x') and isDown("finger4",'x')):
        return "thumbsup"
    elif(isDown("thumb")  and isDown("finger1",'x') and isDown("finger2",'x') and isDown("finger3",'x') and isDown("finger4",'x')):
        print("thumbsdown")
        return "thumbsdown"
    elif(isUP("finger4") and not isUP("finger3") and not isUP("finger2") and not isUP("finger1") and abs(hand["thumb"][3]['x']-hand["finger1"][1]['x'])<0.06):
        return "pinky"
    elif(isUP("finger4") and isUP("thumb") and not isUP("finger3") and not isUP("finger2") and not isUP("finger1")):
        return "trimmer"
    
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

def DistanceBTW(partname):
    return abs(hand[partname][3]['y']-hand[partname][0]['y'])


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

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (imageFace * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
    output = np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))
    return output

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


hand = {"thumb":[],
        "finger1": [],
        "finger2":[],
        "finger3":[],
        "finger4":[],}

def isUP(partname):
    if(hand[partname][3]['y']<hand[partname][2]['y'] and hand[partname][2]['y']<hand[partname][1]['y'] and hand[partname][1]['y']<hand[partname][0]['y']):
        return True
    else:
        return False
def isDown(partname,axis='y'):
    
    if(axis=='y'and hand[partname][1][axis]<hand[partname][2][axis] and hand[partname][1][axis]<hand[partname][3][axis]):
        return True
    if((axis=='x'and hand[partname][1][axis]>hand[partname][2][axis] and hand[partname][1][axis]>hand[partname][3][axis])):
        return True
    else:
        return False

def ArePartsAway(part1,part2):
    distancestart = abs(hand[part1][0]['x'] - hand[part2][0]['x'])
    distanceend = abs(hand[part1][3]['x'] - hand[part2][3]['x'])
    if(distanceend>1.3*distancestart and abs(hand[part1][3]['y']-hand[part1][0]['y'])>0.1):
        return True
    else:
        return False

def PositionGeneration(results):
    #x is o right, 1 left. all values in middle
    #y is 0 up, 1 down, all values in middle
    hand["thumb"]=[]
    hand["finger1"]=[]
    hand["finger2"]=[]
    hand["finger3"]=[]
    hand["finger4"]=[]
    i=0
    while(i<21):
        pos = protobuf_to_dict(results.multi_hand_landmarks[0].landmark[i])
        del pos['z']

        if( 1 <= i <= 4):
            hand["thumb"].append(pos)
        if( 5 <= i <= 8):
            hand["finger1"].append(pos)
        if( 9 <= i <= 12):
            hand["finger2"].append(pos)
        if( 13 <= i <= 16):
            hand["finger3"].append(pos)
        if( 17 <= i <= 20):
            hand["finger4"].append(pos)
        i=i+1
    
    

# the main driver function that will open the camera and detect
def modelDetection():
    # starting the camera
    cam = cv2.VideoCapture(0)
    right_eye = cv2.imread('filters/red_eyes_right.png', cv2.IMREAD_UNCHANGED)
    left_eye = cv2.imread('filters/red_eyes_left.png', cv2.IMREAD_UNCHANGED)
    smoke = cv2.VideoCapture('filters/smoke.gif')
    Hat_filter = cv2.imread('filters/Magic_Hat.png', cv2.IMREAD_UNCHANGED)
    Yoru_filter = cv2.imread('filters/Yoru_Mask.png', cv2.IMREAD_UNCHANGED)
    Mustache_filter = cv2.imread('filters/mustache.png', cv2.IMREAD_UNCHANGED)
    Glass_filter = cv2.imread('filters/thug_glasses.png', cv2.IMREAD_UNCHANGED)
    Mask_filter = cv2.imread('filters/Mask.png', cv2.IMREAD_UNCHANGED)
    smoke_counter = 0
    flag = [False, False, False, False, False, False, False]


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
        mp_hands_results = hands.process(image)
        # coverting image to BGR because CV loves BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # if mp_face_results.multi_face_landmarks:
        #     for face_landmarks in mp_face_results.multi_face_landmarks:
                # drawing face landmarks, tip : FACE_CONNECTIONS changed to FACEMESH_TESSELATION
                # mp_drawing.draw_landmarks(image, face_landmarks, mp_face.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                # mp_drawing.draw_landmarks(image, face_landmarks, mp_face.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

        if mp_face_results.multi_face_landmarks:
            
            mouthStatus = isOpen(image, mp_face_results, 'MOUTH', threshold=15)
            LeyeStatus = isOpen(image, mp_face_results, 'LEFT EYE', threshold=5)
            ReyeStatus = isOpen(image, mp_face_results, 'RIGHT EYE', threshold=5)

            # Iterate over the found faces.
            for face_num, face_landmarks in enumerate(mp_face_results.multi_face_landmarks):
                
                Head_dstMat = GetRectCoords(mp_face_results.multi_face_landmarks, image, "Head")
                Face_dstMat = GetRectCoords(mp_face_results.multi_face_landmarks, image, "Face")
                Mus_dstMat = GetRectCoords(mp_face_results.multi_face_landmarks, image, "Mustache")
                Eye_dstMat = GetRectCoords(mp_face_results.multi_face_landmarks, image, "Eyes")
                Mask_dstMat = GetRectCoords(mp_face_results.multi_face_landmarks, image, "Covid")
                
                if (flag[5]):
                    if ReyeStatus[face_num] == 'OPEN':
                        image = overlay(image, right_eye, face_landmarks, 'RIGHT EYE', mp_face.FACEMESH_RIGHT_EYE)
                    
                    if LeyeStatus[face_num] == 'OPEN':
                        image = overlay(image, left_eye, face_landmarks, 'LEFT EYE', mp_face.FACEMESH_LEFT_EYE)
                
                    if mouthStatus[face_num] == 'OPEN':
                        image = overlay(image, smoke_frame, face_landmarks, 'MOUTH', mp_face.FACEMESH_LIPS)

                

        if mp_hands_results.multi_hand_landmarks:
            
            PositionGeneration(mp_hands_results)
            
            if(GestureChecker() == "thumbsup"):
                flag[0] = True

            elif (GestureChecker() == "thumbsdown"):
                flag[1] = True

            elif (GestureChecker() == "spiderman"):
                flag[2] = True

            elif (GestureChecker() == "handUP_open"):
                flag[3] = True

            elif (GestureChecker() == "handUP_close"):
                flag[4] = True

            elif (GestureChecker() == "gun"):
                flag[5] = True

            elif (GestureChecker() == "Veetit"):
                print("closing all")
                flag = [False, False, False, False, False, False,False]
            elif (GestureChecker() == "trimmer"):
                flag[6] = True

            else:
                pass

            # for hand_landmarks in mp_hands_results.multi_hand_landmarks:
            #     mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

        if (flag[0]):
            image = applyFilter(Hat_filter, image, Head_dstMat)
        
        if (flag[1]):
            image = applyFilter(Glass_filter, image, Eye_dstMat)
        
        if (flag[2]):
            image = applyFilter(Mustache_filter, image, Mus_dstMat)
        
        if (flag[3]):
            image = applyFilter(Yoru_filter, image, Face_dstMat)
        
        if (flag[4]):
            image = applyFilter(Mask_filter, image, Mask_dstMat)
        
        if (flag[6]):
            flag[6]=False
            captured_image = cv2.copyMakeBorder(src=image, top=5, bottom=5, left=5, right=5,
                                                    borderType=cv2.BORDER_CONSTANT)
            cv2.imwrite('Captured_Image.png', captured_image)
            resized = cv2.resize(captured_image, (400, 300)) 
            cv2.imshow("Captured",resized)
            cv2.waitKey(1000)
            cv2.destroyWindow("Captured")
            


        cv2.imshow('Nova Detect', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

modelDetection()