import cv2
import mediapipe as mp
from protobuf_to_dict import protobuf_to_dict
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hand = {"thumb":[],
        "finger1": [],
        "finger2":[],
        "finger3":[],
        "finger4":[],}

def isUP(partname):
    if(hand[partname][3]['y']<hand[partname][2]['y'] and hand[partname][2]['y']<hand[partname][1]['y'] and hand[partname][1]['y']<hand[partname][0]['y']):
        return True
    
def PositionGeneration(results):
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
    
    if(isUP("finger1") and isUP("finger2") and not isUP("finger3") and not isUP("finger4") ):
        print("UP")
    else:
        print("gesture does not meet")
    #x is o right, 1 left. all values in middle
    #y is 0 up, 1 down, all values in middle
    #z is irrelevant



def StartDetection():
# For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            
    # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                PositionGeneration(results)                
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    
    # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()

StartDetection()