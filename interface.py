from controller import Controller
from trackpad import Trackpad
import pyautogui

MONITOR_SIZE = pyautogui.size()
trackpad = Trackpad(700,300,(MONITOR_SIZE[0], MONITOR_SIZE[1]),500)

controller = Controller()



max_images_taken = 300
image_size = 300

do_machine_learning = True


import time
import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


# CONSTANTS
labels = ["1 Finger", '4 Fingers', "fist"]
pointer_finger_index = 8


def get_direction(pt1, pt2):
    if pt2[0] > pt1[0]:
        return 'right'
    else:
        return 'left'


def take_action(prev_positions):
    
    # THIS IS LAGGING IT
    
    prev_hands_labels = list(map(lambda x: x['label'], prev_positions))
    prev_hands_points = list(map(lambda x: x['points']['lmList'], prev_positions))
    moved_dist = math.dist((prev_hands_points[0][pointer_finger_index][0], prev_hands_points[0][pointer_finger_index][1]), (prev_hands_points[-1][pointer_finger_index][0], prev_hands_points[-1][pointer_finger_index][1])) 
    
    
    if prev_hands_labels[-1] == '1 Finger': # MOVING MOUSE
        pointer_finger_pos = (int(prev_hands_points[-1][pointer_finger_index][0]), int(prev_hands_points[-1][pointer_finger_index][1]))
        mapped_pointer_finger_pos = trackpad.map_pos(pointer_finger_pos) 
        controller.set_mouse_pos(mapped_pointer_finger_pos)
        print('Moved mouse.')
    
        # retrain 1 finger
        if math.dist( (prev_hands_points[-1][4][0], prev_hands_points[-1][4][1]),  (prev_hands_points[-1][10][0], prev_hands_points[-1][10][1]) ) < 30:
            controller.click()
            print('Click!')
        
        return True
        
    if prev_hands_labels.count('4 Fingers') > 2 and moved_dist > 175: # FOUR FINGER SWIPE
        # prev_swipe_dirs = [get_direction((prev_positions[i]['points']['lmList'][0], prev_positions[i]['points']['lmList'][1]), (prev_positions[i+1]['points']['lmList'][0], prev_positions[i+1]['points']['lmList'][1])) for i in range(len(prev_positions)-1)]
        # prev_swipe_dirs = [get_direction((prev_hands_points[i][0],prev_hands_points[i][1]), (prev_hands_points[i][0],prev_hands_points[i][1])) for i in range(len(prev_positions)-1)]
        swipe_dir = get_direction((prev_hands_points[0][pointer_finger_index][0], prev_hands_points[0][pointer_finger_index][1]), (prev_hands_points[-1][pointer_finger_index][0], prev_hands_points[-1][pointer_finger_index][1]))
        
        controller.four_finger_swipe(swipe_dir)
        
        return True
    
    
    return False





clf = Classifier("Model/keras_model.h5", "Model/labels.txt")

detector = HandDetector(maxHands=2, detectionCon=0.5, minTrackCon=0.5)
capture = cv.VideoCapture(2)

counter = 0


prev_positions = []



while True:
    success, image = capture.read()
    image = cv.flip(image, 1) # Flip so video is a mirror
    original = image.copy()
    
    hands, image = detector.findHands(image)
     
    
    if hands:
        hand = hands[0]
        # limit = lambda num, minn, maxn: max(min(maxn, num), minn)
        x, y, w, h = hand['bbox']
        
        bbox_margin = 30
        
        
        # Cropping only the bbox of the hand
        try:
            cropped = image[y-bbox_margin:y+h+bbox_margin, x-bbox_margin:x+w+bbox_margin]
        except:
            continue
        
        
        # Resizing the cropped image to fill the blank image
        aspect_ratio = w/h
        
        if aspect_ratio > 1: # width > height
            scale_factor = image_size/w
            dimensions = (math.floor(scale_factor*w), math.floor(scale_factor*h))
            try:
                cropped = cv.resize(cropped, dimensions)
            except Exception as e:
                print(str(e))
        elif aspect_ratio < 1: # height > width or width == height
            scale_factor = image_size/h
            dimensions = (math.floor(scale_factor*w), math.floor(scale_factor*h))
            try:
                cropped = cv.resize(cropped, dimensions)
            except Exception as e:
                print(str(e))
        
        
        # Overlaying the resized image on the blank image
        blank = np.ones((image_size, image_size, 3), np.uint8)*255
        try:
            margin_lateral = image_size - cropped.shape[1]
            margin_vertical = image_size - cropped.shape[0]
            blank[math.floor(margin_vertical/2):cropped.shape[0]+math.floor(margin_vertical/2), math.floor(margin_lateral/2):cropped.shape[1]+math.floor(margin_lateral/2)] = cropped
        except Exception as e:
            print(str(e))
        
        index = 0
        if do_machine_learning:
            prediction, index = clf.getPrediction(blank)
        
        prev_positions.append({'label': labels[index],
                               'points': hand})
            
        
    if hands:
        x = x-bbox_margin
        y = y-bbox_margin
        w = w+(bbox_margin*2)
        h = h+(bbox_margin*2)
        original = cv.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 3)
        original = cv.putText(original, labels[index], (x,y), cv.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
        
        original = cv.circle(original, (hand['lmList'][pointer_finger_index][0], hand['lmList'][pointer_finger_index][1]), 8, (255,0,0), -1)
        
        # print(math.dist( (hand['lmList'][4][0], hand['lmList'][4][1]),  (hand['lmList'][5][0], hand['lmList'][5][1]) ))
        # original = cv.circle(original, (hand['lmList'][4][0], hand['lmList'][4][1]), 8, (255,0,0), -1)
        # original = cv.circle(original, (hand['lmList'][5][0], hand['lmList'][5][1]), 8, (255,0,0), -1)
        
        
        # clear prev positions every once in a while
        if take_action(prev_positions):
            prev_positions = []
            print('Reset prev_positions')
        elif len(prev_positions) > 2:
            prev_positions = []
            print('Reset prev_positions')
    else:
        prev_positions = []
        print('Reset prev_positions')
    
    
    original = trackpad.show(original)
    
    cv.imshow("Image", original)

    pressed_key = cv.waitKey(1) 
    
    if pressed_key == ord('s'):
        break
        
    
capture.release()
cv.destroyAllWindows()