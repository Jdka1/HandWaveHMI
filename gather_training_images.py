# Hold "t" to train
save_folder_path = r"data/click/"
max_images_taken = 300
image_size = 300


# Gathering hand images for model training

import time
import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

detector = HandDetector(maxHands=2, detectionCon=0.5, minTrackCon=0.5)
capture = cv.VideoCapture(2)

counter = 0

while True:
    success, image = capture.read()
    image = cv.flip(image, 1) # Flip so video is a mirror
    
    hands, image = detector.findHands(image)
     
    
    for hand in hands:
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
        
        cv.imshow("Cropped", blank)
            
        
            
    cv.imshow("Image", image)

    pressed_key = cv.waitKey(1) 
    
    if pressed_key == ord('s'):
        break
    if pressed_key == ord('t'):
        cv.imwrite(f"{save_folder_path}Image_at_{time.time()}.jpg", blank)
        counter += 1
        print(counter)
        if counter >= max_images_taken:
            print('\n\nTaken max amount of images')
            break
        
    
capture.release()
cv.destroyAllWindows()