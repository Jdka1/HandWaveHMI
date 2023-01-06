import cv2 as cv
import pyautogui


class Trackpad:
    def __init__(self, x, y, MONITOR_DIM, h):
        self.x = x
        self.y = y
        self.ASPECT_RATIO = MONITOR_DIM[0] / MONITOR_DIM[1]
        self.h = h
        self.w = int(self.h*self.ASPECT_RATIO)
        self.scale_factor = MONITOR_DIM[0] / self.w
        
    def show(self, image):
        return cv.rectangle(image, (self.x, self.y), (self.x+self.w, self.y+self.h), (255,0,0), 5)
    
    
    def map_pos(self, pos):
        pos_on_trackpad = (pos[0] - self.x, pos[1] - self.y)
        print (self.scale_factor)
        return (pos_on_trackpad[0]*self.scale_factor, pos_on_trackpad[1]*self.scale_factor)