import pyautogui


class Controller:
    def __init__(self):
        pass
    
    def four_finger_swipe(self, direction):
        if direction not in ['left', 'right']:
            return
        pyautogui.keyDown('ctrl')
        pyautogui.press(direction)
        pyautogui.keyUp('ctrl')

    def set_mouse_pos(self, coord):
        pyautogui.moveTo(coord[0], coord[1])
        
    def click(self):
        pyautogui.click()

