# Hand-Gesture-Computer-Interface

An interface for interacting with your computer using hand gestures 👋 💻

## Installation

```bash
git clone https://github.com/Jdka1/Hand-Gesture-Computer-Interface.git
cd Hand-Gesture-Computer-Interface
python3 interface.py
```

### Dependancies

This project was built using tensorflow for object detection to classify gestures, and pyautogui for mapping the gestures to keyboard and mouse actions.

## Actions
- Four Finger Swipe: Swipe across live feed with an opened palm and fingers together ✋
- Move Mouse: Move hand around with your index finger and thumb out 👆
- Click: While your hand is in the 'Move Mouse' position, move your thumb towards your index finger to click 🖱️

Due to the speed of the object detection being slow, the mouse moving capapilities are slightly limited.

## Training

The training of this model is done through opencv's hand detection software to crop the area of the hand to standardize the training images. The cropped image is then placed on a white background so different resolutions can all be trained on. The model is trained on 300 images of each gesture for consistency.
