import cv2
import pyttsx3
from ultralytics import YOLO

def cameracapture():
  cap=cv2.VideoCapture(0)
  count=0
  while True:
    ret,frame=cap.read()
    if not ret: 
      break
    count += 1
    if count % 3 != 0:  
      continue
    frame=cv2.resize(frame,(1020,600))

    # Add the following code to increase the frame rate while displaying the output
    cv2.imshow("FRAME",frame)
    detect_objects_and_speak(frame)
    if cv2.waitKey(1) == 27:
      break

model = YOLO("best.pt")
def detect_objects_and_speak(image):
    detections = model(image)
    print(detections)

    # Loop over the detections and generate a voice output for each object
    for detection in detections:
        # print(detection)
        classes = {0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask', 4: 'NO-Safety Vest',
                   5: 'Person', 6: 'Safety Cone', 7: 'Safety Vest', 8: 'machinery', 9: 'vehicle'}
        detectionss = detection.boxes[detection.boxes.conf >= 0.3]
        class_name = detectionss.cls
        print(detectionss.cls.tolist())
        temp = detectionss.cls.tolist()
        for i in range(len(temp)):
            temp[i] = int(temp[i])
        ot = []
        for i in range(len(temp)):
            ot.append(classes.get(temp[i]))
        print(ot)
        l = []
        pred = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']
        for i in range(len(ot)):
            if (ot[i] == 'NO-Hardhat' or ot[i] == 'NO-Mask' or ot[i] == 'NO-Safety Vest'):
                l.append(ot[i])
        print(l)
        confidence = detection.boxes.conf
        x1, y1, x2, y2 = detection.boxes.xyxy[0]
        # Generate a voice output describing the object
        for i in range(len(l)):
            if (l[i] == 'NO-Hardhat'):
                voice_output = f"Detected a {l[i]}, please wear hardhat."
            elif (l[i] == 'NO-Mask'):
                voice_output = f"Detected a {l[i]}, please wear mask."
            if (l[i] == 'NO-Safety Vest'):
                voice_output = f"Detected a {l[i]}, please wear Safety Vest."
            print(confidence)
            # Speak the voice output
            engine = pyttsx3.init()
            engine.say(voice_output)
            engine.runAndWait()


image = cv2.imread("img5.jpg")
# cv2.imshow("frame",model(image))
detect_objects_and_speak(image)
