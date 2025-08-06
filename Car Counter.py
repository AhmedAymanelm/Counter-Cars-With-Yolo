import cvzone
import numpy as np
from ultralytics import YOLO
import cv2
import math
from sort import *

cap = cv2.VideoCapture("fast motion cars moving on highway.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 )
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Load and prepare the mask
mask = cv2.imread("Untitled design.png")
mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)


tracking = Sort(max_age=20 , min_hits=3, iou_threshold=0.3)
limits = [300,400, 1100, 400]

counttrack = []
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detected = np.empty((0, 5))

    # Resize mask to match current frame
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    imgeregion = cv2.bitwise_and(frame, mask_resized)

    Imgegraphic = cv2.imread("jpeg.png", cv2.IMREAD_UNCHANGED)
    Imgegraphic = cv2.resize(Imgegraphic, (0, 0), fx=0.5, fy=0.5)

    print(Imgegraphic.shape)
    frame = cvzone.overlayPNG(frame,Imgegraphic,(-80,-30))


    results = model(imgeregion, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255),2)

            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            CurrentClass = classNames[cls]

            if (CurrentClass == "car" or CurrentClass == "truck")  and conf > 0.3:
               # cvzone.putTextRect(frame, f' {CurrentClass}  {conf}', (max(0,x1), max(36,y1)))
               cvzone.cornerRect(frame, (x1, y1, w, h), l=4)
               currentarry =  np.array([x1, y1, x2, y2, conf])
               detected = np.vstack((detected, currentarry))



        resulttracker = tracking.update(detected)
        cv2.line(frame, (limits[0], limits[1]), (limits[2],limits[3]) , (0, 0, 255), 2)

        for result in resulttracker:
            x1, y1, x2, y2, Id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(frame, (x1, y1, w, h), l = 9, colorR=(0, 0, 255),rt=3)
            cvzone.putTextRect(frame, f"{int(Id)}", (max(0,x1), max(36,y1)),scale=2,thickness=3,offset=10)

            cx, cy  = x1+w //2 , y1+h//2
            cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), cv2.FILLED)

            if limits[1] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15 :
                if counttrack.count(Id) == 0:
                   counttrack.append(Id)
                   cv2.line(frame, (limits[0], limits[1]), (limits[2],limits[3]) , (0, 255, 0), 2)

    # cvzone.putTextRect(frame, f"count {len(counttrack)}",(50,50),scale=2,thickness=3,offset=10)
    cv2.putText(frame, str(len(counttrack)), (255,110),cv2.FONT_HERSHEY_PLAIN,9,(0,0,255),4)


    cv2.imshow('Imshowframe', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
