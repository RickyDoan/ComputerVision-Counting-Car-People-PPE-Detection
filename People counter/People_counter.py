import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# cap = cv2.VideoCapture(0) # Detect webcam
# cap.set(3, 1280)
# cap.set(4, 640)

cap = cv2.VideoCapture('../2 - Running Yolo with Webcam/videos/people.mp4')  # Detect video
# cap = cv2.VideoCapture("car_highway_2.mp4")

model = YOLO("../2 - Running Yolo with Webcam/videos/yolov8n.pt")

# Define output video parameters
frame_width = int(cap.get(3))  # Get video width
frame_height = int(cap.get(4))  # Get video height
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second

# Define video writer (Codec: XVID or MP4V for .mp4 output)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change to 'mp4v' if using .mp4
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

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

mask = cv2.imread("mask.png")
# mask = cv2.imread('area_detection_2.png')

# Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# Create limits for red line 1
limits1 = [180, 320, 350, 320]
# Create limits for red line 2
limits2 = [450, 320, 700, 320]

# Get count
total_count_up = []
total_count_down = []

while True:
    success, image = cap.read()
    imageRegion = cv2.bitwise_and(image, mask)
    # Create car graphic instead of count text
    graphic = cv2.imread("counting_up_down.png", cv2.IMREAD_UNCHANGED)
    image = cvzone.overlayPNG(image, graphic, (730, 260))
    result = model(imageRegion, stream=True)
    # Get detection from dets
    detections = np.empty((0, 5))

    for r in result:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            (x1, y1, x2, y2) = box.xyxy[0]
            (x1, y1, x2, y2) = int(x1), int(y1), int(x2), int(y2)

            # print(x1, y1, x2, y2)
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Confidence
            w, h = (x2 - x1), (y2 - y1)
            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)

            # Class name
            cls = int(box.cls[0])
            # cvzone.putTextRect(image, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=0.9, thickness=1,
            #                    offset=3)

            vehicle_classes_name = classNames[cls]
            if vehicle_classes_name == 'person' and conf > 0.3:
                # cvzone.cornerRect(image, (x1,y1,w,h))
                # cvzone.cornerRect(image, (x1, y1, w, h), l=9)
                # For tracking
                numpyArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, numpyArray))
    if not success:
        print("Ignoring empty camera frame or reconnecting your camera.")
        continue

    # cv2.imshow("ImageRegion", imageRegion)

    resultsTracker = tracker.update(detections)
    # Detect the red line
    line1 = cv2.line(image, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 0, 255), 3)
    line2 = cv2.line(image, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 0, 255), 3)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        (x1, y1, x2, y2) = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = (x2 - x1), (y2 - y1)
        cvzone.cornerRect(image, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(image, f" {int(id)}", (max(0, x1), max(35, y1)), scale=0.9, thickness=2,
                           offset=5)
        # Create the circle center on objects
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Count up
        if limits1[0] < cx < limits1[2] and limits1[1] - 15 < cy < limits1[1] + 15:
            # Condition when get id already detect or not, if not == 0, do append
            if total_count_up.count(id) == 0:
                total_count_up.append(id)
                # Make the greenline when counting
                line1 = cv2.line(image, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 255, 0), 3)

        # Count down
        if limits2[0] < cx < limits2[2] and limits2[1] - 15 < cy < limits2[1] + 15:
            # Condition when get id already detect or not, if not == 0, do append
            if total_count_down.count(id) == 0:
                total_count_down.append(id)
                # Make the greenline when counting
                line2 = cv2.line(image, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 255, 0), 3)

    # Get text count
    # cvzone.putTextRect(image, f" Counting : {len(total_count)}", (70, 70), scale=2.5, thickness=3)
    cv2.putText(image, str(len(total_count_up)), (929,345), cv2.FONT_HERSHEY_PLAIN, 5, (139,195,75), 5 )
    cv2.putText(image, str(len(total_count_down)), (1191,345), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255), 5 )

    # Write frame to output video
    out.write(image)

    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
