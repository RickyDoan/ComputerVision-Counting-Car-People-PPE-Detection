from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0) # Detect webcam
# cap.set(3, 1280)
# cap.set(4, 640)

cap = cv2.VideoCapture("../2 - Running Yolo with Webcam/videos/ppe-2-1.mp4") # Detect video

model = YOLO("model/ppe.pt")

# Define output video parameters
frame_width = int(cap.get(3))  # Get video width
frame_height = int(cap.get(4))  # Get video height
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second

# Define video writer (Codec: XVID or MP4V for .mp4 output)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change to 'mp4v' if using .mp4
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# classNames = ["boots", "gloves", "helmet", "human", "vest"]
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

while True:
    success, image = cap.read()
    result = model(image)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            (x1, y1, x2, y2) = box.xyxy[0]
            (x1, y1, x2, y2) = int(x1), int(y1), int(x2), int(y2)

            # print(x1, y1, x2, y2)
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Confidence
            w,h = (x2-x1), (y2-y1)
            cvzone.cornerRect(image, (x1,y1,w,h))

            conf = math.ceil((box.conf[0]*100))/100
            print(conf)

            # Class name
            cls = int(box.cls[0])

            class_name_current = classNames[cls]
            print(class_name_current)
            if conf > 0.5:
                if class_name_current == 'NO-Hardhat' or class_name_current == 'NO-Mask' or class_name_current == 'NO-Safety Vest' :
                    text_color = (0, 255, 0)
                    box_color = (0, 0, 255)
                else :
                    text_color = (0, 255, 0)
                    box_color = (0, 128, 0)

                cvzone.putTextRect(image, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=1,
                                   thickness=1, colorT=text_color, colorR=box_color)
                cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

    if not success:
        print("Ignoring empty camera frame or reconnecting your camera.")
        continue
      
    # Write frame to output video
    out.write(image)
  
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
