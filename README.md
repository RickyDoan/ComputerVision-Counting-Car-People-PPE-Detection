# Computer Vision Projects: Car Counter, People Counter, and PPE Detection

## Overview

### This repository demonstrates how to build and implement three computer vision applications using OpenCV and deep learning techniques:

* Car Counter - Detects and counts vehicles in a video stream using object detection.

* People Counter - Tracks and counts people appearing in a video frame.

* PPE Detection - Identifies whether individuals are wearing Personal Protective Equipment (PPE) such as helmets and vests.

Implementation Details

### 1. Car Counter
![](https://github.com/RickyDoan/ComputerVision-Counting-Car-People-PPE-Detection/blob/main/Car%20Counter/counting_car.gif)
- Concept :Vehicle detection and tracking is crucial in traffic management and smart surveillance systems. This project leverages deep learning-based object detection to track and count vehicles moving in a predefined region of interest.

- Preprocessing: The input video frame is read and preprocessed for better detection accuracy.

- Object Detection: A deep learning model (e.g., YOLO, SSD, or MobileNet) is used to detect vehicles.

- Tracking and Counting: Detected vehicles are assigned unique IDs and counted when they cross a specific region (e.g., a virtual line on the road).

- Output Processing: The count and tracking results are overlaid on the video output.

### 2. People Counter
![](https://github.com/RickyDoan/ComputerVision-Counting-Car-People-PPE-Detection/blob/main/People%20counter/pp_counting.gif)
- Concept : People counting is widely used in crowd monitoring, retail analytics, and security surveillance. This project utilizes deep learning-based object detection models to track individuals in a scene.

- Frame Processing: Video frames are loaded and processed to improve detection.

- Detection Model: A pre-trained deep learning model (e.g., YOLO, Faster R-CNN) is used to detect people in each frame.

- Tracking Mechanism: A tracking algorithm maintains the identity of individuals across frames to avoid double counting.

- Counting Logic: Individuals are counted when they pass a certain boundary or threshold.

### 3. PPE Detection
![](https://github.com/RickyDoan/ComputerVision-Counting-Car-People-PPE-Detection/blob/main/PPE%20detection/ppe_detection.gif)
### 🚀 Ensuring Workplace Safety with AI-Powered PPE Detection! 🦺👷‍♂️
* This computervision project which help to improve the safety workplace when staffs entering the construction spot.
* I built a PPE Detection System using deep learning and computer vision to identify helmets, vests, and other protective gear in real time. If someone did not protect thereself by gears, it may detect then create the signal to warning.
- 🔍 How It Works:
- ✅ I was using pre-trained Models (YOLO, TensorFlow) to recognize PPE on the Roboflow datasets.
- ✅ Helped to automated Compliance Checks – Detects violations and flags missing safety gear.
- ✅ Scalable for Real-World Use – This can be deployed on cameras in construction sites, factories, and warehouses.
- 💡 What's the goal? --> It helps to reduce workplace risks, ensure safety, and drive compliance using AI-powered automation.

* Conclusion

These projects showcase the power of computer vision in solving real-world challenges. By leveraging deep learning and object detection, we can build intelligent systems for automation, safety, and analytics.
