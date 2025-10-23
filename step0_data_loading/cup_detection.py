# Install ultralytics if not already installed
# pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO

# Load YOLOv8 pretrained model
model = YOLO("step0_data_loading/best.pt")  # lightweight model; can use yolov8s.pt for more accuracy

# Load image or video frame
image_path = "/home/h2r/Downloads/dog_data/BDL049_Star_side_cam/2/Color/_Color_0713.png"  # replace with your image path
img = cv2.imread(image_path)

# Run detection
results = model(img)

# Parse and visualize detections
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        if label == "cup":  # filter only cup detections
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show result
cv2.imshow("Cup Detection", img)