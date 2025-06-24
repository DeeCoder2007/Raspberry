import time
import cv2
import torch
from picamera2 import Picamera2
from ultralytics import YOLO

# Load YOLOv10n model
model = YOLO("yolov10n.pt")

# Initialize PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Allow camera to warm up
time.sleep(2)

# For FPS calculation
prev_time = time.time()

while True:
    frame = picam2.capture_array()
    
    # Inference
    results = model(frame, imgsz=640, conf=0.4)[0]

    # Get detections
    boxes = results.boxes
    num_objects = len(boxes)

    # Draw boxes and labels
    annotated_frame = frame.copy()
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls = int(box.cls)
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Overlay FPS and object count
    overlay_text = f"FPS: {fps:.2f} | Objects: {num_objects}"
    cv2.putText(annotated_frame, overlay_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("YOLOv10n Live Detection", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
picam2.close()
