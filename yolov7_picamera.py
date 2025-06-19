from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
import torch
import cv2
import numpy as np
from picamera2 import Picamera2
from time import time

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model directly
weights = 'yolov7-tiny.pt'
model = attempt_load(weights, map_location=device)
model.eval()

# Get class names
names = model.module.names if hasattr(model, 'module') else model.names

# Initialize Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# Inference loop
try:
    while True:
        start = time()
        frame = picam2.capture_array()
        img0 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Preprocess
        img = letterbox(img0, new_shape=640)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)

        num_objects = 0

        # Draw results
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                num_objects = len(det)
                for *xyxy, conf, cls in det:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        fps = 1 / (time() - start)
        stats_text = f"FPS: {fps:.2f} | Objects: {num_objects}"
        cv2.putText(img0, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)

        cv2.imshow("YOLOv7 Detection", img0)
        if cv2.waitKey(1) == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
