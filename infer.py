import argparse
import torch
import cv2
from picamera2 import Picamera2
from time import time, sleep
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 Inference on Raspberry Pi with PiCamera")
    parser.add_argument('--weights', type=str, required=True, help='Path to .pt weight file')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.weights, force_reload=False)
    model.conf = args.conf
    model.iou = 0.45
    model.classes = None

    # Initialize PiCamera2
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (args.imgsz, args.imgsz)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()
    sleep(2)

    print("[INFO] Starting inference. Press Ctrl+C to stop.")
    prev_time = time()

    try:
        while True:
            frame = picam2.capture_array()
            start_time = time()
            
            # Inference
            results = model(frame)
            detections = results.xyxy[0]  # detections tensor [x1, y1, x2, y2, conf, cls]
            object_count = len(detections)

            # Render results on frame
            results.render()  # updates results.imgs[0]
            annotated_frame = results.imgs[0]

            # Calculate FPS
            end_time = time()
            fps = 1.0 / (end_time - start_time)

            # Overlay FPS and object count
            cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Objects: {object_count}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Show result
            cv2.imshow("YOLOv5 Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[INFO] Inference stopped by user.")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
