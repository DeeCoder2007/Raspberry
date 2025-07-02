import argparse
import time
import torch
import cv2
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLOv5 weights')
    parser.add_argument('--source', type=str, default='0', help='Source: 0=webcam, 1=PiCam (if using), or path to video file/URL')
    parser.add_argument('--img-size', type=int, default=640, help='Inference image size (e.g., 320, 640)')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from {args.weights}...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.weights, force_reload=True)
    model.to(device).eval()

    cap = cv2.VideoCapture(0 if args.source == '0' else args.source)
    if not cap.isOpened():
        print(f"Error opening video stream or file: {args.source}")
        return

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize if required by source
        frame_resized = cv2.resize(frame, (args.img_size, args.img_size))

        # Inference
        results = model(frame_resized)

        # Parse results
        detections = results.xyxy[0]  # (x1, y1, x2, y2, conf, cls)
        count = len(detections)

        # Draw boxes
        for *xyxy, conf, cls in detections:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        # Overlay FPS and count
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Objects: {count}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("YOLOv5 Live Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
