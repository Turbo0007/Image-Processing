from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("Could not open the webcam. Please check your setup.")
    exit()

model = YOLO("yolo-Weights/yolov8n.pt")

class_names = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
    "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair dryer", "toothbrush"
]

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab a frame from the webcam. Exiting...")
        break

    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            confidence = round(float(box.conf[0]), 2)
            print(f"Detected object with confidence: {confidence}")

            cls = int(box.cls[0])
            class_name = class_names[cls]
            print(f"Class detected: {class_name}")

            label = f"{class_name} {confidence}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 1
            text_color = (255, 255, 255)
            background_color = (0, 0, 0)

            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            text_x, text_y = x1, max(y1 - 10, text_height + 10)

            cv2.rectangle(frame, (text_x, text_y - text_height - 4), (text_x + text_width, text_y + baseline - 4), background_color, -1)
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, text_thickness)

    cv2.imshow("YOLO Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam and resources released. Goodbye!")
