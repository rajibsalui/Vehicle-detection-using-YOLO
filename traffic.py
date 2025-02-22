from ultralytics import YOLO
import torch
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"  # It is used to check if CUDA is available
print(f"Using device: {device}")

model = YOLO("yolov8n.pt").to(device) # It is used to load the YOLOv8 model

cap = cv2.VideoCapture("traffic.mp4") # It is used to read the video file

frame_width = 640
frame_height = 480

while cap.isOpened():
    ret, frame = cap.read() # It is used to read the frame from the video
    if not ret: # It is used to check if the frame is empty
        break

    
    frame = cv2.resize(frame, (frame_width, frame_height))  # It is used to resize the frame

    results = model(frame, device=device)  # It is used to detect the vehicles in the frame
    vehicle_count = len(results[0].boxes)  # It is used to count the number of vehicles detected in the frame

    # It is used to draw the bounding boxes on the detected vehicles
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # It is used to display the vehicle count on the frame
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Traffic Detection", frame)  # It is used to display the frame
    if cv2.waitKey(1) & 0xFF == ord("q"):  # It is used to exit the loop when "q" is pressed
        break

cap.release()  # It is used to release the video file
cv2.destroyAllWindows()  # It is used to close all OpenCV windows
