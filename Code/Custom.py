from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Load video
video_path = r"C:\Users\Soham Shenoy\Downloads\WhatsApp Video 2025-03-09 at 4.25.42 PM.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video. Check the file path!")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writer for saving output
output_path = "output.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Define COCO class IDs for vehicles (car=2, bus=5, truck=7)
vehicle_classes = [2, 5, 7]

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video ends

    # Perform YOLOv8 inference
    results = model(frame)[0]  

    # Process detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0].item())  # Class ID

        if cls in vehicle_classes and conf > 0.5:  # Filter vehicles
            label = f"{model.names[cls]} ({conf:.2f})"  

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Car Detection", frame)

    # Save frame to output video
    out.write(frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()