import cv2
import numpy as np
import time
from ultralytics import YOLO
import requests
import threading
import queue
import sys
import os

# ESP32-CAM URL (Update with your ESP32's IP)
ESP32_STREAM_URL = "http://192.168.251.133:81/stream"

# Configuration
LINE_Y = 300
offset = 10
PIXELS_TO_METERS = 0.05
RECONNECT_TIMEOUT = 30  # seconds
FRAME_QUEUE_SIZE = 10

# Tracking data
vehicles = {}
vehicle_speeds = {}
vehicle_count = 0

# Load YOLOv8 Model
try:
    print("Loading YOLO model...")
    model = YOLO('yolov8s.pt')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    sys.exit(1)

# Frame reading queue for threaded capture
frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
stop_thread = threading.Event()

def create_capture(url, max_retries=3):
    """Create video capture with retry logic"""
    for i in range(max_retries):
        print(f"Connection attempt {i+1}/{max_retries}")
        
        # Try with different capture methods
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        # Set timeout property to avoid hanging
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        
        if cap.isOpened():
            print("Connection established!")
            return cap
        
        print(f"Attempt {i+1} failed. Retrying...")
        time.sleep(1)
    
    return None

def mjpeg_stream_reader():
    """Alternative MJPEG stream reader using requests"""
    print("Starting MJPEG stream reader thread...")
    
    while not stop_thread.is_set():
        try:
            # Use requests to get MJPEG stream
            response = requests.get(ESP32_STREAM_URL, stream=True, timeout=5)
            
            if response.status_code != 200:
                print(f"Error: HTTP status {response.status_code}")
                time.sleep(2)
                continue
                
            bytes_data = b''
            for chunk in response.iter_content(chunk_size=1024):
                if stop_thread.is_set():
                    break
                    
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')  # JPEG start
                b = bytes_data.find(b'\xff\xd9')  # JPEG end
                
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    try:
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None and not frame_queue.full():
                            frame_queue.put(frame, block=False)
                    except Exception as e:
                        print(f"Error decoding frame: {e}")
                        
        except Exception as e:
            print(f"Stream connection error: {e}")
            time.sleep(2)
            
    print("Stream reader thread stopped")

def opencv_stream_reader():
    """OpenCV stream reader thread"""
    print("Starting OpenCV stream reader thread...")
    
    last_reconnect_time = time.time()
    cap = create_capture(ESP32_STREAM_URL)
    
    if cap is None:
        print("Failed to connect using OpenCV. Switching to MJPEG method.")
        return
    
    while not stop_thread.is_set():
        try:
            # Check if we need to reconnect
            current_time = time.time()
            if current_time - last_reconnect_time > RECONNECT_TIMEOUT:
                print("Reconnecting to prevent timeout...")
                cap.release()
                cap = create_capture(ESP32_STREAM_URL)
                if cap is None:
                    print("Reconnection failed. Exiting thread.")
                    break
                last_reconnect_time = current_time
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame. Reconnecting...")
                cap.release()
                cap = create_capture(ESP32_STREAM_URL)
                if cap is None:
                    print("Reconnection failed. Exiting thread.")
                    break
                last_reconnect_time = time.time()
                continue
            
            if not frame_queue.full():
                frame_queue.put(frame, block=False)
                
        except Exception as e:
            print(f"Error in OpenCV reader: {e}")
            time.sleep(1)
    
    if cap is not None:
        cap.release()
    
    print("OpenCV reader thread stopped")

def try_local_video_fallback():
    """Try to use a local video file as fallback"""
    video_paths = [
        "traffic.mp4",
        "videos/traffic.mp4",
        "data/traffic.mp4",
        "sample_video.mp4"
    ]
    
    for path in video_paths:
        if os.path.exists(path):
            print(f"Using local video file: {path}")
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                return cap
    
    # Try webcam as last resort
    print("Trying webcam...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap
    
    return None

def main():
    global vehicle_count
    
    # First try OpenCV method
    reader_thread = threading.Thread(target=opencv_stream_reader)
    reader_thread.daemon = True
    reader_thread.start()
    
    # Give it a moment to connect
    time.sleep(3)
    
    # If queue is still empty, try MJPEG method
    if frame_queue.empty():
        print("OpenCV method not working. Switching to MJPEG stream method...")
        stop_thread.set()
        reader_thread.join(timeout=1)
        
        # Reset and try MJPEG method
        stop_thread.clear()
        reader_thread = threading.Thread(target=mjpeg_stream_reader)
        reader_thread.daemon = True
        reader_thread.start()
        
        time.sleep(3)
    
    # If both methods fail, try local video
    if frame_queue.empty():
        print("Both stream methods failed. Trying local video fallback...")
        stop_thread.set()
        reader_thread.join(timeout=1)
        
        cap = try_local_video_fallback()
        if cap is None:
            print("ERROR: Could not connect to any video source.")
            return
        
        # Create window
        cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
        
        # Process using direct video capture
        process_direct_video(cap)
        return
    
    # Create window
    cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
    
    # Process frames from the queue
    last_frame_time = time.time()
    while True:
        try:
            # Get frame with timeout
            try:
                frame = frame_queue.get(timeout=1)
                last_frame_time = time.time()
            except queue.Empty:
                # Check if we've been waiting too long for a frame
                if time.time() - last_frame_time > 10:
                    print("No frames received for 10 seconds. Exiting...")
                    break
                continue
            
            # Process the frame with YOLO
            process_frame(frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(0.1)
    
    # Clean up
    stop_thread.set()
    if reader_thread.is_alive():
        reader_thread.join(timeout=1)
    cv2.destroyAllWindows()
    print("Program terminated")

def process_direct_video(cap):
    """Process frames directly from a video capture"""
    global vehicle_count
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break
        
        process_frame(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame):
    """Process a single frame with YOLO"""
    global vehicle_count, vehicles, vehicle_speeds
    
    start_time = time.time()
    
    # Get frame dimensions
    h, w, _ = frame.shape
    
    # Run YOLO detection
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy() if results and results[0].boxes is not None else np.array([])
    
    new_vehicles = {}
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf, cls = det[4], int(det[5])
        
        # Class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
        if cls in [2, 3, 5, 7] and conf > 0.4:
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            new_vehicles[(centroid_x, centroid_y)] = (x1, y1, x2, y2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
            
            # Add vehicle class label
            class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
            class_label = class_names.get(cls, "Vehicle")
            conf_text = f"{class_label}: {conf:.2f}"
            cv2.putText(frame, conf_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Check for line crossing
    for (cx, cy), bbox in new_vehicles.items():
        if LINE_Y - offset < cy < LINE_Y + offset:
            if (cx, cy) not in vehicles:
                vehicles[(cx, cy)] = time.time()
                vehicle_count += 1
            else:
                elapsed_time = time.time() - vehicles[(cx, cy)]
                distance_m = PIXELS_TO_METERS * abs(cy - LINE_Y)
                speed = distance_m / elapsed_time * 3.6
                vehicle_speeds[(cx, cy)] = round(speed, 2)
    
    # Draw Line
    cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (255, 0, 0), 2)
    cv2.putText(frame, "Counting Line", (w-150, LINE_Y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Display count and speeds
    cv2.putText(frame, f'Count: {vehicle_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Limit displayed speeds to most recent 5
    recent_speeds = list(vehicle_speeds.items())[-5:] if len(vehicle_speeds) > 5 else vehicle_speeds.items()
    
    y_offset = 80
    for i, ((_, _), speed) in enumerate(recent_speeds):
        cv2.putText(frame, f'Vehicle {i+1}: {speed} km/h', (50, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
    
    # Calculate and display FPS
    process_time = time.time() - start_time
    fps = 1 / process_time if process_time > 0 else 0
    cv2.putText(frame, f'FPS: {fps:.1f}', (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show the frame
    cv2.imshow("Vehicle Detection", frame)

if __name__ == "__main__":
    main()