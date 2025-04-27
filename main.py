import os
import cv2
import time
from config import *
from detector import ObjectDetect
from visualizer import DetectionVisualizer

def main():
    # Initialize
    os.makedirs("helmet_off", exist_ok=True)
    detector = ObjectDetect(HELMET_MODEL_PATH, PERSON_MODEL_PATH)
    visualizer = DetectionVisualizer()
    
    # Open video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    window_name = 'Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    frame_count = 0
    last_save_time = 0
    save_interval = 1.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Detect objects
            results_helmet, results_person = detector.detect(
                frame, 
                HELMET_CONF_THRESHOLD, 
                PERSON_CONF_THRESHOLD
            )
            
            # Visualize results
            frame, has_no_helmet = visualizer.draw_detections(
                frame, 
                results_helmet, 
                results_person, 
                ROI_POINTS
            )
            
            # บันทึกเฟรมเมื่อพบคนไม่สวมหมวก
            current_time = time.time()
            if has_no_helmet and (current_time - last_save_time) >= save_interval:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join("helmet_off", f"violation_{timestamp}_{frame_count}.jpg")
                cv2.imwrite(save_path, frame)
                last_save_time = current_time
            
            # Display results
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
