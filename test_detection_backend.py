"""Test backend DetectionService with video file"""

import sys
import cv2
import json
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.app.services.detect import DetectionService
from backend.app.configuration import DetectionConfig

# Config
VIDEO_PATH = r"src\case\case_03.mp4"
MOTO_MODEL = "yolov8n.pt"
HELMET_MODEL = r"backend\train\best_26.pt"
OUTPUT_FILE = "test_output/backend_detections.json"

print(f"🚀 Testing with: {VIDEO_PATH}")
print(f"Models: {MOTO_MODEL}, {HELMET_MODEL}")

# Initialize service with detection config
detection_config = DetectionConfig(
    pad_filter=80,
    helmet_detect_confidence=0.20,
    helmet_detect_imgsz=1280,
    motorcycle_confidence=0.5,
    line_position_percent=0.5,
)
service = DetectionService(Path(MOTO_MODEL), Path(HELMET_MODEL), detection_config)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Cannot open: {VIDEO_PATH}")
    exit()

# Get video info
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"📊 Video: {total_frames} frames @ {fps:.1f} FPS")

# Process
all_records = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Run detection
    annotated_frame, records = service.detect_and_track(frame)

    # Save records
    for record in records:
        result = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "frame": frame_idx,
            "motorcycle_track_id": record.motorcycle_track_id,
            "helmet_status": record.helmet_status,
            "passenger_count": record.passenger_count,
            "over_capacity": record.over_capacity,
            "violation": record.violation,
            "status": "VIOLATION"
            if record.violation
            else "OK"
            if record.helmet_status
            else "NO DETECT",
        }
        all_records.append(result)

        # Print to console
        print(json.dumps(result, ensure_ascii=False))

    # Display
    cv2.imshow("Detection", annotated_frame)

    # Log progress
    if frame_idx % 100 == 0:
        print(f"📊 Processed {frame_idx}/{total_frames} frames")

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("🛑 User quit")
        break

# Save results
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_records, f, indent=2, ensure_ascii=False)

print(f"\n💾 Saved {len(all_records)} detections to {OUTPUT_FILE}")
print(f"✅ Done - Processed {frame_idx} frames")

# Cleanup
cap.release()
cv2.destroyAllWindows()
service.reset_tracks()
