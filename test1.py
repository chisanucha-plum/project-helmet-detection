import cv2
from ultralytics import YOLO
import json


# ─── config ───────────────────────────────────────────────────────────────────
VIDEO_PATH   = r"src\case\case_03.mp4"
MOTO_MODEL   = "yolov8n.pt"
HELMET_MODEL = r"backend\train\best_26.pt"
PAD_FILTER   = 80
MAX_PASSENGERS = 2

# ─── models ───────────────────────────────────────────────────────────────────
moto_model   = YOLO(MOTO_MODEL)
helmet_model = YOLO(HELMET_MODEL)

# ─── state ────────────────────────────────────────────────────────────────────
track_history = {}   
counted_ids   = set()

# ─── video ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
line_x = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]
    
    # Set LINE_X dynamically on first frame (50% of frame width)
    if line_x is None:
        line_x = w // 2

    # Draw detection line (blue, semi-transparent)
    overlay = frame.copy()
    cv2.line(overlay, (line_x, 0), (line_x, h), (255, 100, 0), 3)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    result = moto_model.track(frame, persist=True, tracker="bytetrack.yaml",
                               classes=[3], conf=0.5, verbose=False)[0]

    if result.boxes is None:
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
        continue

    for box in result.boxes:
        if box.id is None: continue

        tid          = int(box.id.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx           = int((x1 + x2) / 2)   

        # Draw motorcycle bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Check line crossing (right → left)
        prev_cx = track_history.get(tid)
        track_history[tid] = cx             

        if not (prev_cx is not None and prev_cx > line_x and cx <= line_x and tid not in counted_ids):
            continue
            
        counted_ids.add(tid)

        # Detect helmets on full frame
        hdet = helmet_model(frame, conf=0.20, imgsz=1280, verbose=False)[0]
        helmet_labels = []

        if hdet.boxes:
            for hbox in hdet.boxes:
                hx1, hy1, hx2, hy2 = map(int, hbox.xyxy[0].tolist())
                hcx = int((hx1 + hx2) / 2)
                hcy = int((hy1 + hy2) / 2)

                # Filter helmets near motorcycle (PAD_FILTER boundary)
                if (x1 - PAD_FILTER) <= hcx <= (x2 + PAD_FILTER) and \
                   (y1 - PAD_FILTER) <= hcy <= (y2 + PAD_FILTER):
                    label = hdet.names[int(hbox.cls.item())]
                    helmet_labels.append(label)
                    
                    # Draw helmet bounding box with label
                    color = (0, 255, 0) if label == "helmet" else (0, 0, 255)
                    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
                    cv2.putText(frame, label, (hx1, hy1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Determine helmet status and violation
        helmet_status = all(l == "helmet" for l in helmet_labels) if helmet_labels else False
        passenger_count = len(helmet_labels)
        over_capacity = passenger_count > MAX_PASSENGERS
        violation = not helmet_status or over_capacity

        record = {
            "motorcycle_track_id": tid,
            "helmet_status": helmet_status,
            "passenger_count": passenger_count,
            "over_capacity": over_capacity,
            "violation": violation,
        }
        print(json.dumps(record, ensure_ascii=False))

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()