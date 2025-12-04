import json
import os

import cv2
import numpy as np

roi_points = []
selecting_roi = True


def draw_roi(event, x, y, flags, param):
    """Mouse callback to collect polygon points. Left click to add, right click to finish.

    When right-clicking, at least 3 points are required to close the polygon.
    """
    global roi_points, selecting_roi

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN and len(roi_points) > 2:
        selecting_roi = False


# Resolve path relative to this file so the script works when launched from other CWDs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_REL_PATH = os.path.join("..", "..", "src", "case", "case_11.mp4")
VIDEO_PATH = os.path.normpath(os.path.join(BASE_DIR, VIDEO_REL_PATH))

if not os.path.exists(VIDEO_PATH):
    # Fallback to original relative path (maintain backwards compatibility)
    VIDEO_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "case", "case_03.mp4"))

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Unable to open video file: {VIDEO_PATH}")

ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    raise RuntimeError(f"Failed to read first frame from video: {VIDEO_PATH}")

WINDOW_NAME = "Select ROI"
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, draw_roi)
cv2.resizeWindow(WINDOW_NAME, 940, 680)

# instructions = (
#     "Left-click to add points.\n"
#     "Right-click (when >=3 points) to finish.\n"
#     "Press 'q' to quit without selecting."
# )
# print(instructions)

while selecting_roi:
    # frame is guaranteed non-None due to the checks above; copy once per loop to draw
    temp_frame = frame.copy()

    # draw connecting lines between chosen points
    for i in range(1, len(roi_points)):
        cv2.line(temp_frame, roi_points[i - 1], roi_points[i], (0, 255, 0), 2)

    # if polygon has at least 3 points, draw it filled (or outline)
    if len(roi_points) > 2:
        cv2.polylines(
            temp_frame,
            [np.array(roi_points)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

    cv2.imshow("Select ROI", temp_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
# Convert tuples to list-of-lists with ints, print compact JSON, and save to file silently
roi_list = [[int(x), int(y)] for (x, y) in roi_points]
print(json.dumps(roi_list, separators=(",", ":")))

# Save to a file next to this script (no extra console output)
OUT_PATH = os.path.normpath(os.path.join(BASE_DIR, "roi_points.json"))
try:
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(roi_list, f, ensure_ascii=False)
except Exception:
    # Keep silent on save errors to minimize console output; you can log or raise if desired
    pass
