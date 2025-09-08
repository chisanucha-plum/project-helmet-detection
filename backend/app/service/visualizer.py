from datetime import datetime

import cv2
import numpy as np
from app.configuration import Configuration


class DetectionVisualizer:
    """
    Class for drawing detection results (persons and helmets) onto video frames
    and checking various conditions such as ROI and object size.
    """

    def __init__(self):
        config = Configuration.get_config()
        # Define colors in BGR (Blue, Green, Red) format
        self.colors = {
            "person": config.detection_visualizer.colors.person,
            "helmet_on": config.detection_visualizer.colors.helmet_on,
            "helmet_off": config.detection_visualizer.colors.helmet_off,
            "roi": config.detection_visualizer.colors.roi,
        }
        # Define constants for person size ratio
        self.min_person_height_width_ratio = (
            config.detection_visualizer.person_validation.min_height_width_ratio
        )

        self.timestamp_font = config.detection_visualizer.timestamp_settings.font
        self.timestamp_scale = config.detection_visualizer.timestamp_settings.scale
        self.timestamp_color = config.detection_visualizer.timestamp_settings.color
        self.timestamp_thickness = (
            config.detection_visualizer.timestamp_settings.thickness
        )

        self.detection_font = cv2.FONT_HERSHEY_SIMPLEX
        self.detection_scale = config.detection_visualizer.detection_settings.scale
        self.detection_thickness = (
            config.detection_visualizer.detection_settings.thickness
        )

    def is_in_roi(self, point, roi_points):
        """
        Checks if a point (bounding box center) is within the Region of Interest (ROI).
        If no ROI is defined, it's always considered within the ROI.
        """
        if not roi_points:
            return True

        point_int = (int(point[0]), int(point[1]))
        return cv2.pointPolygonTest(np.array(roi_points), point_int, False) >= 0

    def is_valid_person_size(self, width, height):
        """
        Checks if the bounding box size is appropriate for a person,
        based on height-to-width ratio. Prevents division by zero.
        """
        if width <= 0:
            return False
        ratio = height / width
        return ratio >= self.min_person_height_width_ratio

    def draw_detections(self, frame, results_helmet, results_person, roi_points):
        """
        Draws detection results (persons, helmets) and ROI onto the frame.
        """
        found_person_no_helmet_in_roi = False

        if roi_points:
            roi_np = np.array(roi_points, np.int32)
            cv2.polylines(frame, [roi_np], True, self.colors["roi"], 2)

        if (
            hasattr(results_person, "boxes")
            and results_person.boxes is not None
            and results_person.boxes.data is not None
        ):
            for box in results_person.boxes.data:
                if len(box) < 6:
                    continue

                x1, y1, x2, y2 = map(int, box[:4])

                width = x2 - x1
                height = y2 - y1

                if (
                    width > 0
                    and height > 0
                    and self.is_valid_person_size(width, height)
                ):
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    if self.is_in_roi((center_x, center_y), roi_points):
                        cv2.rectangle(
                            frame, (x1, y1), (x2, y2), self.colors["person"], 2
                        )

        if (
            hasattr(results_helmet, "boxes")
            and results_helmet.boxes is not None
            and results_helmet.boxes.data is not None
        ):
            for box in results_helmet.boxes.data:
                if len(box) < 6:
                    continue

                x1, y1, x2, y2, conf, cls = box[:6]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                if self.is_in_roi((center_x, center_y), roi_points):
                    is_no_helmet = int(cls) == 0  # Class 0 usually means 'no_helmet'

                    color = (
                        self.colors["helmet_off"]
                        if is_no_helmet
                        else self.colors["helmet_on"]
                    )
                    label = "No Helmet" if is_no_helmet else "Helmet"

                    if is_no_helmet:
                        found_person_no_helmet_in_roi = True

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), color, self.detection_thickness
                    )

                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),  # Text position
                        self.detection_font,
                        self.detection_scale,
                        color,
                        self.detection_thickness,
                    )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame,
            timestamp,
            (10, 30),
            self.timestamp_font,
            self.timestamp_scale,
            self.timestamp_color,
            self.timestamp_thickness,
        )

        # Return the drawn frame and the status of finding a person without a helmet in ROI
        return frame, found_person_no_helmet_in_roi
