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
        self.colors = {
            "motorcycle": config.detection_visualizer.colors.motorcycle,
            "helmet_on": config.detection_visualizer.colors.helmet_on,
            "helmet_off": config.detection_visualizer.colors.helmet_off,
            "roi": config.detection_visualizer.colors.roi,
        }
        # Define constants for motorcycle size validation
        self.min_motorcycle_width = (
            config.detection_visualizer.motorcycle_validation.min_width
        )
        self.min_motorcycle_height = (
            config.detection_visualizer.motorcycle_validation.min_height
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

    def is_valid_motorcycle_size(self, width, height):
        """
        Checks if the bounding box size is appropriate for a motorcycle.
        Uses configuration values for minimum width and height.
        """
        if width <= 0 or height <= 0:
            return False

        # Check minimum size requirements from configuration
        if width < self.min_motorcycle_width or height < self.min_motorcycle_height:
            return False

        # Motorcycles are generally wider than tall (ratio < 1) or close to square
        # ใช้ ratio ที่เหมาะสำหรับมอเตอร์ไซค์: 0.3 <= ratio <= 2.0
        ratio = height / width
        return 0.3 <= ratio <= 2.0  # More flexible ratio for motorcycles

    def _draw_roi(self, frame, roi_points):
        """Draw the ROI polygon on the frame if roi_points exist."""
        if roi_points:
            roi_np = np.array(roi_points, np.int32)
            cv2.polylines(frame, [roi_np], True, self.colors["roi"], 2)

    def _draw_motorcycles(self, frame, results_motorcycle, roi_points):
        """Draw motorcycle boxes and labels."""
        if not (
            hasattr(results_motorcycle, "boxes")
            and results_motorcycle.boxes is not None
            and results_motorcycle.boxes.data is not None
        ):
            return

        for box in results_motorcycle.boxes.data:
            if len(box) < 6:
                continue

            x1, y1, x2, y2 = map(int, box[:4])

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            if not self.is_valid_motorcycle_size(width, height):
                continue

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if not self.is_in_roi((center_x, center_y), roi_points):
                continue

            # Use motorcycle color
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["motorcycle"], 2)

            # Add motorcycle label
            cv2.putText(
                frame,
                "Motorcycle",
                (x1, y1 - 10),
                self.detection_font,
                self.detection_scale,
                self.colors["motorcycle"],
                self.detection_thickness,
            )

    def _draw_helmets(self, frame, results_helmet, roi_points):
        """Draw helmet/no-helmet boxes and labels, return True if a no-helmet person in ROI is found."""
        found_person_no_helmet_in_roi = False

        if not (
            hasattr(results_helmet, "boxes")
            and results_helmet.boxes is not None
            and results_helmet.boxes.data is not None
        ):
            return found_person_no_helmet_in_roi

        for box in results_helmet.boxes.data:
            if len(box) < 6:
                continue

            x1, y1, x2, y2, conf, cls = box[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if not self.is_in_roi((center_x, center_y), roi_points):
                continue

            is_no_helmet = int(cls) == 0  # Class 0 usually means 'no_helmet'
            color = (
                self.colors["helmet_off"] if is_no_helmet else self.colors["helmet_on"]
            )
            label = "No Helmet" if is_no_helmet else "Helmet"

            if is_no_helmet:
                found_person_no_helmet_in_roi = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.detection_thickness)

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),  # Text position
                self.detection_font,
                self.detection_scale,
                color,
                self.detection_thickness,
            )

        return found_person_no_helmet_in_roi

    def draw_detections(self, frame, results_helmet, results_motorcycle, roi_points):
        """
        Draws detection results (motorcycles, helmets) and ROI onto the frame.
        """
        self._draw_roi(frame, roi_points)
        self._draw_motorcycles(frame, results_motorcycle, roi_points)

        found_person_no_helmet_in_roi = self._draw_helmets(
            frame, results_helmet, roi_points
        )
        return frame, found_person_no_helmet_in_roi
