import json
import os
from dataclasses import dataclass
from functools import lru_cache

try:
    from dotenv import load_dotenv

    load_dotenv()  # Load environment variables from .env file
except ImportError:
    pass  # dotenv not installed, skip loading .env file


@dataclass
class ColorsConfig:
    motorcycle: list[int]
    helmet_on: list[int]
    helmet_off: list[int]
    roi: list[int]

    @staticmethod
    def from_dict(data: dict) -> "ColorsConfig":
        return ColorsConfig(
            motorcycle=data["motorcycle"],
            helmet_on=data["helmet_on"],
            helmet_off=data["helmet_off"],
            roi=data["roi"],
        )


@dataclass
class MotorcycleValidationConfig:
    min_width: int
    min_height: int

    @staticmethod
    def from_dict(data: dict) -> "MotorcycleValidationConfig":
        return MotorcycleValidationConfig(
            min_width=data["min_width"], min_height=data["min_height"]
        )


@dataclass
class TimestampSettingsConfig:
    font: int
    scale: float
    color: list[int]
    thickness: int

    @staticmethod
    def from_dict(data: dict) -> "TimestampSettingsConfig":
        return TimestampSettingsConfig(
            font=data["font"],
            scale=data["scale"],
            color=data["color"],
            thickness=data["thickness"],
        )


@dataclass
class DetectionSettingsConfig:
    font: int
    scale: float
    thickness: int

    @staticmethod
    def from_dict(data: dict) -> "DetectionSettingsConfig":
        return DetectionSettingsConfig(
            font=data["font"], scale=data["scale"], thickness=data["thickness"]
        )


@dataclass
class DetectionVisualizerConfig:
    colors: ColorsConfig
    motorcycle_validation: MotorcycleValidationConfig
    timestamp_settings: TimestampSettingsConfig
    detection_settings: DetectionSettingsConfig
    roi_points_video: list[list[int]]
    roi_points_webcam: list[list[int]]

    @staticmethod
    def from_dict(data: dict) -> "DetectionVisualizerConfig":
        return DetectionVisualizerConfig(
            colors=ColorsConfig.from_dict(data["colors"]),
            motorcycle_validation=MotorcycleValidationConfig.from_dict(
                data["motorcycle_validation"]
            ),
            timestamp_settings=TimestampSettingsConfig.from_dict(
                data["timestamp_settings"]
            ),
            detection_settings=DetectionSettingsConfig.from_dict(
                data["detection_settings"]
            ),
            roi_points_video=data["roi_points_video"],
            roi_points_webcam=data["roi_points_webcam"],
        )

    def get_roi_points(self, use_webcam: bool) -> list[list[int]]:
        """เลือก ROI points ตาม input source"""
        return self.roi_points_webcam if use_webcam else self.roi_points_video


@dataclass
class ModelSettingsConfig:
    helmet_model_path: str
    motorcycle_model_path: str
    helmet_conf_threshold: float
    motorcycle_conf_threshold: float
    helmet_detection_interval: int

    @staticmethod
    def from_dict(data: dict) -> "ModelSettingsConfig":
        return ModelSettingsConfig(
            helmet_model_path=data["helmet_model_path"],
            motorcycle_model_path=data["motorcycle_model_path"],
            helmet_conf_threshold=data["helmet_conf_threshold"],
            motorcycle_conf_threshold=data["motorcycle_conf_threshold"],
            helmet_detection_interval=data["helmet_detection_interval"],
        )


@dataclass
class ApplicationSettingsConfig:
    video_path: str
    use_webcam: bool
    webcam_id: int

    @staticmethod
    def from_dict(data: dict) -> "ApplicationSettingsConfig":
        return ApplicationSettingsConfig(
            video_path=data["video_path"],
            use_webcam=data["use_webcam"],
            webcam_id=data["webcam_id"],
        )


@dataclass
class GemeniConfig:
    model: str
    api_key: str

    @staticmethod
    def from_dict(data: dict) -> "GemeniConfig":
        api_key = data.get("api_key")
        if "api_key_env" in data:
            api_key = os.environ.get(data["api_key_env"], api_key)

        return GemeniConfig(
            model=data["model"],
            api_key=api_key,
        )


@dataclass
class Configuration:
    detection_visualizer: DetectionVisualizerConfig
    model_settings: ModelSettingsConfig
    application_settings: ApplicationSettingsConfig
    gemeni: GemeniConfig

    @staticmethod
    def from_dict(data: dict) -> "Configuration":
        return Configuration(
            detection_visualizer=DetectionVisualizerConfig.from_dict(
                data["detection_visualizer"]
            ),
            model_settings=ModelSettingsConfig.from_dict(data["model_settings"]),
            application_settings=ApplicationSettingsConfig.from_dict(
                data["application_settings"]
            ),
            gemeni=GemeniConfig.from_dict(data["gemeni"]),
        )

    @staticmethod
    @lru_cache
    def get_config() -> "Configuration":
        site = os.environ.get("SITE", "development")
        with open(f"config.{site}.json", "r") as f:
            data = json.load(f)
        return Configuration.from_dict(data)
