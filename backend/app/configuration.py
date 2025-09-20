import json
import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class ColorsConfig:
    person: list[int]
    helmet_on: list[int]
    helmet_off: list[int]
    roi: list[int]

    @staticmethod
    def from_dict(data: dict) -> "ColorsConfig":
        return ColorsConfig(
            person=data["person"],
            helmet_on=data["helmet_on"],
            helmet_off=data["helmet_off"],
            roi=data["roi"],
        )


@dataclass
class PersonValidationConfig:
    min_height_width_ratio: float

    @staticmethod
    def from_dict(data: dict) -> "PersonValidationConfig":
        return PersonValidationConfig(
            min_height_width_ratio=data["min_height_width_ratio"]
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
    person_validation: PersonValidationConfig
    timestamp_settings: TimestampSettingsConfig
    detection_settings: DetectionSettingsConfig
    roi_points: list[list[int]]

    @staticmethod
    def from_dict(data: dict) -> "DetectionVisualizerConfig":
        return DetectionVisualizerConfig(
            colors=ColorsConfig.from_dict(data["colors"]),
            person_validation=PersonValidationConfig.from_dict(
                data["person_validation"]
            ),
            timestamp_settings=TimestampSettingsConfig.from_dict(
                data["timestamp_settings"]
            ),
            detection_settings=DetectionSettingsConfig.from_dict(
                data["detection_settings"]
            ),
            roi_points=data["roi_points"],
        )


@dataclass
class ModelSettingsConfig:
    helmet_model_path: str
    person_model_path: str
    helmet_conf_threshold: float
    person_conf_threshold: float
    helmet_detection_interval: int

    @staticmethod
    def from_dict(data: dict) -> "ModelSettingsConfig":
        return ModelSettingsConfig(
            helmet_model_path=data["helmet_model_path"],
            person_model_path=data["person_model_path"],
            helmet_conf_threshold=data["helmet_conf_threshold"],
            person_conf_threshold=data["person_conf_threshold"],
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
class Configuration:
    detection_visualizer: DetectionVisualizerConfig
    model_settings: ModelSettingsConfig
    application_settings: ApplicationSettingsConfig

    @staticmethod
    def from_dict(data: dict) -> "Configuration":
        return Configuration(
            detection_visualizer=DetectionVisualizerConfig.from_dict(
                data["detection_visualizer"]
            ),
            model_settings=ModelSettingsConfig.from_dict(
                data["model_settings"]
            ),  # Parse new section
            application_settings=ApplicationSettingsConfig.from_dict(
                data["application_settings"]
            ),  # Parse new section
        )

    @staticmethod
    @lru_cache
    def get_config() -> "Configuration":
        site = os.environ.get("SITE", "development")
        with open(f"config.{site}.json", "r") as f:
            data = json.load(f)
        return Configuration.from_dict(data)
