import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal


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
        use_webcam = bool(data["use_webcam"])
        video_path = data.get("video_path", "")
        video_path_env = data.get("video_path_env", "RTSP_VIDEO_PATH")
        video_path = os.environ.get(video_path_env, video_path)

        if not use_webcam and (not video_path or not str(video_path).strip()):
            raise ValueError(
                f"Missing video source: set {video_path_env} in environment or provide application_settings.video_path"
            )

        return ApplicationSettingsConfig(
            video_path=video_path,
            use_webcam=use_webcam,
            webcam_id=data["webcam_id"],
        )


@dataclass
class PostgresConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @staticmethod
    def from_env() -> "PostgresConfig":
        host = os.environ.get("DATABASE_HOST", "localhost")
        port = int(os.environ.get("DATABASE_PORT", "5432"))
        user = os.environ.get("DATABASE_USER", "postgres")
        password = os.environ.get("DATABASE_PASSWORD", "password")
        database = os.environ.get("DATABASE_NAME", "helmet_detection")

        return PostgresConfig(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
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
class RefreshTokenCookie:
    key: str
    value: str
    httponly: bool
    secure: bool
    max_age: int
    path: str
    domain: str | None
    samesite: Literal["lax", "strict", "none"] = "lax"

    @staticmethod
    def from_dict(obj: Any) -> "RefreshTokenCookie":
        _key = str(obj.get("key"))
        _value = str(obj.get("value"))
        _httponly = bool(obj.get("httponly", False))
        _secure = bool(obj.get("secure", False))
        _samesite = str(obj.get("samesite", "lax"))
        _max_age = int(obj.get("max_age", 2592000))
        _path = str(obj.get("path", "/"))
        _domain = obj.get("domain")

        return RefreshTokenCookie(
            key=_key,
            value=_value,
            httponly=_httponly,
            secure=_secure,
            samesite=_samesite,
            max_age=_max_age,
            path=_path,
            domain=_domain,
        )


@dataclass
class Key:
    secret_key: str
    algorithm: str = "HS256"
    access_token_minutes: int = 30

    @staticmethod
    def from_dict(obj: Any) -> "Key":
        _secret_key = str(obj.get("secret_key"))
        _algorithm = str(obj.get("algorithm", "HS256"))
        _access_token_minutes = int(obj.get("access_token_minutes", 30))
        return Key(
            secret_key=_secret_key,
            algorithm=_algorithm,
            access_token_minutes=_access_token_minutes,
        )


@dataclass
class Configuration:
    detection_visualizer: DetectionVisualizerConfig
    model_settings: ModelSettingsConfig
    application_settings: ApplicationSettingsConfig
    gemeni: GemeniConfig
    postgres: PostgresConfig
    refresh_token_cookie: RefreshTokenCookie
    key: Key

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
            postgres=PostgresConfig.from_env(),  # อ่านจาก .env เท่านั้น
            refresh_token_cookie=RefreshTokenCookie.from_dict(
                data["refresh_token_cookie"]
            ),
            key=Key.from_dict(data["key"]),
        )

    @staticmethod
    @lru_cache
    def get_config() -> "Configuration":
        site = os.environ.get("SITE", "development")
        with open(f"config.{site}.json", "r") as f:
            data = json.load(f)
        return Configuration.from_dict(data)
