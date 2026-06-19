import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


# ── Detection settings ───────────────────────────────────────────────────────


@dataclass
class DetectionConfig:
    """Configuration for motorcycle and helmet detection."""

    pad_filter: int  # Padding for helmet proximity to motorcycle
    helmet_detect_confidence: float  # Confidence threshold for helmet detection
    helmet_detect_imgsz: int  # Image size for helmet detection model
    motorcycle_confidence: float  # Confidence threshold for motorcycle detection
    line_position_percent: (
        float  # Detection line as percentage of frame width (0.0-1.0)
    )

    @staticmethod
    def from_dict(data: dict) -> "DetectionConfig":
        """Create DetectionConfig from dictionary."""
        return DetectionConfig(
            pad_filter=data.get("pad_filter", 80),
            helmet_detect_confidence=data.get("helmet_detect_confidence", 0.20),
            helmet_detect_imgsz=data.get("helmet_detect_imgsz", 1280),
            motorcycle_confidence=data.get("motorcycle_confidence", 0.5),
            line_position_percent=data.get("line_position_percent", 0.5),
        )


# ── Model settings ───────────────────────────────────────────────────────────


@dataclass
class ModelSettingsConfig:
    moto_model_path: str
    helmet_model_path: str
    helmet_conf_threshold: float
    jpeg_quality: int

    @staticmethod
    def from_dict(data: dict) -> "ModelSettingsConfig":
        return ModelSettingsConfig(
            moto_model_path=data.get("moto_model_path", "yolov8n"),
            helmet_model_path=data["helmet_model_path"],
            helmet_conf_threshold=data["helmet_conf_threshold"],
            jpeg_quality=data.get("jpeg_quality", 60),
        )


# ── Application settings ─────────────────────────────────────────────────────


@dataclass
class ApplicationSettingsConfig:
    video_path: str
    use_webcam: bool
    webcam_id: int

    @staticmethod
    def from_dict(data: dict) -> "ApplicationSettingsConfig":
        # RTSP_VIDEO_PATH env overrides config.json and forces use_webcam=False
        rtsp_override = os.environ.get("RTSP_VIDEO_PATH", "").strip()
        if rtsp_override:
            return ApplicationSettingsConfig(
                video_path=rtsp_override,
                use_webcam=False,
                webcam_id=data.get("webcam_id", 0),
            )

        use_webcam_env = os.environ.get("USE_WEBCAM", "").strip().lower()
        if use_webcam_env in ("true", "1", "yes"):
            use_webcam = True
        elif use_webcam_env in ("false", "0", "no"):
            use_webcam = False
        else:
            use_webcam = data["use_webcam"]

        return ApplicationSettingsConfig(
            video_path=data["video_path"],
            use_webcam=use_webcam,
            webcam_id=data.get("webcam_id", 0),
        )


# ── Database settings ────────────────────────────────────────────────────────


@dataclass
class PostgresConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @staticmethod
    def from_env() -> "PostgresConfig":
        return PostgresConfig(
            host=os.environ.get("DATABASE_HOST", "localhost"),
            port=int(os.environ.get("DATABASE_PORT", "5432")),
            user=os.environ.get("DATABASE_USER", "postgres"),
            password=os.environ.get("DATABASE_PASSWORD", "password"),
            database=os.environ.get("DATABASE_NAME", "helmet_detection"),
        )


# ── Auth settings ────────────────────────────────────────────────────────────


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
        return RefreshTokenCookie(
            key=str(obj.get("key")),
            value=str(obj.get("value")),
            httponly=bool(obj.get("httponly", False)),
            secure=bool(obj.get("secure", False)),
            samesite=str(obj.get("samesite", "lax")),
            max_age=int(obj.get("max_age", 2592000)),
            path=str(obj.get("path", "/")),
            domain=obj.get("domain"),
        )


@dataclass
class Key:
    secret_key: str
    algorithm: str = "HS256"
    access_token_minutes: int = 30

    @staticmethod
    def from_dict(obj: Any) -> "Key":
        return Key(
            secret_key=str(obj.get("secret_key")),
            algorithm=str(obj.get("algorithm", "HS256")),
            access_token_minutes=int(obj.get("access_token_minutes", 30)),
        )


# ── Root config ──────────────────────────────────────────────────────────────


@dataclass
class Configuration:
    model_settings: ModelSettingsConfig
    application_settings: ApplicationSettingsConfig
    postgres: PostgresConfig
    refresh_token_cookie: RefreshTokenCookie
    key: Key
    detection: DetectionConfig

    @staticmethod
    def from_dict(data: dict) -> "Configuration":
        return Configuration(
            model_settings=ModelSettingsConfig.from_dict(data["model_settings"]),
            application_settings=ApplicationSettingsConfig.from_dict(
                data["application_settings"]
            ),
            postgres=PostgresConfig.from_env(),
            refresh_token_cookie=RefreshTokenCookie.from_dict(
                data["refresh_token_cookie"]
            ),
            key=Key.from_dict(data["key"]),
            detection=DetectionConfig.from_dict(data.get("detection", {})),
        )

    @staticmethod
    @lru_cache
    def get_config() -> "Configuration":
        """Load and cache application configuration from JSON file.

        Determines config file based on SITE environment variable (default: development).
        Returns cached configuration to avoid reloading.

        Returns:
            Configuration object with all settings

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config JSON is invalid
        """
        site = os.environ.get("SITE", "development")
        config_path = Path(f"config.{site}.json")

        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)

        config = Configuration.from_dict(data)
        src = config.application_settings
        source_desc = (
            f"webcam id={src.webcam_id}"
            if src.use_webcam
            else f"RTSP: {src.video_path}"
        )
        logger.info(f"Video source: {source_desc}")
        return config
