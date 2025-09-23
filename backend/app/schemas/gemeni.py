from typing import Optional

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    """Helmet analysis result schema."""

    helmet_status: Optional[bool] = Field(
        None, description="Overall helmet compliance (true if all persons wear helmets)"
    )
    passenger_count: Optional[int] = Field(
        None, description="Total number of passengers detected"
    )
    violations: Optional[str] = Field(
        None, description="Description of violations if any"
    )


class ImageInfo(BaseModel):
    """Image information schema."""

    filename: str = Field(..., description="Full path to the image file")
    timestamp: str = Field(
        ..., description="Image file modification timestamp (Thailand time)"
    )
    file_size: Optional[int] = Field(None, description="File size in bytes")


class SnapshotDirectoryInfo(BaseModel):
    """Snapshot directory information schema."""

    path: str = Field(..., description="Path to snapshots directory")
    total_files: int = Field(..., description="Total number of snapshot files")
    exists: bool = Field(..., description="Whether the directory exists")


class GeminiServiceInfo(BaseModel):
    """Gemini AI service information schema."""

    available: bool = Field(..., description="Whether Gemini service is available")
    status: str = Field(..., description="Service status description")


class HelmetComplianceResponse(BaseModel):
    """Response schema for helmet compliance analysis endpoint."""

    success: bool = Field(..., description="Whether the request was successful")
    analysis: AnalysisResult = Field(
        ..., description="Helmet compliance analysis result"
    )
    image_info: ImageInfo = Field(
        ..., description="Information about the analyzed image"
    )
    analysis_timestamp: str = Field(
        ..., description="Timestamp when analysis was performed (Thailand time)"
    )


class LatestSnapshotResponse(BaseModel):
    """Response schema for latest snapshot info endpoint."""

    success: bool = Field(..., description="Whether the request was successful")
    image_info: ImageInfo = Field(
        ..., description="Information about the latest snapshot"
    )
    snapshots_directory: SnapshotDirectoryInfo = Field(
        ..., description="Information about snapshots directory"
    )
    gemini_service: GeminiServiceInfo = Field(
        ..., description="Information about Gemini AI service"
    )
    message: str = Field(..., description="Thai language status message")


class ErrorResponse(BaseModel):
    """Error response schema."""

    success: bool = Field(False, description="Always false for error responses")
    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
