import json
import logging
import os
from typing import Optional

from app.configuration import Configuration
from app.schemas.gemeni import AnalysisResult
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self, model: str = None, api_key: str = None):
        config = Configuration.get_config()
        self.model = model or config.gemeni.model
        self.api_key = config.gemeni.api_key
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.client = None

    def count_motorcycle_riders(self, image_path: str) -> Optional[int]:
        """
        Count the number of people riding motorcycles in the image.
        Args:
            image_path: Path to the image file
        Returns:
            Number of motorcycle riders, or None if error
        """
        if not self.client or not os.path.exists(image_path):
            return None

        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=image_data,
                        )
                    ),
                    "Count only the number of people riding the motorcycle(s). Do not count background people. Answer only with a number.",
                ],
            )

            return int(response.text.strip())
        except Exception:
            return None

    def analyze_helmet_compliance(self, image_path: str) -> Optional[AnalysisResult]:
        """
        Analyze helmet compliance for motorcycle riders.
        วิเคราะห์การสวมหมวกกันน็อคของผู้ขับขี่มอเตอร์ไซค์
        """
        if not self.client or not os.path.exists(image_path):
            return None

        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            prompt = """Analyze this motorcycle image for helmet compliance.

            IMPORTANT: Return ONLY clean JSON without markdown formatting.
            Do NOT wrap your response in ```json blocks.
            
            Return this exact JSON structure:
            {"helmet": true/false, "total_person": number, "violations": "text"}
            
            Rules:
            - helmet: true only if ALL motorcycle riders wear helmets
            - total_person: count only people riding motorcycles
            - violations: describe violations or "None" if compliant
            
            Valid example: {"helmet": false, "total_person": 1, "violations": "Driver not wearing helmet"}"""

            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=image_data,
                        )
                    ),
                    prompt,
                ],
            )

            analysis_text = response.text.strip()
            return self._parse_to_analysis_result(analysis_text)

        except Exception as e:
            logging.warning(f"Gemini analysis failed: {e}")
            return None

    def _parse_to_analysis_result(self, raw_text: str) -> AnalysisResult:
        """Parse Gemini JSON response to AnalysisResult schema."""
        try:
            cleaned_text = raw_text.strip()

            # Remove markdown code blocks (```json ... ```)
            if cleaned_text.startswith("```"):
                lines = cleaned_text.split("\n")
                # Find the start and end of JSON content
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_json = not in_json
                        continue
                    if in_json:
                        json_lines.append(line)
                cleaned_text = "\n".join(json_lines)

            data = json.loads(cleaned_text)
            return AnalysisResult(
                helmet=data.get("helmet"),
                total_person=data.get("total_person"),
                violations=data.get("violations", "None"),
            )

        except json.JSONDecodeError as e:
            logging.warning(f"JSON parsing failed: {e}, falling back to simple parsing")

            # Simple fallback - extract basic info
            return AnalysisResult(
                helmet=False,  # Default to false for safety
                total_person=None,
                violations=f"Parsing error: {raw_text[:100]}...",
            )

    def is_service_available(self) -> bool:
        """
        Check if Gemini service is available.
        ตรวจสอบว่าบริการ Gemini พร้อมใช้งานหรือไม่
        """
        return self.client is not None


gemini_service = GeminiService()
