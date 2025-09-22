import os

from google import genai
from google.genai import types

# API Key
client = genai.Client(api_key="AIzaSyA-Y0nsZA33VProqj0VFzwaUcgd1mUvpxg")

# path ของภาพ
image_path = r"snapshots\capture_20250921_200451_838_helmet_mc_mc_2_5.jpg"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"ไม่พบไฟล์: {image_path}")

with open(image_path, "rb") as f:
    image_data = f.read()

response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents=[
        {
            "role": "user",
            "parts": [
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/jpeg",
                        data=image_data,
                    )
                ),
                types.Part(
                    text="Count only the number of people riding the motorcycle(s). Do not count background people. Answer only with a number."
                ),
            ],
        }
    ],
)

print("จำนวนผู้ขับขี่/ผู้โดยสารบนรถมอเตอร์ไซค์:", response.text)
