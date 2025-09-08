from google import genai
from google.genai import types

# สร้าง client
client = genai.Client(api_key="AIzaSyA-Y0nsZA33VProqj0VFzwaUcgd1mUvpxg")

# โหลดภาพที่ต้องการตรวจสอบ
image_path = r"src\image\persons.png"

with open(image_path, "rb") as f:
    image_data = f.read()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part(
            inline_data=types.Blob(
                mime_type="image/png",
                data=image_data,
            )
        ),
        "Count only the number of people riding the motorcycle(s). Do not count background people. Answer only with a number.",
    ],
)

print("จำนวนผู้ขับขี่/ผู้โดยสารบนรถมอเตอร์ไซค์:", response.text)
