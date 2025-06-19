from ultralytics import YOLO
import cv2
import os

# Dictionary for Thai characters
THAI_CHARS = {
    'A00': 'ก', 'A01': 'ข', 'A02': 'ฃ', 'A03': 'ค', 'A04': 'ฅ', 
    'A05': 'ฆ', 'A06': 'ง', 'A07': 'จ', 'A08': 'ฉ', 'A09': 'ช',
    'A10': 'ซ', 'A11': 'ฌ', 'A12': 'ญ', 'A13': 'ฎ', 'A14': 'ฏ',
    'A15': 'ฐ', 'A16': 'ฑ', 'A17': 'ฒ', 'A18': 'ณ', 'A19': 'ด',
    'A20': 'ต', 'A21': 'ถ', 'A22': 'ท', 'A23': 'ธ', 'A24': 'น',
    'A25': 'บ', 'A26': 'ป', 'A27': 'ผ', 'A28': 'ฝ', 'A29': 'พ',
    'A30': 'ฟ', 'A31': 'ภ', 'A32': 'ม', 'A33': 'ย', 'A34': 'ร',
    'A35': 'ล', 'A36': 'ว', 'A37': 'ฮ'  
}

# Fix model path
model_path = os.path.join("model", "plate_Epoch40.pt")
model = YOLO(model_path)

# Fix image path
image_path = os.path.join("case_plate", "s.jpg")
image = cv2.imread(image_path)

# Add error handling for file loading
if image is None:
    raise ValueError(f"Could not load image from {image_path}")

results = model(image)

try:
    for result in results:
        detected_text = []  # เก็บตัวอักษรที่ตรวจจับได้
        
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            original_label = model.names[int(cls)]
            
            # แสดงผลเฉพาะเมื่อ confidence สูงพอ
            if conf > 0.5:  # ปรับค่า threshold ตามความเหมาะสม
                if original_label in THAI_CHARS:
                    label = THAI_CHARS[original_label]
                else:
                    label = original_label
                
                detected_text.append(label)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # แสดงผลทั้งหมดที่ตรวจจับได้
        if detected_text:
            print("Detected text:", " ".join(detected_text))
    cv2.imshow("Plate Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error during detection: {str(e)}")
    if 'image' in locals():
        cv2.destroyAllWindows()
