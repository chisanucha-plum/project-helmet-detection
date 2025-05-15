from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import os

# Initialize models
yolo_model = YOLO('model/plate_Epoch40.pt')
reader = easyocr.Reader(['th', 'en'])

def detect_and_read_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "ไม่สามารถโหลดรูปภาพได้"
    
    # 1. YOLO detection for plate sections
    results = yolo_model(img)
    detections = []
    
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box[:4])
                class_name = result.names[int(cls)]
                # Crop the detected region
                crop_img = img[y1:y2, x1:x2]
                # Read text from cropped image using EasyOCR
                ocr_result = reader.readtext(crop_img)
                
                if ocr_result:
                    text = ocr_result[0][1]  # Get the text
                    ocr_conf = ocr_result[0][2]  # Get OCR confidence
                else:
                    text = ""
                    ocr_conf = 0
                
                detections.append({
                    'y': y1,
                    'class': class_name,
                    'yolo_conf': conf,
                    'ocr_text': text,
                    'ocr_conf': ocr_conf,
                    'box': (x1, y1, x2, y2)
                })
    
    # Sort by vertical position
    detections.sort(key=lambda x: x['y'])
    
    # Draw results on image
    for det in detections:
        x1, y1, x2, y2 = det['box']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # แสดงเฉพาะข้อความที่อ่านได้จาก OCR
        if det['ocr_text']:
            label = det['ocr_text']
        else:
            label = det['class']  # ถ้าไม่มีข้อความ ใช้ชื่อ class แทน
            
        # ปรับขนาดและตำแหน่งของข้อความให้เหมาะสม
        cv2.putText(img, label, 
                   (x1, y1-5),  # ปรับตำแหน่งข้อความ
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7,  # ปรับขนาดตัวอักษร
                   (0, 255, 0),  # สีเขียว
                   2)  # ความหนา
    
    # Save result image
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "detected_" + os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    
    return detections, output_path

def preprocess_image(img):
    """ปรับปรุงคุณภาพภาพก่อนส่งให้ OCR"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)
    return denoised

def format_results(detections):
    """จัดรูปแบบผลลัพธ์ให้อ่านง่าย"""
    result = []
    for det in detections:
        if validate_detection(det):
            result.append({
                'ประเภท': det['class'],
                'ข้อความ': det['ocr_text'] or 'ไม่สามารถอ่านได้',
                'ความเชื่อมั่น': f"YOLO: {det['yolo_conf']:.2f}, OCR: {det['ocr_conf']:.2f}"
            })
    return result

def validate_detection(detection):
    """ตรวจสอบความถูกต้องของการตรวจจับ"""
    # ตรวจสอบความเชื่อมั่นขั้นต่ำ
    if detection['yolo_conf'] < 0.5 or detection['ocr_conf'] < 0.3:
        return False
        
    # ตรวจสอบขนาดของ bounding box
    x1, y1, x2, y2 = detection['box']
    width = x2 - x1
    height = y2 - y1
    if width < 20 or height < 20:
        return False
        
    return True

if __name__ == "__main__":
    image_path = "case_plate/s.jpg"
    detections, output_path = detect_and_read_plate(image_path)
    
    print("\nผลการตรวจจับ:")
    for det in detections:
        print(f"ประเภท: {det['class']}")
        print(f"ข้อความ: {det['ocr_text']}")
        print(f"ความเชื่อมั่น YOLO: {det['yolo_conf']:.2f}")
        print(f"ความเชื่อมั่น OCR: {det['ocr_conf']:.2f}")
        print("---")
    
    print(f"\nบันทึกภาพผลลัพธ์ที่: {output_path}")
