from ultralytics import YOLO
import cv2

model = YOLO("plate_Epoch40.pt")  

image_path = "testplate\f.jpg" 
image = cv2.imread(image_path)

results = model(image)  

for result in results:
    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
        x1, y1, x2, y2 = map(int, box[:4])
        label = model.names[int(cls)] 
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.imshow("Plate Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
