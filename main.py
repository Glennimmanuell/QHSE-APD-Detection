import torch
import cv2
from pathlib import Path
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"pake GPU/CPU ?: {device}")

model_path = "best.pt"
model = YOLO(model_path)
model.to(device)

cap = cv2.VideoCapture(0)

class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
unsafe_classes = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']

while True:
    ret, frame = cap.read()
    results = model(frame, device=device)
    
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, cls = box.tolist()
            label = f"{class_names[int(cls)]} ({score:.2f})"
            
            if class_names[int(cls)] in unsafe_classes:
                color = (0, 0, 255)
            elif class_names[int(cls)] == 'Person':
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow("QHSE APD Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()