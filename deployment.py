import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

class Detection:
    def __init__(self, model_path: str, classes: list):
        self.model = YOLO(model_path)  # Load mô hình từ best.pt
        self.classes = classes

    def __call__(self, image_path: str, conf_threshold: float = 0.3):
        # Đọc ảnh
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # Chạy mô hình YOLOv8
        results = self.model(image)[0]

        # Nếu không có kết quả, trả về rỗng
        if not results.boxes:
            print("❌ Không phát hiện được vật thể nào!")
            return results, {'boxes': [], 'confidences': [], 'classes': []}

        boxes, confidences, class_ids = [], [], []

        for result in results.boxes:
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            conf = result.conf[0].item()
            class_id = int(result.cls[0].item())

            if conf > conf_threshold:
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                confidences.append(round(conf * 100, 2))

                # Kiểm tra class_id có hợp lệ không
                if 0 <= class_id < len(self.classes):
                    class_ids.append(self.classes[class_id])
                else:
                    class_ids.append("Unknown")

        return results, {'boxes': boxes, 'confidences': confidences, 'classes': class_ids}

def draw_boxes(image_path, results):
    image = results[0].plot()  # Ảnh Matplotlib
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Chuyển sang OpenCV
    cv2.imshow("Kết quả dự đoán", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Mở hộp thoại chọn file ảnh
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    img_pth = filedialog.askopenfilename(
        title="Chọn ảnh để dự đoán", 
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
    )

    if not img_pth:
        print("❌ Không có ảnh nào được chọn!")
        exit()

    # Danh sách class của YOLOv8
    CLASSES_YOLO = ['damaged door', 'damaged window', 'damaged headlight', 'damaged mirror', 
                    'dent', 'damaged hood', 'damaged bumper', 'damaged wind shield']

    # Load mô hình
    detection = Detection(model_path='best.pt', classes=CLASSES_YOLO)

    # Chạy mô hình và hiển thị kết quả
    results, output = detection(img_pth)
    print(output)
    
    # Nếu có kết quả, vẽ bounding box
    if output['boxes']:
        draw_boxes(img_pth, results)
