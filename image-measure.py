import cv2
import numpy as np
from ultralytics import YOLO

# --- KONFIGURASI ---
# Ganti nama kelas dan lebar objek referensi Anda di sini
REFERENCE_OBJECT_CLASS_NAME = "bottle"  # contoh: 'cell phone'
REFERENCE_OBJECT_WIDTH_CM = 5.69           # contoh: lebar cell phone dalam cm

# Muat model YOLO
model = YOLO('assets/model/yolo11m.pt')  # atau model lain sesuai kebutuhan

# Tambahkan dictionary warna untuk beberapa kelas umum
CLASS_COLORS = {
    "cell phone": (0, 255, 255),
    "person": (255, 0, 0),
    "book": (0, 128, 255),
    "bottle": (0, 255, 0),
    "cup": (255, 0, 255),
    # Tambahkan kelas lain sesuai kebutuhan
}

def draw_bounding_box(frame, box, pixels_per_cm=None):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = box.conf[0]
    class_id = int(box.cls[0])
    class_name = model.names[class_id]

    color = CLASS_COLORS.get(class_name, (0, 255, 0))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{class_name} ({confidence:.2f})"
    if pixels_per_cm is not None and pixels_per_cm > 0:
        width_px = x2 - x1
        height_px = y2 - y1
        width_cm = width_px / pixels_per_cm
        height_cm = height_px / pixels_per_cm
        label += f" | W: {width_cm:.1f}cm, H: {height_cm:.1f}cm"
    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, cv2.FILLED)
    cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# --- GANTI PATH GAMBAR DI SINI ---
image_path = "assets/img/mybottle.jpeg"  # Ganti dengan nama file gambar Anda
frame = cv2.imread(image_path)
if frame is None:
    print("Error: Gambar tidak ditemukan.")
    exit()

results = model(frame)
pixels_per_cm = None

# Cari objek referensi
for box in results[0].boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    if class_name.lower() == REFERENCE_OBJECT_CLASS_NAME.lower():
        x1, y1, x2, y2 = box.xyxy[0]
        width_in_pixels = x2 - x1
        if width_in_pixels > 0:
            pixels_per_cm = width_in_pixels / REFERENCE_OBJECT_WIDTH_CM
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
            cv2.putText(frame, f"Reference: {REFERENCE_OBJECT_CLASS_NAME}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            break

# Gambar bounding box dan ukuran untuk semua objek
for box in results[0].boxes:
    draw_bounding_box(frame, box, pixels_per_cm)

cv2.imshow("Deteksi dan Pengukuran Objek pada Gambar", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()