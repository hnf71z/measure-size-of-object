import cv2
import numpy as np
from ultralytics import YOLO

# --- KONFIGURASI ---
# Ganti nama kelas dan lebar objek referensi Anda di sini
# Nama kelas harus sesuai dengan dataset COCO yang digunakan YOLOv8, misal: 'person', 'car', 'cell phone', 'book'
# Untuk daftar lengkap nama kelas, kunjungi: https://roboflow.com/models/yolov8
REFERENCE_OBJECT_CLASS_NAME = "cell phone"  # Ganti dengan objek referensi Anda, contoh: 'cell phone'
REFERENCE_OBJECT_WIDTH_CM = 7.69     # Ganti dengan lebar aktual objek referensi Anda dalam CM (misal: lebar buku A5)

# Muat model YOLOv8 yang sudah dilatih sebelumnya
# 'yolov8n.pt' adalah model terkecil dan tercepat. Untuk akurasi lebih tinggi, gunakan 'yolov8m.pt' atau 'yolov8l.pt'
model = YOLO('assets/model/yolo11s.pt')

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

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
    """Fungsi untuk menggambar bounding box dan menampilkan ukuran."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = box.conf[0]
    class_id = int(box.cls[0])
    class_name = model.names[class_id]

    # Pilih warna sesuai kelas, default hijau jika tidak ada di dictionary
    color = CLASS_COLORS.get(class_name, (0, 255, 0))

    # Gambar bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Siapkan teks label
    label = f"{class_name} ({confidence:.2f})"
    
    # Hitung dan tampilkan ukuran jika pixels_per_cm tersedia
    if pixels_per_cm is not None and pixels_per_cm > 0:
        width_px = x2 - x1
        height_px = y2 - y1
        
        width_cm = width_px / pixels_per_cm
        height_cm = height_px / pixels_per_cm
        
        label += f" | W: {width_cm:.1f}cm, H: {height_cm:.1f}cm"

    # Tampilkan label di atas bounding box
    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, cv2.FILLED)
    cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


while True:
    # Baca frame dari kamera
    success, frame = cap.read()
    if not success:
        print("Error: Gagal membaca frame.")
        break

    # Tidak perlu resize, langsung deteksi pada frame asli
    results = model(frame)
    
    pixels_per_cm = None

    # --- Langkah 1: Cari Objek Referensi dan Hitung Rasio Piksel/CM ---
    # Loop melalui semua objek yang terdeteksi untuk menemukan objek referensi
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        if class_name.lower() == REFERENCE_OBJECT_CLASS_NAME.lower():
            x1, y1, x2, y2 = box.xyxy[0]
            width_in_pixels = x2 - x1
            
            # Hitung rasio hanya jika lebar piksel valid
            if width_in_pixels > 0:
                pixels_per_cm = width_in_pixels / REFERENCE_OBJECT_WIDTH_CM
                # Gambar bounding box khusus untuk referensi
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
                cv2.putText(frame, f"Reference: {REFERENCE_OBJECT_CLASS_NAME}", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                break # Hentikan pencarian setelah referensi ditemukan

    # --- Langkah 2: Ukur Semua Objek Menggunakan Rasio yang Ditemukan ---
    # Loop sekali lagi untuk menggambar semua bounding box dan ukurannya
    for box in results[0].boxes:
        draw_bounding_box(frame, box, pixels_per_cm)

    # Tampilkan frame yang sudah diproses
    cv2.imshow("Deteksi dan Pengukuran Objek YOLOv8", frame)

    # Hentikan program jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()