# Measure Object YOLO

Proyek ini menggunakan model YOLO untuk mendeteksi objek pada gambar atau video (kamera) secara real-time, serta mengukur ukuran objek berdasarkan referensi objek yang diketahui lebarnya (dalam cm).

## Fitur
- Deteksi objek menggunakan model YOLOv8.
- Pengukuran otomatis objek pada gambar atau kamera dengan referensi objek.
- Mendukung input dari gambar statis maupun kamera/webcam.

## Struktur Folder
```
camera-realtime.py      # Deteksi & pengukuran objek secara real-time dari kamera
image-measure.py        # Deteksi & pengukuran objek dari gambar
assets/
    img/
        mybottle.jpeg   # Contoh gambar untuk pengujian
    model/
        yolo11m.pt      # Model YOLO custom/standar
        yolo11n.pt
        yolo11s.pt
        yolov8n.pt
```

## Persyaratan
- Python 3.x
- OpenCV (`opencv-python`)
- PyTorch
- ultralytics (untuk YOLOv8)

Install dependensi:
```sh
pip install opencv-python torch ultralytics
```

## Cara Menjalankan

### 1. Deteksi & Pengukuran dari Gambar
Edit path gambar pada `image-measure.py` jika perlu:
```python
image_path = "assets/img/mybottle.jpeg"
```
Jalankan:
```sh
python image-measure.py
```

### 2. Deteksi & Pengukuran Real-time dari Kamera
Jalankan:
```sh
python camera-realtime.py
```
Tekan `q` untuk keluar dari mode kamera.

## Catatan
- Pastikan file model YOLO (`.pt`) sudah tersedia di folder `assets/model/`.
- Ubah nama kelas referensi dan lebar objek referensi pada kode jika menggunakan objek referensi yang berbeda.
- Hasil pengukuran akan tampil pada jendela gambar/video.

## Lisensi
Proyek ini hanya untuk keperluan pembelajaran.
