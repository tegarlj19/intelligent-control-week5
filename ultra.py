from ultralytics import YOLO
import cv2
import torch

# Load model YOLOv8 Pose dan pindahkan ke GPU jika tersedia
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n-pose.pt").to(device)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Indeks titik kunci berdasarkan model YOLOv8 Pose
KEYPOINTS_TO_KEEP = {
    0: "Hidung", 1: "Mata Kiri", 2: "Mata Kanan",
    3: "Telinga Kiri", 4: "Telinga Kanan", 5: "Bahu Kiri",
    6: "Bahu Kanan", 7: "Siku Kiri", 8: "Siku Kanan",
    9: "Pergelangan Tangan Kiri", 10: "Pergelangan Tangan Kanan",
    11: "Pinggul Kiri", 12: "Pinggul Kanan", 13: "Lutut Kiri",
    14: "Lutut Kanan", 15: "Pergelangan Kaki Kiri", 16: "Pergelangan Kaki Kanan"
}

# Definisi koneksi antara titik kunci untuk membentuk pose tubuh
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Kepala
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Lengan kiri dan kanan
    (5, 11), (6, 12), (11, 12),  # Badan atas
    (11, 13), (13, 15), (12, 14), (14, 16),  # Kaki kiri dan kanan
    (5, 6)  # Bahu kiri ke bahu kanan
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Konversi frame ke RGB untuk YOLOv8
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi pose menggunakan YOLOv8
    results = model(frame_rgb)
    
    annotated_frame = frame.copy()

    for result in results:
        if result.keypoints is not None:  # Pastikan ada keypoints yang terdeteksi
            keypoints = result.keypoints.xy.cpu().numpy()  # Konversi ke numpy
            boxes = result.boxes.xyxy.cpu().numpy()  # Koordinat bounding box
            confidences = result.boxes.conf.cpu().numpy()  # Confidence score

            for i, person in enumerate(keypoints):  # Iterasi setiap orang dalam frame
                # Gambar bounding box
                if i < len(boxes):
                    x_min, y_min, x_max, y_max = map(int, boxes[i])
                    confidence = confidences[i]  # Confidence score
                    
                    # Kotak bounding box warna kuning
                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                    
                    # Tambahkan teks confidence score
                    cv2.putText(annotated_frame, f"Conf: {confidence:.2f}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Gambar titik kunci dengan validasi
                valid_points = {}  # Simpan hanya titik yang valid

                for j, name in KEYPOINTS_TO_KEEP.items():
                    if j < len(person):  # Pastikan indeks valid
                        x, y = int(person[j][0]), int(person[j][1])
                        
                        # Cek apakah koordinat berada dalam area gambar
                        if 0 < x < frame.shape[1] and 0 < y < frame.shape[0]:
                            valid_points[j] = (x, y)  # Simpan hanya titik yang valid
                            cv2.circle(annotated_frame, (x, y), 6, (0, 255, 0), -1)  # Titik hijau lebih besar
                            cv2.putText(annotated_frame, name, (x + 5, y - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Gambar koneksi antar titik untuk membentuk pose
                for (p1, p2) in POSE_CONNECTIONS:
                    if p1 in valid_points and p2 in valid_points:  # Pastikan kedua titik valid
                        x1, y1 = valid_points[p1]
                        x2, y2 = valid_points[p2]
                        cv2.line(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Garis biru
    
    cv2.imshow("YOLOv8 Pose Estimation with Bounding Box", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
