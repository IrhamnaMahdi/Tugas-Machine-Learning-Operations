# Aplikasi Klasifikasi Spesies Burung  
**Klasifikasi Citra Menggunakan Deep Learning (MobileNetV2 & Streamlit)**

Proyek ini merupakan aplikasi **klasifikasi spesies burung berbasis citra** yang dibangun menggunakan **Deep Learning (TensorFlow – MobileNetV2)** dan ditampilkan melalui **aplikasi web interaktif menggunakan Streamlit**.  
Pengguna dapat mengunggah gambar burung dan sistem akan memprediksi spesies burung beserta tingkat kepercayaannya (confidence).

---

## Fitur Utama

- Upload gambar burung (JPG / PNG)
- Model Deep Learning berbasis **MobileNetV2 (Transfer Learning)**
- Visualisasi confidence prediksi
- Menampilkan hasil prediksi utama beserta persentase
- Inferensi cepat dengan model caching
- Antarmuka web interaktif dan modern (Streamlit)

---

## Struktur Proyek

1. tubes_mlops.py # Aplikasi Streamlit
2. MLOps.ipnyb # Script training model
3. class_indices.json # Class label mapping
4. README.md # Dokumentasi proyek
5. requirements.txt # Dependensi Python

## Dataset

- **Sumber**: Kaggle  
- **Link**: https://www.kaggle.com/datasets/muhammadadeelkaggle/birds-dataset  
- **Format**: Folder-based classification  

Contoh struktur dataset:
---
Birds dataset/
- amazon green parrot/
- gray parrot/
- macaw/
- white parrot/
---

Setiap folder merepresentasikan satu kelas / spesies burung.

## Arsitektur Model

- **Base Model**: MobileNetV2 (pretrained ImageNet)
- **Ukuran Input**: 224 × 224 × 3 (RGB)
- **Strategi Training**:
  - Feature extraction (base model dibekukan)
  - Fine-tuning pada layer akhir
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam

---

## Cara Menjalankan Proyek

### 1. Clone Repository

```
bash
git clone https://github.com/IrhamnaMahdi/Tugas-Machine-Learning-Operations.git
cd Tugas-Machine-Learning-Operations
```

### 2️. Install Dependensi

Disarankan menggunakan virtual environment:

```
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi Streamlit

```
streamlit run app.py
```
Kemudian buka browser dan akses:
```
http://localhost:8501
```

### Teknologi yang Digunakan
```
Python 3.9+
TensorFlow / Keras
MobileNetV2
Streamlit
NumPy
Pillow (PIL)
Matplotlib
```

## Evaluasi Model

Model dievaluasi menggunakan dataset validasi (20% split) selama 30 epoch untuk fase *feature extraction* dan dilanjutkan dengan 15 epoch untuk fase *fine-tuning*. Berikut adalah ringkasan performa model:

### 1. Metrik Performa Akhir
Berdasarkan hasil training terakhir (Fine-Tuning Epoch 5/15), model mencapai performa sebagai berikut:

| Metrik | Training Set | Validation Set |
| :--- | :--- | :--- |
| **Accuracy** | 98.69% | 94.59% |
| **Loss** | 0.0437 | 0.2898 |

### 2. Analisis Proses Training
Proses pelatihan dilakukan dalam dua tahap strategi *Transfer Learning*:

* **Fase 1: Feature Extraction (Frozen Base)**
    * Model dilatih dengan *base model* MobileNetV2 yang dibekukan (*frozen*).
    * Akurasi validasi mencapai **94.59%** pada Epoch ke-13, menunjukkan bahwa model pre-trained ImageNet sudah memiliki ekstraksi fitur yang sangat baik untuk mengenali pola burung.
    * Model menggunakan `Dropout(0.5)` untuk mencegah *overfitting* yang signifikan.

* **Fase 2: Fine-Tuning**
    * Melakukan *unfreeze* pada 30% layer teratas dari MobileNetV2.
    * Dilatih kembali dengan *learning rate* yang sangat kecil (`1e-4` turun ke `3e-5`) untuk menyesuaikan bobot secara spesifik terhadap dataset spesies burung.
    * Hasil akhirnya menunjukkan model sangat *robust* dengan akurasi training mendekati 99% dan validasi yang stabil di angka ~94-95%.

### 3. Konfigurasi Hyperparameter
* **Optimizer**: Adam
* **Learning Rate**: 
    * *Base Training*: 1e-3
    * *Fine Tuning*: 1e-4 (dengan ReduceLROnPlateau)
* **Loss Function**: Categorical Crossentropy
* **Batch Size**: 16
* **Image Size**: 224 x 224 pixel
