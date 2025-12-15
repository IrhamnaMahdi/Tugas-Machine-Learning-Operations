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
- green_parrot/
- gray_parrot/
- macaw/
- white_parrot/
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
