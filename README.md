# 🧠 Enhanced CNN-BiLSTM with Random Forest for ASD Detection

This project presents a hybrid deep learning and machine learning approach for the **early detection of Autism Spectrum Disorder (ASD)** using fMRI image data. It combines **Convolutional Neural Networks (CNN)** and **Bidirectional LSTM (BiLSTM)** for feature extraction and uses a **Random Forest classifier** for final classification, enhancing both performance and interpretability.

---

## 📌 Objective

To improve ASD detection accuracy by integrating deep feature extraction from CNN-BiLSTM with the powerful ensemble learning capability of Random Forest, thereby achieving better classification results on fMRI-based datasets.

---

## 🔍 Key Features

- 🧠 CNN for spatial feature extraction from brain images
- 🔁 BiLSTM for learning temporal dependencies in sequential data
- 🌲 Random Forest for robust and interpretable classification
- 📈 Evaluation with metrics: Accuracy, Precision, Recall, F1-Score
- 📊 Visualizations: Training history, confusion matrix, ROC curve

---

## 🧬 Dataset

- **Source**: ABIDE I Dataset (Autism Brain Imaging Data Exchange)
- **Data type**: Preprocessed 2D fMRI image slices
- **Classes**: ASD (Autism Spectrum Disorder) vs TD (Typically Developing)

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

---

## 📁 Files

- `ASD_Detection_Source_code.ipynb` – Main notebook with model implementation and results
- `README.md` – Project overview and setup guide
- (Optional) `models/`, `data/` – Add relevant folders for saved models and datasets

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Enhanced-CNN-BiLSTM-ASD-Detection.git
cd Enhanced-CNN-BiLSTM-ASD-Detection
pip install -r requirements.txt
jupyter notebook ASD_Detection_Source_code.ipynb

| Metric    | Value (%) |
| --------- | --------- |
| Accuracy  | \~98%     |
| Precision | High      |
| Recall    | High      |
| F1-Score  | High      |
