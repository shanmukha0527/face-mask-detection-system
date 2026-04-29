# 😷 Face Mask Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

A deep learning-based **Face Mask Detection System** that identifies whether a person is wearing a mask or not using image input and real-time webcam.

Built using **MobileNetV2 (Transfer Learning)** and **OpenCV DNN face detection**.

---

## 🚀 Features

* 📸 Mask detection from uploaded images
* 🎥 Real-time detection using webcam
* 🧠 Transfer learning with MobileNetV2
* 📊 Accuracy and loss visualization
* ⚡ Fast and efficient performance

---

## 🎥 Demo

### 🟢 With Mask Detection

![With Mask](Screenshot%202026-04-29%20081227.png)

### 🔴 Without Mask Detection

![No Mask](Screenshot%202026-04-29%20080734.png)

### 🌐 Streamlit Web App Interface

![UI](Screenshot%202026-04-29%20081035.png)

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* Streamlit
* NumPy, Matplotlib, Scikit-learn

---

## 📂 Project Structure

```bash id="p4u2hk"
face-mask-detection-system/
│── app.py
│── detect_realtime.py
│── train_model.py
│── requirements.txt
│── README.md
│── mask_detector.h5
│── deploy.prototxt
│── res10_300x300_ssd_iter_140000.caffemodel
│── screenshots/
│── .gitignore
│── LICENSE
```

---

## 📊 Model Performance

* Training Accuracy: ~94%
* Validation Accuracy: ~96%

---

## 📁 Dataset

Dataset used: **Face Mask Detection Dataset (Kaggle)**

* Total Images: 7553
* Classes:

  * 😷 With Mask
  * ❌ Without Mask

👉 https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

---

## ⚙️ Installation

```bash id="pj0wya"
git clone https://github.com/shanmukha0527/face-mask-detection-system.git
cd face-mask-detection-system
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Run Streamlit Web App

```bash id="czkmbx"
streamlit run app.py
```

Upload image → get prediction instantly.

---

### 🔹 Run Real-Time Detection

```bash id="24i45z"
python detect_realtime.py
```

Press **Q** to exit.

---

### 🔹 Train Model (Optional)

```bash id="4e4xdt"
python train_model.py
```

---

## ⚡ How It Works

1. Detect face using OpenCV DNN
2. Extract face region
3. Resize to 224×224
4. MobileNetV2 predicts mask / no mask
5. Display result with bounding box

---

## ⚠️ Important Notes

* ❌ Do NOT upload `kaggle.json`
* Dataset is not included in repo
* Model file (`.h5`) is pre-trained

---

## 📌 Future Improvements

* 🔔 Alert system for no-mask detection
* 📊 Count number of people without mask
* 🌐 Deploy on Streamlit Cloud
* 📱 UI improvements

---

## 👨‍💻 Author

**Gondrala Shanmukha Akhilesh**
B.Tech CSE (AI & ML)
