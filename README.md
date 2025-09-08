![Chest X-Ray](https://www.surgeryjournal.co.uk/cms/10.1016/j.mpsur.2009.08.006/asset/1787dabd-923a-4e4b-8c86-a149332a37a6/main.assets/gr2.jpg)

# 🩺 Pneumonia Detection from Chest X-Rays using Deep Learning

This project builds a **Convolutional Neural Network (CNN)** using **Transfer Learning (MobileNetV2)** to classify chest X-ray images into **NORMAL** or **PNEUMONIA**.  
The model was trained on the **Chest X-Ray Images (Pneumonia)** dataset and achieved high accuracy on unseen test images.  

---

## 📌 Project Overview
- **Dataset:** [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- **Classes:**  
  - ✅ NORMAL  
  - ❌ PNEUMONIA  
- **Techniques Used:**  
  - Image preprocessing (resizing, normalization)  
  - Handling class imbalance by undersampling  
  - Data augmentation (rotation, flipping, zoom, shifts)  
  - Transfer Learning with **MobileNetV2**  
  - EarlyStopping to prevent overfitting  

---

## 📂 Workflow
1. **Load Dataset** (train/val/test with NORMAL & PNEUMONIA folders)  
2. **Balance Dataset** (undersample pneumonia images to match normal images)  
3. **Convert Images to Arrays + Normalize** (scale pixels to 0–1)  
4. **Split into Train/Validation/Test sets**  
5. **Data Augmentation** with `ImageDataGenerator`  
6. **Model Building**  
   - Pretrained **MobileNetV2** (frozen base)  
   - Added custom dense layers for binary classification  
7. **Model Training** with `EarlyStopping`  
8. **Evaluation** (achieved **~90% Test Accuracy**)  
9. **Prediction System** (takes X-ray image input and predicts NORMAL or PNEUMONIA)  

---

## 📊 Results
- **Training Accuracy:** ~92%  
- **Validation Accuracy:** ~85–92%  
- **Test Accuracy:** **90.32%**  

The model shows strong generalization ability on unseen test images.

---

## ⚙️ Tech Stack
- **Python**  
- **TensorFlow / Keras**  
- **NumPy, Matplotlib**  
- **PIL (Image Processing)**  

---

## 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Run prediction
python predict.py --image path/to/xray_image.jpg
