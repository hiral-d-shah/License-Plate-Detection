# 🚘 License Plate Detection & Refinement

A **Streamlit-based application** for detecting and refining license plate images using **YOLO for detection** and **OpenCV for image enhancements**.

[Demo](https://huggingface.co/spaces/HiralShah62/License-Plate-Detection-System)

---

## 🔧 Features

### **🚘 Real-Time License Plate Detection**
- Upload an **image** or **video**, or use a **webcam** for real-time detection.
- Uses **YOLOv11** for accurate license plate detection.
- Extracts text using **EasyOCR**.
- Displays detected plates in a structured table.

### **🛠 Refine Image for Detection**
- Upload an image and apply **pre-processing techniques** to enhance detection.
- Select from detected license plates to refine specific sections.
- **Image enhancement options:**
  - **Noise Reduction** (Bilateral & Non-Local Means Denoising)
  - **Sharpening** (Enhances edges for better OCR accuracy)
  - **Grayscale Conversion** (Improves contrast)
  - **Adaptive Thresholding** (Dynamic brightness adjustment)
  - **Edge Detection** (Canny filter for better boundary detection)
  - **Invert Colors** (Helpful for light text on dark backgrounds)
  - **Auto Enhancement** (Applies recommended adjustments automatically)
  - **Gamma Correction, CLAHE, Histogram Equalization, Scaling, Blur, Contrast & Brightness adjustments**

---

## 📌 How to Run Locally

### **1️⃣ Install dependencies**
```sh
pip install -r requirements.txt
```

### **2️⃣ Run the application**
```sh
streamlit run app.py
```

---

## 📁 Project Structure
```
📂 License-Plate-Detection
│── app.py                 # Main navigation & home page
│── pages/
│   ├── detect.py          # License plate detection module
│   ├── refine.py          # Image enhancement module
│── requirements.txt       # Required Python packages
│── README.md              # Project documentation
```

---

## 🚀 Deployment
You can deploy this app on **Streamlit Cloud** or **Hugging Face Spaces**:

1. **Fork & Push to GitHub**
2. **Deploy via Streamlit Sharing / Hugging Face Spaces**

[![Deploy on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-on-spaces-md.svg)]([your-huggingface-link](https://huggingface.co/spaces/HiralShah62/License-Plate-Detection-System))

---

## 📌 Future Enhancements
- **Real-time tracking of license plates**.
- **OCR post-processing to correct errors**.
- **Additional denoising & image restoration techniques**.

---

## 📝 License
This project is **open-source** under the **MIT License**. Feel free to modify and improve it!

🌟 If you like this project, don’t forget to **star the repository**!

