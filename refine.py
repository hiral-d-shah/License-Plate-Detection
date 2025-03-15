import streamlit as st
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
import tempfile

# Upload image
@st.cache_resource
def load_model():
	model = YOLO('yolo11n-custom.pt')
	model.fuse()
	return model

model = load_model()

reader = easyocr.Reader(['en'])
def apply_filters(image, noise, sharpen, grayscale, threshold, edges, invert, auto, blur, contrast, brightness, scale, denoise, hist_eq, gamma, clahe):
	img = np.array(image)
	
	# Auto Enhancement
	if auto:
		lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
		l, a, b = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
		l = clahe.apply(l)
		img = cv2.merge((l, a, b))
		img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)


	# Scaling
	if scale != 1.0:
		height, width = img.shape[:2]
		img = cv2.resize(img, (int(width * scale), int(height * scale)))
	
	# Noise Reduction (Bilateral Filtering)
	if noise:
		img = cv2.bilateralFilter(img, 9, 75, 75)

	# Noise Reduction (Non-Local Means)
	if denoise:
		img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
	
	# Sharpening
	if sharpen:
		kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
		img = cv2.filter2D(img, -1, kernel)
	
	# Convert to Grayscale
	if grayscale or threshold or hist_eq or clahe:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	# Histogram Equalization
	if hist_eq:
		img = cv2.equalizeHist(img)
	
	# CLAHE (Contrast Limited Adaptive Histogram Equalization)
	if clahe:
		clahe_filter = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
		img = clahe_filter.apply(img)
	
	# Adaptive Thresholding
	if threshold:
		img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	
	# Edge Detection
	if edges:
		img = cv2.Canny(img, 100, 200)
	
	# Invert Colors
	if invert:
		img = cv2.bitwise_not(img)
	
	# Blur
	if gamma != 1.0:
		inv_gamma = 1.0 / gamma
		table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
		img = cv2.LUT(img, table)

	# Blur
	if blur:
		img = cv2.GaussianBlur(img, (2*blur + 1, 2*blur + 1), 0)
	
	# Contrast & Brightness
	if contrast != 1.0 or brightness != 0:
		img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
	
	return img

st.title("🖼️ Refine Image for Detection")
st.write("Enhance the license plate image for better recognition.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
	# Read image
	img = Image.open(uploaded_file)
	img = np.array(img)

	# Detect license plates
	st.write("🔍 Detecting license plates...")
	results = model.predict(img, conf=0.15, iou=0.3, classes=[0])  
	plates = results[0].boxes.xyxy if len(results) > 0 else []

	if len(plates) == 0:
		st.error("❌ No license plates detected. Try another image.")
	else:
		st.write("📌 **Select a License Plate by Clicking Below**")
		
		# Show detected plates in a grid
		if "selected_plate_index" not in st.session_state:
			st.session_state.selected_plate_index = 0

		selected_plate_index = st.session_state.get("selected_plate_index", 0)
		cols = st.columns(len(plates))  # Create dynamic columns

		for i, (x1, y1, x2, y2) in enumerate(plates):
			plate_img = img[int(y1):int(y2), int(x1):int(x2)]
			plate_img = Image.fromarray(plate_img)

			with cols[i]:  # Place each image in a column
				st.image(plate_img, caption=f"Plate {i+1}", use_container_width =True)
				if st.button(f"Select Plate {i+1}", key=f"plate_{i}"):
					st.session_state["selected_plate_index"] = i

		# Get the selected plate
		selected_index = st.session_state["selected_plate_index"]
		x1, y1, x2, y2 = map(int, plates[selected_index])
		cropped_plate = img[y1:y2, x1:x2]
		refined_img = cropped_plate.copy()

		# Sidebar for enhancements
		st.sidebar.header("🔧 Enhancement Options")
		blur = st.sidebar.slider("🔹 Blur", 0, 10, 0)
		contrast = st.sidebar.slider("🔹 Contrast", 0.5, 2.0, 1.0)
		brightness = st.sidebar.slider("🔹 Brightness", 0.5, 2.0, 1.0)
		gamma = st.sidebar.slider("Gamma Correction", 0.1, 3.0, 1.0, 0.1)
		scale = st.sidebar.slider("🔹 Scale", 1.0, 10.0, 5.0)
		noise = st.sidebar.checkbox("Noise Reduction (Bilateral)")
		denoise = st.sidebar.checkbox("Denoise (Non-Local Means)")
		sharpen = st.sidebar.checkbox("Sharpening")
		hist_eq = st.sidebar.checkbox("Histogram Equalization")
		clahe = st.sidebar.checkbox("CLAHE (Advanced Contrast)")
		grayscale = st.sidebar.checkbox("Grayscale Conversion")
		threshold = st.sidebar.checkbox("Adaptive Thresholding")
		edges = st.sidebar.checkbox("Edge Detection")
		invert = st.sidebar.checkbox("Invert Colors")
		auto = st.sidebar.checkbox("Auto Enhancement")

		refined_img = apply_filters(refined_img, noise, sharpen, grayscale, threshold, edges, invert, auto, blur, contrast, brightness, scale, denoise, hist_eq, gamma, clahe)

		st.image(refined_img, caption="Refined License Plate", use_container_width=True)

		if st.button("📖 Detect License Plate Text"):
			with st.spinner("🔎 Reading text..."):
				ocr_result = reader.readtext(np.array(refined_img), detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-")
				plate_text = " ".join(ocr_result).upper() if ocr_result else "❌ No text detected."
			
			# Show detected text
			st.subheader("📜 Detected License Plate:")
			st.code(plate_text, language="plaintext")