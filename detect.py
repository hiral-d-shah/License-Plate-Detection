import streamlit as st
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
import tempfile

@st.cache_resource
def load_model():
	model = YOLO('yolo11n-custom.pt')
	model.fuse()
	return model

model = load_model()

reader = easyocr.Reader(['en'])

def detect_license_plate(image):
	results = model.predict(image, conf=0.15, iou=0.3, classes=[0])
	plate_texts = []
	img_array = np.array(image)
	# img = cv2.imread(image_path)
	img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	img_height, img_width, _ = img.shape

	for result in results:
		for bbox in result.boxes.xyxy:
			x1, y1, x2, y2 = map(int, bbox.tolist())
			plate = img[int(y1):int(y2), int(x1):int(x2)]
			scale=2
			height, width = plate.shape[:2]
			plate = cv2.resize(plate, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
			
			text = reader.readtext(plate, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-")
			text = " ".join(text).upper()
			
			text_scale = max(1, width / 250)
			thickness = max(2, width // 200)
			cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
			(text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
			text_x = x1 + (width - text_width) // 2  # Centered horizontally
			text_y = y1 - 10 if y1 > 50 else y2 + text_height + 20  # Above unless too high
			text_box_y1 = text_y - text_height - 5
			text_box_y2 = text_y + 5
			cv2.rectangle(img, (text_x - 8, text_box_y1 - 3), (text_x + text_width + 8, text_box_y2 + 3), (0, 0, 0), -1)
			cv2.rectangle(img, (text_x - 5, text_box_y1), (text_x + text_width + 5, text_box_y2), (255, 255, 255), -1)
			cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), thickness)

			plate_texts.append(text)
	image = img
	return image, plate_texts

st.title("🚘 Real-Time License Plate Detection", anchor=False)

st.write("For better license plate detection, ensure you use high-quality images. If detection is unclear, try enhancing the image first. Use the Refine Image for Detection tool.")

st.write("Upload an image, upload a video, or use your webcam for real-time license plate detection.")

option = st.radio("Choose Input Source:", ("Upload Image", "Upload Video", "Webcam"), horizontal=True )


if option == "Upload Image":
	uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
	if uploaded_file:
		img = Image.open(uploaded_file)
		st.write("Processing...")

		processed_img, plate_texts = detect_license_plate(img)

		st.image(processed_img, caption="Detected Plates Image", use_container_width=True)
		st.write("**Detected License Plates:**")
		if plate_texts:
			plates = pd.DataFrame({"License Plate": plate_texts})
			plates.index = plates.index + 1
			st.table(plates)
		else:
			st.write("No license plates detected.")

elif option == "Upload Video":
	uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
	if uploaded_video is not None:
		st.write("Processing video...")
		tfile = tempfile.NamedTemporaryFile(delete=False)
		tfile.write(uploaded_video.read())
		cap = cv2.VideoCapture(tfile.name)
		frame_placeholder = st.empty()

		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break
			processed_frame, plate_texts = detect_license_plate(frame)

			frame_placeholder.image(processed_frame, caption="Detected Plates Video", use_container_width=True)
		cap.release()

elif option == "Webcam":
	if "running" not in st.session_state:
		st.session_state.running = True
	if st.button("Stop"):
		st.session_state.running = False

	st.write("Starting Webcam... Press **Stop** to end.")
	cap = cv2.VideoCapture(0)
	frame_placeholder = st.empty()

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			st.warning("Failed to capture webcam feed.")
			break

		processed_frame, plate_texts = detect_license_plate(frame)

		frame_placeholder.image(processed_frame, channels="BGR", caption="Webcam Feed", use_container_width=True)

		if not st.session_state.running:
			break

	cap.release()