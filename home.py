import streamlit as st

st.title("Welcome to License Plate Detection App", anchor=False)
st.write("Select a feature below to detect or enhance license plate images.")

# Buttons for navigation
col1, col2 = st.columns(2)

with col1:
    if st.button("🚘 Detect License Plates"):
        st.switch_page("detect.py")  # Navigates to the detection page
    st.caption("Upload an image, video, or use a webcam for real-time detection.")

with col2:
    if st.button("🖼️ Enhance Image for Detection"):
        st.switch_page("refine.py")  # Navigates to the image enhancement page
    st.caption("Improve image quality for better license plate recognition.")