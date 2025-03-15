import streamlit as st

pages = [
    st.Page("home.py", title="Home", icon="🏠"),
    st.Page("detect.py", title="Real-Time License Plate Detection", icon="🚘"),
    st.Page("refine.py", title="Refine Image for Detection", icon="🖼️")
]
nav = st.navigation(pages, position="sidebar", expanded=False)
nav.run()