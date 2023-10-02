from pathlib import Path
import streamlit as st
import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

from dashboard import display_dashboard

from au import authenticate



def display_authentication_page():
    # Call the authenticate function from au.py
    authenticate()

    




def display_main_page():
    st.title("Welcome")
    st.header("Detect with _Confidence_, Powered by **:green[YOLO]**!")

    # Sidebar
    st.sidebar.header("YOLO Models")

    

    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST,
        key='models_selectbox'
    )

    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 30, 100, 50)) / 100

    model_path = ""
    if model_type:
        model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
    else:
        st.error("Please Select Model in Sidebar")

    #--------------- Load pretrained DL model------------
    
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Unable to load model. Please check the specified path: {model_path}")

    #--------------Image/video options-------------------

    st.sidebar.header("Image/Video Upload")
    source_selectbox = st.sidebar.selectbox(
        "Select Source",
        config.SOURCES_LIST
    )

    if source_selectbox == config.SOURCES_LIST[0]:  # Image
        infer_uploaded_image(confidence, model)
    elif source_selectbox == config.SOURCES_LIST[1]:  # Video
        infer_uploaded_video(confidence, model)
    elif source_selectbox == config.SOURCES_LIST[2]:  # Webcam
        infer_uploaded_webcam(confidence, model)
    else:
        st.error("Currently only 'Image' and 'Video' source are implemented")



# if __name__ == "__main__":
#     main()










# def main_page():
#     st.title("Welcome")
#     st.header("Detect with _Confidence_, Powered by **:green[YOLO]**!")
    
#     # Sidebar
#     st.sidebar.header("YOLO Models")
    
#     if not st.session_state.get('authenticated', False):
#         if st.sidebar.button("Login/Sign Up"):
#             st.session_state.page = 'authentication'  # Redirect to authentication page
#     else:
#         if st.sidebar.button("Dashboard"):
#             st.session_state.page = 'dashboard'
    
#     model_type = st.sidebar.selectbox(
#         "Select Model",
#         config.DETECTION_MODEL_LIST
#     )

#     confidence = float(st.sidebar.slider(
#         "Select Model Confidence", 30, 100, 50)) / 100

#     model_path = ""
#     if model_type:
#         model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
#     else:
#         st.error("Please Select Model in Sidebar")

#     # Load pretrained DL model
#     try:
#         model = load_model(model_path)
#     except Exception as e:
#         st.error(f"Unable to load model. Please check the specified path: {model_path}")

#     # Image/video options
#     st.sidebar.header("Image/Video Upload")
#     source_selectbox = st.sidebar.selectbox(
#         "Select Source",
#         config.SOURCES_LIST
#     )

#     if source_selectbox == config.SOURCES_LIST[0]:  # Image
#         infer_uploaded_image(confidence, model)
#     elif source_selectbox == config.SOURCES_LIST[1]:  # Video
#         infer_uploaded_video(confidence, model)
#     elif source_selectbox == config.SOURCES_LIST[2]:  # Webcam
#         infer_uploaded_webcam(confidence, model)
#     else:
#         st.error("Currently only 'Image' and 'Video' source are implemented")

# def main():
#     # Display content based on session state
#     if st.session_state.page == 'main':
#         main_page()
#     elif st.session_state.page == 'authentication':
#         display_authentication_page()  # Call the function from the other file

# if __name__ == "__main__":
#     main()

