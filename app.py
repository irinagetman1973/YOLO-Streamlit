from pathlib import Path
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam
from auth import display_authentication_page
from dashboard import display_dashboard


# setting page layout
st.set_page_config(
    page_title="Vehicle Tracking with YOLOv8",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
    )


#    ------# Initialize session state variables------------
if 'page' not in st.session_state:
    st.session_state.page = 'main'


def main():
    # Set default page if not set
    if 'page' not in st.session_state:
        st.session_state.page = 'main'
    
    # Check the current page and display content accordingly
    if st.session_state.page == 'main':
        display_main_page()
    elif st.session_state.page == 'authentication':
        display_authentication_page()
    elif st.session_state.page == 'dashboard':
        display_dashboard()  
    display_sidebar()
    

def display_sidebar():   

    if not st.session_state.get('authenticated', False):
        if st.sidebar.button("Login/Sign Up", key="login_signup_button"):
            st.session_state.page = 'authentication'
            st.experimental_rerun()
    else:
        if st.sidebar.button("Dashboard"):
            st.session_state.page = 'dashboard'
            st.experimental_rerun()


def display_main_page():
    st.title("Welcome")
    st.header("Detect with _Confidence_, Powered by **:green[YOLO]**!")

    # Sidebar
    st.sidebar.header("YOLO Models")

    # if not st.session_state.get('authenticated', False):
    #     if st.sidebar.button("Login/Sign Up"):
    #         st.session_state.page = 'authentication'  # Redirect to authentication page
    #         return  # Exit the function immediately after setting the page
    # else:
    #     if st.sidebar.button("Dashboard"):
    #         st.session_state.page = 'dashboard'
    #         return  # Exit the function immediately after setting the page

    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
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



if __name__ == "__main__":
    main()










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

