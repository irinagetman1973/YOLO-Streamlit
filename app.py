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

    uploaded_file = st.sidebar.file_uploader(label="Choose a file...")

        # Check if a file has been uploaded or instructions have been displayed before:
    if uploaded_file is None :
            st.session_state['instructions_displayed'] = True  # Set to True so instructions are not displayed again
            st.markdown("""
            <style>
                .green {
                    color: green;
                    font-weight: bold;
                    font-size: larger;
                }
            </style>

            #  <span class="green">_**YOLO**_</span> _models evaluation app_

            Welcome to the YOLO models evaluation app! This platform allows you to test and evaluate different YOLOv8 models on your own data. 
            Here's how to get started:

            **Step 1: Choose a YOLOv8 Model:**
            In the sidebar, you'll find a dropdown box where you can select from different YOLOv8 models: 8l, 8m, 8s, 8x, and 8n. 
            Here's a brief on what differentiates them (as per Ultralytics documentation):
            
            - **8l:** This model has a larger size and offers higher accuracy.
            - **8m:** A medium-sized model offering a good balance between size, speed, and accuracy.
            - **8s:** A smaller model, faster but with slightly lower accuracy.
            - **8x:** An extended model with more layers, providing higher accuracy but at the cost of speed.
            - **8n:** A nominal model that provides a balance between size and accuracy.

            **Step 2: Adjust Confidence Score:**
            Adjust the confidence score using the slider in the sidebar. A higher confidence score will result in fewer detections but with higher certainty.

            **Step 3: Select Data Type:**
            Choose the type of data you'd like the model to process: image, video, or webcam feed, from the dropdown box.

            **Step 4: Upload Your File:**
            Use the file uploader to select the file from your local machine.

            Explore and have fun with real-time object detection!
            
            """, unsafe_allow_html=True)

    # if uploaded_file:
    #     if source_selectbox == config.SOURCES_LIST[0]:  # Image
    #         infer_uploaded_image(confidence, model)
    #     elif source_selectbox == config.SOURCES_LIST[1]:  # Video
    #         infer_uploaded_video(confidence, model)
    #     else:
    #         st.error("Currently only 'Image' and 'Video' source are implemented")
    if uploaded_file:
        if source_selectbox == config.SOURCES_LIST[0]:  # Image
            infer_uploaded_image(confidence, model, uploaded_file)
        elif source_selectbox == config.SOURCES_LIST[1]:  # Video
            infer_uploaded_video(confidence, model, uploaded_file)
    # else:
    #     st.error("Please upload a file.")

    


# if __name__ == "__main__":
#     main()

