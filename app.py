from pathlib import Path
import streamlit as st
import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video
from streamlit_lottie import st_lottie
import requests

import json
from au import authenticate



def display_authentication_page():
    # Call the authenticate function from au.py
    authenticate()

 #--------------Lottie Animation Loader-------------------

def load_lottie_url(url:str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()    

def load_lottie_file(file_path:str):
    with open(file_path, "r") as file:
        return json.load(file)




def display_main_page():
    
    # st.image('images/banner.jpg', use_column_width=True)
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
            st.session_state['instructions_displayed'] = True  
            
            lottie = """
            <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
            <lottie-player src="https://raw.githubusercontent.com/irinagetman1973/YOLO-Streamlit/main/animation_sphere.json" background="transparent" speed="1" style="width: 400px; height: 400px;" loop autoplay></lottie-player>
            """
            st.markdown("""
                <style>
                    iframe {
                        position: fixed;
                        top: 16rem;
                        bottom: 0;
                        left: 1205;
                        right: 0;
                        margin: auto;
                        z-index=-1;
                    }
                </style>
                """, unsafe_allow_html=True
            )


            st.components.v1.html(lottie, width=410, height=410)

            st.title("Welcome to the :green[**_YOLO_**] models evaluation app!")
            st.divider()

            col1, col2 = st.columns([0.7, 0.3])  # Create a 2-column layout with equal width for each column

            with col1:
                    
                  # Title
                    

                    # Step 1
                    st.write(":one: :blue[**Choose a YOLOv8 Model:**]")
                    st.write("""
                    In the sidebar, you'll find a dropdown box where you can select from different YOLOv8 models:
                    - :green[**8l:**] This model has a larger size and offers higher accuracy.
                    - :green[**8m:**] yA medium-sized model offering a good balance between size, speed, and accuracy.
                    - :green[**8s:**] A smaller model, faster but with slightly lower accuracy.
                    - :green[**8x:**] An extended model with more layers, providing higher accuracy but at the cost of speed.
                    - :green[**8n:**] A nominal model that provides a balance between size and accuracy.
                    """)
                    
                    st.divider()

                    # Step 2
                    st.write(":two: :blue[**Adjust Confidence Score:**]")
                    st.write("Adjust the confidence score using the slider in the sidebar. A higher confidence score will result in fewer detections but with higher certainty.")

                    st.divider()

                    # Step 3
                    st.write(":three: :blue[**Select Data Type:**]")
                    st.write("Choose the type of data you'd like the model to process: image or video from the dropdown box.")
                    st.divider()

                    # Step 4
                    st.write(":four: :blue[**Upload Your File:**]")
                    st.write("Use the file uploader to select the file from your local machine.")
                    st.divider()

                    st.write("Explore and have fun with real-time object detection! :green_heart:")

                    st.divider()

                    


            
                
                
                
                
                
             
   
    if uploaded_file:
        if source_selectbox == config.SOURCES_LIST[0]:  # Image
            infer_uploaded_image(confidence, model, uploaded_file)
        elif source_selectbox == config.SOURCES_LIST[1]:  # Video
            infer_uploaded_video(confidence, model, uploaded_file)
    # else:
    #     st.error("Please upload a file.")

    


# if __name__ == "__main__":
#     main()

