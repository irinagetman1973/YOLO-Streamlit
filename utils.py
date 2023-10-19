#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Authors:        Luyao.zhang, irinagetman1973
   @Date:          29/06/2023
   @Description:
-------------------------------------------------
"""
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import streamlit_scrollable_textbox as stx
import tempfile
import config
import numpy as np

import os
import datetime
import io
import base64
import plotly.graph_objects as go
import os
import moviepy.editor as mpy

os.environ['TMPDIR'] = 'C:\\Users\\irina\\tmp'
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = '0'

# Function to create a figure for Plotly
def create_fig(image, detected=False):
    # Convert the image to a data URI
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_data_uri = base64.b64encode(buffer.getvalue()).decode()
    
    # Create a figure
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{image_data_uri}",
            x=0,
            y=image.size[1],
            xref="x",
            yref="y",
            sizex=image.size[0],
            sizey=image.size[1],
            layer="below"
        )
    )
    
    fig.update_layout(
        xaxis_range=[0, image.size[0]],
        yaxis_range=[0, image.size[1]],
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=True),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=True),
        annotations=[
            dict(
                x=0.5,
                y=-0.1,
                showarrow=False,
                text="Detected Image" if detected else "Original Image",
                xref="paper",
                yref="paper"
            )
        ]
    )
    
    return fig

def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Assuming the 'plot' method of the result object 'res' creates an image with detected objects plotted
    res_plotted = res[0].plot()

    # Display the frame with detected objects
    st_frame.image(res_plotted,
                #    caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    pass

@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def scale_image(image, max_width=596, max_height=400):
    """
    Scale the input image while maintaining the aspect ratio, only if 
    the width is greater than max_width or the height is greater than max_height.
    :param image: Input image.
    :param max_width: The maximum width for the image.
    :param max_height: The maximum height for the image.
    :return: Scaled image.
    """
    width, height = image.size
    
   
    # Only scale if either dimension is larger than the max dimensions
    if width > max_width or height > max_height:
        # Determine the scaling factor such that the aspect ratio is maintained
        scaling_factor = min(max_width / width, max_height / height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)

        # Resize the image
        
        return image.resize((new_width, new_height))

    # If no scaling is needed, return the original image
    return image



def infer_uploaded_image(conf, model, uploaded_file):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    if uploaded_file is not None:
    
        col1, col2 = st.columns(2)
        boxes = None

        with col1:
            if uploaded_file:
                uploaded_image = Image.open(uploaded_file)
                fig_orig = create_fig(uploaded_image)

                # Display the original image using Plotly
                st.plotly_chart(fig_orig, use_container_width=True)
                st.markdown("**Original Image**")
            if  uploaded_file:
                st.markdown("""
                ### Instructions:
    1. **Original Image:** The left side displays the original image you provided.
    2. **Detected Image:** The right side showcases the image with highlighted detected objects.
    - Zoom in/out and pan the image to see detections more clearly.
    - View detection results displayed under the detected image, including the class of detected objects, 
                            their coordinates on the image, probability score, and the total number of different detected objects.

        """)
                if st.button("Execution"):
                    with st.spinner("Running..."):
                        detected_image = model.predict(uploaded_image, conf=conf)
                        boxes = detected_image[0].boxes

                    with col2:
                        if boxes:
                            # Get the plotted image with detections from the Results object
                            detected_img_arr = detected_image[0].plot()[:, :, ::-1]  # Assuming this returns a numpy array
                            # Convert the numpy array to a PIL Image object
                            detected_image = Image.fromarray(cv2.cvtColor(detected_img_arr, cv2.COLOR_BGR2RGB))

                            # Now pass the image object to create_fig()
                            fig_detected = create_fig(detected_image, detected=True)

                            # Display the detected image using Plotly
                            st.plotly_chart(fig_detected, use_container_width=True)
                            st.markdown("**Detected Image**")

                            if boxes :
                                
                                detection_results = ""
                                count_dict = {}
                                for box in boxes:
                                        class_id = model.names[box.cls[0].item()]
                                        cords = box.xyxy[0].tolist()
                                        cords = [round(x) for x in cords]
                                        conf = round(box.conf[0].item(), 2)

                                        detection_results += f"<b style='color: blue;'>Object type:</b> {class_id}<br><b style='color: blue;'>Coordinates:</b> {cords}<br><b style='color: blue;'>Probability:</b> {conf}<br>---<br>"
                                        if class_id in count_dict:
                                            count_dict[class_id] += 1
                                        else:
                                            count_dict[class_id] = 1
                                for object_type, count in count_dict.items():
                                        detection_results += f"<b style='color: blue;'>Count of {object_type}:</b> {count}<br>"
                            
                        
                                scrollable_textbox = f"""
                                    <div style="
                                        font-family: 'Source Code Pro','monospace';
                                        font-size: 16px;
                                        overflow-y: scroll;
                                        border: 1px solid #000;
                                        padding: 10px;
                                        width: 500px;
                                        height: 400px;
                                    ">
                                        {detection_results}
                                    </div>
                                """

                            # Display the scrollable textbox using st.markdown
                                st.markdown("""### Results:""")
                                st.markdown(scrollable_textbox, unsafe_allow_html=True)

                        else:
                            st.markdown("""### No objects detected""")
                            st.markdown("""
                                The model did not detect any objects in the uploaded image.
                                Please try with a different image or adjust the model's 
                                confidence threshold in the sidebar and try again.
                            """)        


def infer_uploaded_video(conf, model,uploaded_file):
    

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        cl1, cl2, cl3, cl4 = st.columns([0.25, 0.1, 0.25, 0.4])

        with cl1:
            col1_width = st.slider('Adjust input video size', min_value=0.0, max_value=0.45, value=0.25, step=0.05)

        remaining_width = 1.0 - col1_width * 2

        if remaining_width < 0:
            st.error("The input video size is too large. Please adjust the size to fit the screen.")
            return

        col2_width = col4_width = remaining_width / 2

        col1, col2, col3, col4 = st.columns([col1_width, col2_width, col1_width, col4_width])

        with col1:
            st.video(tfile.name)
            execution_button = st.button("Execution")

        if 'execution_completed' not in st.session_state:
            st.session_state['execution_completed'] = False

        if 'play_button_clicked' not in st.session_state:
            st.session_state['play_button_clicked'] = False


        with col3:
            st_frame = st.empty()  
            temp_output_file = None  

            if execution_button:  
                with st.spinner("Execution in progress. Please wait..."):
                    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vid_cap = cv2.VideoCapture(tfile.name)
                    frame_width, frame_height = int(vid_cap.get(3)), int(vid_cap.get(4))
                    out = cv2.VideoWriter(temp_output_file.name, fourcc, vid_cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

                    progress_bar = st.progress(0)
                    
                    frame_rate = vid_cap.get(cv2.CAP_PROP_FPS)
                    st.info(f"This video frame rate: {frame_rate} frames per second")  # Display frame rate

                    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    for frame_num in range(total_frames):
                        success, image = vid_cap.read()
                        if success:
                            processed_frame = _display_detected_frames(conf, model, st_frame, image)
                            out.write(processed_frame)
                            progress_bar.progress((frame_num + 1) / total_frames)
                        else:
                            break

                    processed_frames_info = f"Total Frames: {total_frames}\nProcessed Frames: {frame_num + 1}"
                    st.info(processed_frames_info)  # Display total and processed frames count

                    out.release()  # Release the VideoWriter
                    vid_cap.release()  # Release the VideoCapture

                    st.success('Execution complete!')
                    st.balloons()

                    st.session_state['execution_completed'] = True
        
   

                 



def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_count = st.empty()
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_count,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
