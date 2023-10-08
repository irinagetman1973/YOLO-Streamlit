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
from results import Results
import os
import datetime
import io
import base64
import plotly.graph_objects as go
import os


os.environ['TMPDIR'] = 'C:\\Users\\irina\\tmp'

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

def _display_detected_frames(conf, model, st_count, st_frame, image):
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
    
    inText = 'Vehicle In'
    outText = 'Vehicle Out'
    if config.OBJECT_COUNTER1 != None:
        for _, (key, value) in enumerate(config.OBJECT_COUNTER1.items()):
            inText += ' - ' + str(key) + ": " +str(value)
    if config.OBJECT_COUNTER != None:
        for _, (key, value) in enumerate(config.OBJECT_COUNTER.items()):
            outText += ' - ' + str(key) + ": " +str(value)
    
    # Plot the detected objects on the video frame
    st_count.write(inText + '\n\n' + outText)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


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



def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )
    st.markdown("""
      ### Instructions:
      1. **Original Image:** On the left is the original image you provided.
      2. **Detected Image:** On the right is the image with detected objects highlighted.
         
         * You can zoom in/out and pan the image to see detections more clearly.
      """)
    col1, col2 = st.columns(2)
    boxes = None

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            fig_orig = create_fig(uploaded_image)

            # Display the original image using Plotly
            st.plotly_chart(fig_orig, use_container_width=True)
            st.markdown("**Original Image**")
        if  source_img:
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

                    detection_results += f"<b>Object type:</b> {class_id}<br><b>Coordinates:</b> {cords}<br><b>Probability:</b> {conf}<br>---<br>"
                    if class_id in count_dict:
                        count_dict[class_id] += 1
                    else:
                        count_dict[class_id] = 1
            for object_type, count in count_dict.items():
                    detection_results += f"<b>Count of {object_type}:</b> {count}<br>"

            scrollable_textbox = f"""
                <div style="
                    font-family: 'monospace';
                    overflow-y: scroll;
                    border: 1px solid #000;
                    padding: 10px;
                    width: 595px;
                    height: 300px;
                ">
                    {detection_results}
                </div>
            """

        # Display the scrollable textbox using st.markdown
            st.markdown(scrollable_textbox, unsafe_allow_html=True)





def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    col1, col2 = st.columns(2)
    with col1:
            st.video(source_video)
     
     

    with col2:

        if source_video:
            if st.button("Execution"):
                with st.spinner("Running..."):
                    try:
                        # config.OBJECT_COUNTER1 = None
                        # config.OBJECT_COUNTER = None
                        tfile = tempfile.NamedTemporaryFile()
                        tfile.write(source_video.read())
                        vid_cap = cv2.VideoCapture(
                            tfile.name)
                        st_count = st.empty()
                        st_frame = st.empty()
                        while (vid_cap.isOpened()):
                            success, image = vid_cap.read()
                            if success:
                                _display_detected_frames(conf,
                                                        model,
                                                        st_count,
                                                        st_frame,
                                                        image
                                                        )
                            else:
                                vid_cap.release()
                                break
                    except Exception as e:
                        st.error(f"Error loading video: {e}")







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
