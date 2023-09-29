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
# import imageio
import tempfile
import config
import numpy as np
from results import Results
import os
import datetime

# import supervision as sv
import os
os.environ['TMPDIR'] = 'C:\\Users\\irina\\tmp'



def _display_detected_frames(conf, model, st_count, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    #image = cv2.resize(image, (720, int(720 * (9 / 16))))

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

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )
        if source_img:
            if st.button("Execution"):
                with st.spinner("Running..."):
                    res = model.predict(uploaded_image, conf=conf)
                    boxes = res[0].boxes

                with col2:
                    st.image(res[0].plot()[:, :, ::-1],
                             caption="Detected Image",
                             use_column_width=True)

                    try:
                        with st.expander("Detection Results"):
                            if boxes is not None:
                                count_dict = {}
                                for box in boxes:
                                    class_id = model.names[box.cls[0].item()]
                                    cords = box.xyxy[0].tolist()
                                    cords = [round(x) for x in cords]
                                    conf = round(box.conf[0].item(), 2)
                                    st.write("Object type:", class_id)
                                    st.write("Coordinates:", cords)
                                    st.write("Probability:", conf)
                                    st.write("---")
                                    
                                    # Add to count dictionary
                                    if class_id in count_dict:
                                        count_dict[class_id] += 1
                                    else:
                                        count_dict[class_id] = 1
                                
                                # Print out counts of each object type
                                for object_type, count in count_dict.items():
                                    st.write(f"Count of {object_type}: {count}")
                            else:
                                st.write("No results found.")
                    except Exception as ex:
                        st.write("Error occurred during inference.")
                        st.write(ex)




# def infer_uploaded_video(conf, model):
#     """
#     Execute inference for uploaded video
#     :param conf: Confidence of YOLOv8 model
#     :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
#     :return: None
#     """
#     source_video = st.sidebar.file_uploader(label="Choose a video...")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.video(source_video)

#     with col2:
#         if source_video:
#             if st.button("Execution"):
#                 with st.spinner("Running..."):
#                     try:
#                         tfile = tempfile.NamedTemporaryFile()
#                         tfile.write(source_video.read())
#                         vid_cap = cv2.VideoCapture(tfile.name)
#                         st_count = st.empty()
#                         st_frame = st.empty()
#                         if not vid_cap.isOpened():
#                             raise ValueError("Failed to open video capture.")


#                         execution_count = 0

#                         output_dir = "results"
#                         os.makedirs(output_dir, exist_ok=True)
#                         now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#                         output_file = os.path.join(output_dir, f"output_{now}.mp4")

#                         # Define the resolution
#                         width = 1280
#                         height = 720

#                         # Create an instance of cv2.VideoCapture
#                         cap = cv2.VideoCapture(0)
#                         cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#                         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#                         # Define the video codec and create the VideoWriter object
#                         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                         frame_rate = 7.0
#                         writer = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height), isColor=True)

#                         # Customize the bounding box
#                         box_annotator = sv.BoxAnnotator(
#                             thickness=2,
#                             text_thickness=2,
#                             text_scale=1
#                         )

#                         while True:
#                             ret, frame = vid_cap.read()
                            
#                             st.write(ret, frame)
#                             st.write(frame.shape)
#                             st.write(frame)

#                             if not ret:
#                                 break
#                             # Perform detection on the frame
#                             result = model(frame, agnostic_nms=True)[0]
#                             detections = sv.Detections.from_yolov8(result)
#                             labels = [
#                                 f"{model.model.names[class_id]} {confidence:0.2f}"
#                                 for _, confidence, class_id, _
#                                 in detections
#                             ]
#                             frame = box_annotator.annotate(
#                                 scene=frame,
#                                 detections=detections,
#                                 labels=labels
#                             )

#                             # Write the frame to the output video file
#                             writer.write(frame)

#                             # Display the frame in Streamlit
#                             st.image(frame, channels="BGR")

#                             execution_count += 1

#                         cap.release()  # Release the video capture
#                         writer.release()  # Release the VideoWriter
#                         cv2.destroyAllWindows()

#                         # Display the total number of executions
#                         st.write(f"The code was executed {execution_count} times.")

#                         # Display the result video
#                         st.video(output_file)

#                     except Exception as e:
#                         st.error(f"Error loading video: {e}")

# def infer_uploaded_video(conf, model):
#     """
#     Execute inference for uploaded video
#     :param conf: Confidence of YOLOv8 model
#     :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
#     :return: None
#     """
#     source_video = st.sidebar.file_uploader(label="Choose a video...")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.video(source_video)

#     with col2:
#         if source_video:
#             if st.button("Execution"):
#                 with st.spinner("Running..."):
#                     try:
#                         tfile = tempfile.NamedTemporaryFile()
#                         tfile.write(source_video.read())
                        
#                         vstream = imageio.get_reader(tfile.name)
                        
#                         # Get the total number of frames
#                         num_frames = vstream.get_length()
                        
#                         # Create a progress bar
#                         progress_bar = st.progress(0)
                        
#                         frames = []
#                         for i, frame in enumerate(vstream):
#                             # Update the progress bar
#                             progress_bar.progress(i / num_frames)
                            
#                             # Convert to PIL image
#                             pil_image = Image.fromarray(frame)
                            
#                             # Perform inference
#                             res = model.predict(pil_image, conf=conf)
#                             boxes = res[0].boxes
                            
#                             # Annotate and append to frames
#                             frame = res[0].plot()[:, :, ::-1]
#                             frames.append(frame)
                            
#                             # Print results
#                             if boxes is not None:
#                                 count_dict = {}
#                                 for box in boxes:
#                                     class_id = model.names[box.cls[0].item()]
#                                     cords = box.xyxy[0].tolist()
#                                     cords = [round(x) for x in cords]
#                                     conf = round(box.conf[0].item(), 2)
#                                     st.write("Object type:", class_id)
#                                     st.write("Coordinates:", cords)
#                                     st.write("Probability:", conf)
#                                     st.write("---")
                                    
#                                     # Add to count dictionary
#                                     if class_id in count_dict:
#                                         count_dict[class_id] += 1
#                                     else:
#                                         count_dict[class_id] = 1
                                
#                                 # Print out counts of each object type
#                                 for object_type, count in count_dict.items():
#                                     st.write(f"Count of {object_type}: {count}")
                            
#                         # Complete the progress bar
#                         progress_bar.progress(1.0)
                        
#                         # Write frames to output video
#                         output_dir = "results"
#                         os.makedirs(output_dir, exist_ok=True)
#                         now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#                         output_file = os.path.join(output_dir, f"output_{now}.mp4")
#                         imageio.mimwrite(output_file, frames, fps=24)

#                         st.video(output_file)

#                     except Exception as e:
#                         st.error(f"Error processing video: {e}")


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
