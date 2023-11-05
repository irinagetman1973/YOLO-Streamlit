"""
-------------------------------------------------
   @File Name:     comparison.py
   @Author:        Irina Getman
   @Date:          20/10/2023
   @Description:   Script containing functions 
                   to compare different models' 
                   performance on the uploaded images.
-------------------------------------------------
"""


import streamlit as st
from pathlib import Path
import sys
import io
from io import BytesIO
from PIL import Image
import cv2
import config
from utils_app import load_model, create_fig
from au import db
import requests
import json
import numpy as np
from firebase_admin import db, firestore
# import yolov7.yolov7_wrapper
from yolov7.yolov7_wrapper import YOLOv7Wrapper






def save_to_firebase(data_to_save, user_id):
    
    
    # Construct the reference to the user's inferences node in Firebase
    ref = db.reference(f'/users/{user_id}/Inferences')
    # Add the timestamp to each data item
    for data in data_to_save:
        data["timestamp"] = {".sv": "timestamp"}
    
    try:
        ref.push(data_to_save)  # Using push to create a unique ID for each inference entry
        st.success("Results saved successfully!")
    except Exception as e:
        st.error(f"Failed to save results to Firebase: {e}")
    

def detect_with_v7(uploaded_file,model_name, confidence_threshold=0.5):
    # Check if an uploaded_file is provided
    if not uploaded_file:
        return None, None

    # Load the image and convert it to a numpy array
    uploaded_img_data = uploaded_file.getvalue()
    im0 = Image.open(BytesIO(uploaded_img_data)).convert('RGB')
    im0_np = np.array(im0)

    # Initialize a dictionary to store results from all models
    model_results = {}

    
    # st.write(f" model name before the wrapper : {model_name}") debugging msg
    yolov7_model = YOLOv7Wrapper(model_name)
    # Perform detection and get back the image and captions (detections)
    detected_image, captions = yolov7_model.detect_and_draw_boxes_from_np(im0_np, confidence_threshold=confidence_threshold)

    # Prepare a dictionary to store counts and entries for each class detected
    boxes = {'count': {}, 'details': ''}
    for caption in captions:
        parts = caption.split()
        class_name = parts[0].split('=')[1]
        # box = ' '.join(parts[1:6]).split('=')[1].strip('()')
        box = parts[1].split('=')[1].strip('()') + parts[2].strip(',') + ', ' + parts[3].strip(',') + ', ' + parts[4].strip(')')
        confidence = float(next(part.split('=')[1].rstrip('%') for part in parts if 'confidence' in part))
        
        # If the class name is not in the dictionary, initialize its entry
        if class_name not in boxes['count']:
            boxes['count'][class_name] = {'count': 0, 'entries': []}

        # Create an entry for this particular detection
        entry = {'coordinates': box, 'confidence': confidence}
        boxes['count'][class_name]['entries'].append(entry)
        boxes['count'][class_name]['count'] += 1

    
    model_results[model_name] = boxes

    # Now, we need to reformat the results to the desired output structure
    results = {}
    for model_name, boxes in model_results.items():
        count_dict = {}
        detection_results = ""  # This ensures we start with a clean slate for the detection results

        # Iterate over the 'count' dictionary inside the 'boxes' dictionary
        for class_name, details in boxes['count'].items():
            for entry in details['entries']:
                
                
                # Append detection info to the results string in the specified format
                detection_results += f"<b style='color: blue;'>Object type:</b> {class_name}<br>"
                detection_results += f"<b style='color: blue;'>Coordinates:</b> {entry['coordinates']}<br>"
                detection_results += f"<b style='color: blue;'>Confidence:</b> {entry['confidence']}%<br>---<br>"
                
            # Populate the count dictionary for each class detected
            if class_name not in count_dict:
                count_dict[class_name] = {'count': details['count']}

        # Assign the structured results to the results dictionary under the current model name
        results[model_name] = {
            "count": count_dict,
            "details": detection_results
        }

    
    if isinstance(detected_image, Image.Image):
        detected_pil_image = detected_image
    else:
        # This will handle the case where detected_image might be a numpy array (or any other type)
        # But if detected_image is neither a PIL Image nor a numpy array, you need to handle that scenario as well
        try:
            detected_pil_image = Image.fromarray(detected_image.astype(np.uint8))
        except Exception as e:
            print(f"Error converting detected_image to PIL image: {e}")
            detected_pil_image = None  # Or you can provide a default image here

    # st.write(model_results)
    return detected_pil_image, results


    
def detect_with_v8(uploaded_file, model, conf=0.5):
    """
    Execute inference for uploaded image with YOLOv8 model.

    Parameters:
    - uploaded_file: The uploaded image file.
    - model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    - conf: Confidence threshold for YOLOv8 model.

    Returns:
    - detected_image: Processed image with detections (PIL Image).
    - results: Dictionary containing detection results.
    """

    if uploaded_file is None:
        return None, None

    try:
        uploaded_image = Image.open(uploaded_file)
    except Exception as e:
        print(f"Error opening uploaded image: {e}")
        return None, None

    detected_image_result = model.predict(uploaded_image, conf=conf)
    boxes = detected_image_result[0].boxes

    # Get the plotted image with detections from the Results object
    try:
        detected_img_arr = detected_image_result[0].plot()[:, :, ::-1]  # Assuming this returns a numpy array
        # Convert the numpy array to a PIL Image object
        detected_image = Image.fromarray(cv2.cvtColor(detected_img_arr, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Error processing detected image: {e}")
        detected_image = None

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

    results = {
        "count": count_dict,
        "details": detection_results
    }
      # Display the results for debugging.

    return detected_image, results
    
def run_detection(detection_model_name, file, confidence):        
            # Detect based on the model
            
            if detection_model_name in config.DETECTION_MODEL_DICT_V7:
                
                    with st.spinner(f'Detecting objects with {detection_model_name}...'):  # YOLOv7
                        model_path = config.DETECTION_MODEL_DICT_V7[detection_model_name]
                        
                        detected_image, results = detect_with_v7(file, detection_model_name, confidence_threshold=confidence)
                        
            else:  # YOLOv8
                model_path = config.DETECTION_MODEL_DICT_V8[detection_model_name] 
                model_instance = load_model(model_path) 
                if not model_instance:
                    st.write(f"Error: Failed to load YOLOv8 model {detection_model_name}")
                detected_image, results = detect_with_v8(file, detection_model_name, conf=confidence)
            
           
                    
            return  detected_image, results

def compare_models_function():
    st.markdown("### Compare different models' performance")
    st.divider()

    
    st.sidebar.divider()
    # New UI for YOLOv7 models
    st.sidebar.markdown("### YOLOv7 Models")
    selected_models_v7 = st.sidebar.multiselect(
        "Select YOLOv7 models for comparison:",
        config.DETECTION_MODEL_LIST_V7, 
        key="models_comparison_selectbox_v7"
    )

    # UI for other models (V8)
    st.sidebar.markdown("### YOLOv8 Models")
    selected_models_v8 = st.sidebar.multiselect(
        "Select other models for comparison:",
        config.DETECTION_MODEL_LIST_V8,
        key="models_comparison_selectbox_v8"
    )
    
    selected_models = selected_models_v7 + selected_models_v8

    conf = float(st.sidebar.slider(
        "Select Model Confidence", 30, 100, 50)) / 100

    if len(selected_models) > 4:
        st.warning("You can only select up to 4 models.")
        return

    uploaded_file = st.sidebar.file_uploader("Choose an image for comparison...", type=["jpg", "png"])
    if uploaded_file is not None and uploaded_file.type.startswith('image/'):
        st.sidebar.divider()
        st.sidebar.image(uploaded_file)

    
            
    
    # Layout for displaying images in 2x2 grid
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    column_layouts = [row1_col1, row1_col2, row2_col1, row2_col2]

    if uploaded_file:
        all_results = {}

        for idx, selected_model_name in enumerate(selected_models):
            if idx > 3:
                break 
            with column_layouts[idx]:
                st.markdown(f"#### Model: {selected_model_name}")
                # Ensure each button has a unique key
                if st.button(f"Run {selected_model_name}", key=f"run_detection_{selected_model_name}"):
                    # st.write(f"Attempting to detect with model: {selected_model_name}")
                    detected_image, results = run_detection(selected_model_name, uploaded_file, conf)  
                    # Store the results
                    all_results[selected_model_name] = (detected_image, results)

        # After all models have been run, handle the displaying of results
        for selected_model_name, (detected_image, results) in all_results.items():
            idx = selected_models.index(selected_model_name)  # Get the index for the layout
            with column_layouts[idx]:
                if detected_image:
                    # Display the detection image
                    st.image(detected_image, use_column_width=True)
                    # Display results
                    # Call the appropriate results handling function based on the model version
                    if selected_model_name in config.DETECTION_MODEL_LIST_V7:
                        handle_v7_results(results, selected_model_name)
                    else:
                        handle_v8_results(results, selected_model_name)
                else:
                    st.error(f"Detection failed or no objects were detected for {selected_model_name}.")



          
 
                

def handle_v7_results(results, model_name):
    # Displaying results with Streamlit
    st.write("### Comparison Results")
    st.table({model: {class_id: details['count'] for class_id, details in res["count"].items()} for model, res in results.items()})

    col_layout_detailed = st.columns(2)

    for index, (model_name, result) in enumerate(results.items()):
        with col_layout_detailed[index % 2]:
            if result["details"].strip():
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
                    {result["details"]}
                </div>
                """
                st.markdown(f"**Detailed results for {model_name}**")
                st.markdown(scrollable_textbox, unsafe_allow_html=True)
            else:
                st.markdown(f"**Detailed results for {model_name}**")
                st.markdown(f"No objects detected by {model_name}. Please try a different image or adjust the model's confidence threshold.")

def handle_v8_results(results, model_name):
    
    # st.write(f"Type of res: {type(results)}")  # Add this for debugging
    if not isinstance(results, dict):
        raise ValueError("res should be a dictionary.")
    # Ensure that the 'count' key exists in each model's results
  
    count = results.get("count")
    details = results.get("details", "").strip()
    
    if count:
        st.write("### Detection Count")
        # Display the counts in some manner, perhaps in a table or list
        for class_id, count in results['count'].items():
            st.write(f"{class_id}: {count}")

    # Detailed results display
    col_layout_detailed = st.columns(2)

    for index, (model_name, result) in enumerate(results['details'].items()):
        with col_layout_detailed[index % 2]:
            
            if details:
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
                    {details}
                </div>
                """
                st.markdown(f"**Detailed results for {model_name}**")
                st.markdown(scrollable_textbox, unsafe_allow_html=True)
            else:
                st.markdown(f"**Detailed results for {model_name}**")
                st.markdown(f"No objects detected by {model_name}. Please try a different image or adjust the model's confidence threshold.")





