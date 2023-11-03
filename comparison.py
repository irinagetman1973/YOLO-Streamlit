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
import yolov7.yolov7_wrapper
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
    

def detect_with_v7(uploaded_file, selected_models_v7, confidence_threshold=0.5):
    # Check if an uploaded_file is provided
    if not uploaded_file:
        return None, None

    # Load the image and convert it to a numpy array
    uploaded_img_data = uploaded_file.getvalue()
    im0 = Image.open(BytesIO(uploaded_img_data)).convert('RGB')
    im0_np = np.array(im0)

    model_results = {}

    # Infer YOLOv7 models using the YOLOv7Wrapper
    for model_name in selected_models_v7:
        yolov7_model = YOLOv7Wrapper(model_name)
        detected_image, captions = yolov7_model.detect_and_draw_boxes_from_np(im0_np, confidence_threshold=confidence_threshold)

        boxes = {'count': {}, 'details': ''}
        for caption in captions:
            parts = caption.split()
            class_name = parts[0].split('=')[1]
            box = ' '.join(parts[1:6])  # "coordinates=(x1, y1, x2, y2)"
            confidence = next(part.split('=')[1].rstrip('%') for part in parts if 'confidence' in part)

            if class_name not in boxes['count']:
                boxes['count'][class_name] = {'count': 0, 'entries': []}

            entry = {'coordinates': box, 'confidence': confidence}
            boxes['count'][class_name]['entries'].append(entry)
            boxes['count'][class_name]['count'] += 1
            boxes['details'] += f"{class_name} {box} {confidence}\n"

        model_results[model_name] = boxes

    # Reformat the current model_results into the desired structure
    results = {}
    for model_name, boxes in model_results.items():
        count_dict = {}
        detection_results = ""

        for class_name, details in boxes['count'].items():
            for entry in details['entries']:
                # detection_results += f"<b style='color: blue;'>Object type:</b> {class_name}<br>"
                # detection_results += f"<b style='color: blue;'>Coordinates:</b> {entry['coordinates']} <span style='color: blue;'>confidence</span>={entry['confidence']}%<br>---<br>"
                detection_results += f"<b style='color: blue;'>Object type:</b> {class_name}<br>"
                detection_results += (
                    f"<b style='color: blue;'>Coordinates:</b> {entry['coordinates']} "
                    f"<strong style='color: blue;'>Confidence:</strong> {entry['confidence']}%<br>---<br>"
                )
                            
            if class_name not in count_dict:
                count_dict[class_name] = {'count': details['count']}

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

    

    # Layout for displaying images in 2x2 grid
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    column_layouts = [row1_col1, row1_col2, row2_col1, row2_col2]

    fig_detected = None

    if uploaded_file:
        for idx, model_name in enumerate(selected_models):

            fig_detected = None
            # Detect based on the model
            if model_name in config.DETECTION_MODEL_DICT_V7:  # YOLOv7
                model_path = config.DETECTION_MODEL_DICT_V7[model_name]
                detected_image, results = detect_with_v7(uploaded_file, selected_models_v7, confidence_threshold=conf)
                fig_detected = create_fig(detected_image, detected=True)
                
                # Handle YOLOv7 results here:
                handle_v7_results(results, model_name)
            else:  # YOLOv8
                model_path = config.DETECTION_MODEL_DICT_V8[model_name] 
                model_instance = load_model(model_path) 
                if not model_instance:
                    st.write(f"Error: Failed to load YOLOv8 model {model_name}")
                    continue

                detected_image, results = detect_with_v8(uploaded_file, model_instance, conf=conf)
                # detected_image, boxes = infer_image(uploaded_image, model, conf)
                if not detected_image:
                    st.write(f"YOLOv8 {model_name} returned no image.")
                if not results or not results.get(model_name):
                    st.write(f"YOLOv8 {model_name} returned no results or malformed results.")
                fig_detected = create_fig(detected_image, detected=True)

                 # Handle YOLOv8 results here:
                handle_v8_results(results,model_name)
                
                
            with column_layouts[idx]:
                if fig_detected:
                    
                    st.plotly_chart(fig_detected, use_container_width=True)
                    st.write(model_name)
                    # for key, value in results['count'].items():
                    #     st.write(f"Detected {key}: {value['count']} times")
                else:
                    st.write("Error: No detections made.")

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
    if results and 'count' in results:
        for key, value in results['count'].items():
            st.write(f"Detected {key}: {value} times")
    else:
        st.write(f"YOLOv8 {model_name} returned no results or malformed results.")


#     if uploaded_file:
            
#         # Load the image and convert it to a numpy array
#         uploaded_img_data = uploaded_file.getvalue()
#         im0 = Image.open(BytesIO(uploaded_img_data)).convert('RGB')
#         im0_np = np.array(im0)
#         st.sidebar.image(im0_np, caption='Uploaded Image.', use_column_width=True)

#         loaded_models = {}
#         model_logs = {}
#         model_results = {}

#         # Infer YOLOv7 models using the YOLOv7Wrapper
#         for model_name in selected_models_v7:
#             yolov7_model = YOLOv7Wrapper(model_name)
#             detected_image, captions = yolov7_model.detect_and_draw_boxes_from_np(im0_np, confidence_threshold=conf)
            
#             boxes = {'count': {}, 'details': ''}
#             for caption in captions:
#                 parts = caption.split()
#                 class_name = parts[0].split('=')[1]
#                 box = ' '.join(parts[1:6]) # "coordinates=(x1, y1, x2, y2)"
#                 confidence = next(part.split('=')[1].rstrip('%') for part in parts if 'confidence' in part)

#                 if class_name not in boxes['count']:
#                     boxes['count'][class_name] = {'count': 0, 'entries': []}
                
#                 entry = {'coordinates': box, 'confidence': confidence}
#                 boxes['count'][class_name]['entries'].append(entry)
#                 boxes['count'][class_name]['count'] += 1
#                 boxes['details'] += f"{class_name} {box} {confidence}\n"

#             model_results[model_name] = boxes

#             st.image(detected_image)

#             # Reformat the current model_results into the desired structure
#             results = {}
#             for model_name, boxes in model_results.items():
#                 count_dict = {}
#                 detection_results = ""
                
#                 for class_name, details in boxes['count'].items():
#                     for entry in details['entries']:
#                         detection_results += f"<b style='color: blue;'>Object type:</b> {class_name}<br>"
#                         detection_results += f"<b style='color: blue;'>Coordinates:</b> {entry['coordinates']}<br>"
#                         detection_results += f"<b style='color: blue;'>Probability:</b> {entry['confidence']}<br>---<br>"

#                     if class_name not in count_dict:
#                         count_dict[class_name] = {'count': details['count']}

#                 results[model_name] = {
#                     "count": count_dict,
#                     "details": detection_results
#                 }

#             # Displaying results with Streamlit
#             st.write("### Comparison Results")
#             st.table({model: {class_id: details['count'] for class_id, details in res["count"].items()} for model, res in results.items()})

#             col_layout_detailed = st.columns(2 if len(selected_models) > 2 else len(selected_models))

#             for index, (model_name, result) in enumerate(results.items()):
#                 with col_layout_detailed[index % 2]:
#                     if result["details"].strip():
#                         scrollable_textbox = f"""
#                         <div style="
#                             font-family: 'Source Code Pro','monospace';
#                             font-size: 16px;
#                             overflow-y: scroll;
#                             border: 1px solid #000;
#                             padding: 10px;
#                             width: 500px;
#                             height: 400px;
#                         ">
#                             {result["details"]}
#                         </div>
#                         """
#                         st.markdown(f"**Detailed results for {model_name}**")
#                         st.markdown(scrollable_textbox, unsafe_allow_html=True)
#                     else:
#                         st.markdown(f"**Detailed results for {model_name}**")
#                         st.markdown(f"No objects detected by {model_name}. Please try a different image or adjust the model's confidence threshold.")






#         # Redirect stdout for model logs
#         original_stdout = sys.stdout
#         sys.stdout = new_stdout = io.StringIO()

        

#         # Loading other models
#         for model_name in selected_models_v8:
#             model_path = config.DETECTION_MODEL_DIR_V8 / model_name
#             try:
#                 loaded_models[model_name] = load_model(model_path)
#             except Exception as e:
#                 st.error(f"Unable to load model '{model_name}'. Error: {e}")
        
#         # Reset stdout to its original form
#         sys.stdout = original_stdout
#         model_logs = new_stdout.getvalue()


#         if model_logs:
#             st.text("Model logs:")
#             st.text(model_logs)

#         # Define the layout for images
#         col_layout = st.columns(2 if len(selected_models) > 2 else len(selected_models))

#         model_results = {}

#         for index, model_name in enumerate(selected_models):
#             model = loaded_models.get(model_name)
#             if not model:
#                 continue  # Model not loaded due to some error
            
#             with col_layout[index % 2]:
#                 detected_image, boxes = infer_image(im0_np, model, conf)
#                 model_results[model_name] = boxes

#                 fig_detected = create_fig(detected_image, detected=True)
#                 st.plotly_chart(fig_detected, use_container_width=True)
#                 st.write(model_name)

#         results = {}
        
#         for model_name, boxes in model_results.items():
#             count_dict = {}
#             detection_results = ""
#             for box in boxes:
#                 class_id = model.names[box.cls[0].item()]
#                 cords = box.xyxy[0].tolist()
#                 cords = [round(x) for x in cords]
#                 conf = round(box.conf[0].item(), 2)

#                 detection_results += f"<b style='color: blue;'>Object type:</b> {class_id}<br><b style='color: blue;'>Coordinates:</b> {cords}<br><b style='color: blue;'>Probability:</b> {conf}<br>---<br>"
#                 if class_id in count_dict:
#                     count_dict[class_id]['count'] += 1
#                     count_dict[class_id]['coordinates'].append(cords)
#                 else:
#                     count_dict[class_id] = {'count': 1, 'coordinates': [cords]}

#             results[model_name] = {
#                 "count": count_dict,
#                 "details": detection_results
#             }

#         st.write("### Comparison Results")
#         st.table({model: {class_id: details['count'] for class_id, details in res["count"].items()} for model, res in results.items()})


#         col_layout_detailed = st.columns(2 if len(selected_models) > 2 else len(selected_models))

#         for index, (model_name, result) in enumerate(results.items()):
#             with col_layout_detailed[index % 2]:
#                 if result["details"].strip():
#                     scrollable_textbox = f"""
#                     <div style="
#                         font-family: 'Source Code Pro','monospace';
#                         font-size: 16px;
#                         overflow-y: scroll;
#                         border: 1px solid #000;
#                         padding: 10px;
#                         width: 500px;
#                         height: 400px;
#                     ">
#                         {result["details"]}
#                     </div>
#                     """
#                     st.markdown(f"**Detailed results for {model_name}**")
#                     st.markdown(scrollable_textbox, unsafe_allow_html=True)
#                 else:
#                     st.markdown(f"**Detailed results for {model_name}**")
#                     st.markdown(f"No objects detected by {model_name}. Please try a different image or adjust the model's confidence threshold.")
        
#         st.divider()

#         if "user" in st.session_state and "uid" in st.session_state["user"]:
#             user_id = st.session_state["user"]["uid"]
#             st.markdown(
#                 """
#                 <style>
#                     .tooltip {
#                         position: relative;
#                         display: inline-block;
#                         cursor: pointer;
#                     }

#                     .tooltip .tooltiptext {
#                         visibility: hidden;
#                         width: 220px;
#                         background-color: #ffa07a;
#                         color: #fff;
#                         text-align: center;
#                         border-radius: 6px;
#                         padding: 5px;
#                         position: absolute;
#                         z-index: 1;
#                         top: -5px;
#                         left: 305%;
#                         margin-left: -110px;
#                         opacity: 0;
#                         transition: opacity 0.3s;
#                     }

#                     .tooltip:hover .tooltiptext {
#                         visibility: visible;
#                         opacity: 1;
#                     }
#                 </style>

#                 <div class="tooltip">
#                     ℹ️
#                     <span class="tooltiptext">Results will be saved to a database for future statistics</span>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )

#             if st.button('Save Results'):
#                 data_to_save = []
#                 for model, data in results.items():
                   
#                     entry = {
#                         "model": model,
#                         "inference_details": [{"class_id": key, "count": value['count'], "coordinates": value['coordinates']} for key, value in data["count"].items()],
#                         "timestamp": {".sv": "timestamp"}
#     }
#                     data_to_save.append(entry)
                
#                 try:
#                     save_to_firebase(data_to_save, user_id)
#                 except Exception as e:
#                     st.error(f"Error saving data to Firebase: {e}")
#         else:
#             st.warning("User not logged in or user ID not available.")


# def infer_image(image, model, conf):
#     # Add logic to check if it's a YOLOv7 model:
#     if isinstance(model, YOLOv7Wrapper):
#         detected_image, boxes = model.predict(image, conf=conf)  # Assuming the YOLOv7 wrapper has a predict method with similar args
#     else:
#         detected_image = model.predict(image, conf=conf)  
#         boxes = detected_image[0].boxes

#     if boxes:
#         detected_img_arr = detected_image[0].plot()[:, :, ::-1]
#         detected_image = Image.fromarray(cv2.cvtColor(detected_img_arr, cv2.COLOR_BGR2RGB))
#     else:
#         detected_image = image

#     return detected_image, boxes


