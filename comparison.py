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
# import plotly.graph_objects as go
from PIL import Image
import cv2
import config
from utils import load_model, create_fig
from au import db
import requests
import json
from firebase_admin import db, firestore


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
    

    


def compare_models_function():
    st.markdown("### Compare different models' performance")
    st.divider()

    selected_models = st.sidebar.multiselect(
        "Select up to 4 models for comparison:",
        config.DETECTION_MODEL_LIST,
        key="models_comparison_selectbox"
    )

    conf = float(st.sidebar.slider(
        "Select Model Confidence", 30, 100, 50)) / 100

    if len(selected_models) > 4:
        st.warning("You can only select up to 4 models.")
        return

    uploaded_file = st.sidebar.file_uploader("Choose an image for comparison...", type=["jpg", "png"])

    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)

        # Load models first to avoid multiple loads
        loaded_models = {}
        for model_name in selected_models:
            model_path = Path(config.DETECTION_MODEL_DIR, str(model_name))
            try:
                loaded_models[model_name] = load_model(model_path)
            except Exception as e:
                st.error(f"Unable to load model '{model_name}'. Error: {e}")

        # Define the layout for images
        col_layout = st.columns(2 if len(selected_models) > 2 else len(selected_models))

        model_results = {}

        for index, model_name in enumerate(selected_models):
            model = loaded_models.get(model_name)
            if not model:
                continue  # Model not loaded due to some error
            
            with col_layout[index % 2]:
                detected_image, boxes = infer_image(uploaded_image, model, conf)
                model_results[model_name] = boxes

                fig_detected = create_fig(detected_image, detected=True)
                st.plotly_chart(fig_detected, use_container_width=True)
                st.write(model_name)

        results = {}
        
        for model_name, boxes in model_results.items():
            count_dict = {}
            detection_results = ""
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

            results[model_name] = {
                "count": count_dict,
                "details": detection_results
            }

        st.write("### Comparison Results")
        st.table({model: res["count"] for model, res in results.items()})

        col_layout_detailed = st.columns(2 if len(selected_models) > 2 else len(selected_models))

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
                    st.markdown(f"### Detailed results for {model_name}")
                    st.markdown(scrollable_textbox, unsafe_allow_html=True)
                else:
                    st.markdown(f"### Detailed results for {model_name}")
                    st.markdown(f"No objects detected by {model_name}. Please try a different image or adjust the model's confidence threshold.")
        
        st.divider()

        if "user" in st.session_state and "uid" in st.session_state["user"]:
            user_id = st.session_state["user"]["uid"]
            st.markdown(
                """
                <style>
                    .tooltip {
                        position: relative;
                        display: inline-block;
                        cursor: pointer;
                    }

                    .tooltip .tooltiptext {
                        visibility: hidden;
                        width: 220px;
                        background-color: #ffa07a;
                        color: #fff;
                        text-align: center;
                        border-radius: 6px;
                        padding: 5px;
                        position: absolute;
                        z-index: 1;
                        top: -5px;
                        left: 305%;
                        margin-left: -110px;
                        opacity: 0;
                        transition: opacity 0.3s;
                    }

                    .tooltip:hover .tooltiptext {
                        visibility: visible;
                        opacity: 1;
                    }
                </style>

                <div class="tooltip">
                    ℹ️
                    <span class="tooltiptext">Results will be saved to a database for future statistics</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button('Save Results'):
                data_to_save = []
                for model, data in results.items():
                    entry = {
                        "model": model,
                        "inference_details": [{"class_id": key, "count": value} for key, value in data["count"].items()],
                        "timestamp": {".sv": "timestamp"}
                    }
                    
                    data_to_save.append(entry)
                
                try:
                    save_to_firebase(data_to_save, user_id)
                except Exception as e:
                    st.error(f"Error saving data to Firebase: {e}")
        else:
            st.warning("User not logged in or user ID not available.")


def infer_image(image, model, conf):
    detected_image = model.predict(image, conf=conf)  
    boxes = detected_image[0].boxes

    if boxes:
        detected_img_arr = detected_image[0].plot()[:, :, ::-1]
        detected_image = Image.fromarray(cv2.cvtColor(detected_img_arr, cv2.COLOR_BGR2RGB))
    else:
        detected_image = image

    return detected_image, boxes

