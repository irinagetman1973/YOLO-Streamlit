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
from firebase_admin import db, storage


def save_to_firebase(data_to_save, user_id):
    
    
    # Construct the reference to the user's inferences node in Firebase
    ref = db.reference(f'/users/{user_id}/Inferences')

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

    if len(selected_models) > 4:
        st.warning("You can only select up to 4 models.")
        return

    uploaded_file = st.sidebar.file_uploader("Choose an image for comparison...", type=["jpg", "png"])

    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)

        # Determine the layout based on the number of selected models
        if len(selected_models) in [1, 2]:
            col_layout = st.columns(len(selected_models))
        elif len(selected_models) in [3, 4]:
            col_layout = st.columns(2)  # 2 columns for 3 or 4 models

        # Dictionary to store results from each model
        model_results = {}

        for index, model_name in enumerate(selected_models):
            model_path = Path(config.DETECTION_MODEL_DIR, str(model_name))
            try:
                model = load_model(model_path)
            except Exception as e:
                st.error(f"Unable to load model '{model_name}'. Error: {e}")
                continue

            # Manage the row placement when there are 3 or 4 models
            if len(selected_models) == 3 and index == 2:
                st.write("")  # This creates a new row
            elif len(selected_models) == 4 and index == 2:
                st.write("")  # This creates a new row

            with col_layout[index % 2]:  
                detected_image, boxes = infer_image(uploaded_image, model)
                model_results[model_name] = boxes

                fig_detected = create_fig(detected_image, detected=True)
                st.plotly_chart(fig_detected, use_container_width=True)
                st.write(model_name)

        # Now, create a table to compare results
        table_data = {}
        for model_name, boxes in model_results.items():
            count_dict = {}
            for box in boxes:
                class_id = model.names[box.cls[0].item()]
                if class_id in count_dict:
                    count_dict[class_id] += 1
                else:
                    count_dict[class_id] = 1
            table_data[model_name] = count_dict

        # Create and display the table
        st.write("### Comparison Results")
        st.table(table_data)

        if "user" in st.session_state and "uid" in st.session_state["user"]:
            user_id = st.session_state["user"]["uid"]
            if st.button('Save Results'):
                data_to_save = [{'model': model, 'class_id': class_id, 'count': count} 
                        for model, class_dict in table_data.items() 
                        for class_id, count in class_dict.items()]
                save_to_firebase(data_to_save, user_id)  # Pass the user_id to the function
        else:
            st.warning("User not logged in or user ID not available.")




def infer_image(image, model):
    detected_image = model.predict(image, conf=0.5)  # You can adjust this confidence
    boxes = detected_image[0].boxes

    if boxes:
        detected_img_arr = detected_image[0].plot()[:, :, ::-1]
        detected_image = Image.fromarray(cv2.cvtColor(detected_img_arr, cv2.COLOR_BGR2RGB))
    else:
        detected_image = image

    return detected_image, boxes

