import sys
sys.path.append('./yolov7')

import streamlit as st
from yolov7.yolov7_wrapper import YOLOv7Wrapper  
from PIL import Image
import numpy as np
from io import BytesIO
import time
import psutil

import psutil
import time
from PIL import Image

import io
import numpy as np
import streamlit as st







def main():
    st.title("YOLOv7 Detection Test")

    # 1. Model Selection:
    model_names = list(YOLOv7Wrapper.MODELS.keys())
    selected_model = st.selectbox('Choose a YOLOv7 Model:', model_names, key="model_selectbox_v7")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    confidence_threshold = st.slider('Confidence Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    if uploaded_file is not None:
        # Load the image and convert it to a numpy array
        uploaded_img_data = uploaded_file.getvalue()
        im0 = Image.open(BytesIO(uploaded_img_data)).convert('RGB')
        im0 = np.array(im0)
        st.image(im0, caption='Uploaded Image.', use_column_width=True)

        # Redirect stdout
        original_stdout = sys.stdout
        sys.stdout = new_stdout = io.StringIO()

        # Initialize the wrapper
        wrapper = YOLOv7Wrapper(model_name=selected_model)

        # Reset stdout to its original form
        sys.stdout = original_stdout

        # Get the output and print it to Streamlit
        output = new_stdout.getvalue()
        if output:
            st.text("Model logs:")
            st.text(output)

        # Record the initial CPU percentage
        initial_cpu = psutil.cpu_percent(interval=None)

        # --- Start time --------
        start_time = time.time()

        # Detect and draw the boxes
        detections_img, captions = wrapper.detect_and_draw_boxes_from_np(im0, confidence_threshold)

        # ------- END TIME -------
        end_time = time.time()
        inference_time = end_time - start_time

        # Calculate the CPU usage during inference
        final_cpu = psutil.cpu_percent(interval=None)
        cpu_usage_during_inference = final_cpu - initial_cpu

        st.markdown(f"Inference time: {inference_time:.4f} seconds")
        st.markdown(f"CPU usage during inference: {cpu_usage_during_inference:.2f}%")

        # Display the detections' captions if available
        if len(captions) > 0:
            st.write("\n".join(captions))
            num_boxes = len(captions)
            st.write(f"Number of detections: {num_boxes}")
        else:
            st.write("No detections found!")

        # Display the results
        st.image(detections_img, caption='Processed Image with Detections', use_column_width=True)












        
       

if __name__ == "__main__":
    main()
