import sys
sys.path.append('./yolov7')

import streamlit as st
from yolov7.yolov7_wrapper import YOLOv7Wrapper  
from PIL import Image
import numpy as np
from io import BytesIO
import tempfile
import cv2



def main():
    st.title("YOLOv7 Detection Test")

    # 1. Model Selection:
    model_names = list(YOLOv7Wrapper.MODELS.keys())
    selected_model = st.selectbox('Choose a YOLOv7 Model:', model_names)

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    confidence_threshold = st.slider('Confidence Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    if uploaded_file is not None:
        # Load the image and convert it to a numpy array
        uploaded_img_data = uploaded_file.getvalue()
        im0 = Image.open(BytesIO(uploaded_img_data)).convert('RGB')
        im0 = np.array(im0)
        st.image(im0, caption='Uploaded Image.', use_column_width=True)

        # Initialize the wrapper
        wrapper = YOLOv7Wrapper(model_name=selected_model)

        # Detect and draw the boxes
        detections_img, captions = wrapper.detect_and_draw_boxes_from_np(im0, confidence_threshold)

        # Display the detections' captions if available
        if captions:
            st.write("\n".join(captions))

        # Display the results
        st.image(detections_img, caption='Processed Image with Detections', use_column_width=True)



# def main():
#     st.title("YOLOv7 Detection Test")

#     uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
#     if uploaded_file is not None:
#         col1, col2 = st.columns(2)

#         with col1:
#             st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
#             st.markdown("**Original Image**")

#         # Initialize the wrapper
#         wrapper = YOLOv7Wrapper(model_name="yolov7-e6.pt")

#         if st.button("Detect Objects"):
#             with st.spinner("Running detection..."):
#                 image = Image.open(uploaded_file)
#                 image_np = np.array(image)
#                 detections_img, captions = wrapper.detect_and_draw_boxes_from_np(image_np)

#             with col2:
#                 if detections_img is not None:
#                     st.image(detections_img, caption='Processed Image with Detections', use_column_width=True)
#                     st.markdown("**Detected Image**")

#                     detection_results = ""
#                     count_dict = {}

#                     for caption in captions:
#                         object_type, coordinates, score = process_caption(caption)  # Assuming you have a method to process captions
#                         detection_results += f"<b style='color: blue;'>Object type:</b> {object_type}<br><b style='color: blue;'>Coordinates:</b> {coordinates}<br><b style='color: blue;'>Probability:</b> {score}<br>---<br>"
#                         if object_type in count_dict:
#                             count_dict[object_type] += 1
#                         else:
#                             count_dict[object_type] = 1

#                     for object_type, count in count_dict.items():
#                         detection_results += f"<b style='color: blue;'>Count of {object_type}:</b> {count}<br>"

#                     scrollable_textbox = f"""
#                         <div style="
#                             font-family: 'Source Code Pro','monospace';
#                             font-size: 16px;
#                             overflow-y: scroll;
#                             border: 1px solid #000;
#                             padding: 10px;
#                             width: 500px;
#                             height: 400px;
#                         ">
#                             {detection_results}
#                         </div>
#                     """
#                     st.markdown("""### Results:""")
#                     st.markdown(scrollable_textbox, unsafe_allow_html=True)

#                 else:
#                     st.markdown("""### No objects detected""")
#                     st.markdown("""
#                         The model did not detect any objects in the uploaded image.
#                         Please try with a different image or adjust the model's 
#                         confidence threshold in the sidebar and try again.
#                     """)

# def process_caption(caption):
#     # Assuming captions are in format "object_type (coordinates): score"
#     # Example: "car (23,45,67,89): 0.85"
#     object_type, rest = caption.split(" ", 1)
#     coordinates, score = rest.split(":")
#     score = float(score.strip())
#     coordinates = coordinates.strip()[1:-1]  # Removing parentheses
#     return object_type, coordinates, score






        
       

if __name__ == "__main__":
    main()
