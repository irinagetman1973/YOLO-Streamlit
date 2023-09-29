import streamlit as st
import time

# Placeholder for stored results. 
stored_results = [
    {"image_name": "image1.jpg", "model": "YOLOv7", "objects_detected": 5, "image_link": "link_to_image1"},
    {"image_name": "image2.jpg", "model": "YOLOv7", "objects_detected": 3, "image_link": "link_to_image2"},
]

def display_dashboard():
    st.subheader("Dashboard")

    # Dashboard sections
    dashboard_sections = ["Model Fine-Tuning", "Results Access", "Query Tool", "Entry Management", "Feedback Section"]
    section = st.sidebar.selectbox("Choose a section", dashboard_sections, key="dashboard_section_selectbox")

    if section == "Model Fine-Tuning":
      st.write("Fine-Tune Your YOLO Model")

        # Sliders for model parameters
      learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001)
      batch_size = st.slider("Batch Size", min_value=1, max_value=128, value=32, step=1)
      epochs = st.slider("Epochs", min_value=1, max_value=100, value=10, step=1)

        # Fine-tune button
      if st.button("Fine-Tune Model"):
            # Placeholder logic for fine-tuning. In a real-world scenario, you'd use these parameters to adjust/train the model.
            st.write(f"Fine-tuning model with Learning Rate: {learning_rate}, Batch Size: {batch_size}, and Epochs: {epochs}")
            st.success("Model fine-tuned successfully!")


    elif section == "Results Access":
      st.write("Access Past Detection Results")

        # Displaying the results in a table format
      for result in stored_results:
            st.write(f"Image Name: {result['image_name']}")
            st.write(f"Model Used: {result['model']}")
            st.write(f"Objects Detected: {result['objects_detected']}")
            st.write(f"[View Detected Image]({result['image_link']})")  # Placeholder link
            st.write("---")  # Line separator for better clarity

        # Download option (this is a placeholder, replace with actual logic to download images)
      if st.button("Download All Results"):
            st.write("Downloading all detected images...") 

    elif section == "Query Tool":
        query = st.text_input("Search for an entry")
        if query:
            st.write(f"Results for '{query}' will be displayed here.")

    elif section == "Entry Management":
        st.write("Options to edit or delete specific entries will be here.")

    elif section == "Feedback Section":
       
        feedback = st.text_area("Leave your feedback")
        if st.button("Submit Feedback"):
            st.write("Thank you for your feedback!")  # Placeholder. You'd typically save this feedback in a database.

        st.write("---")  # Line separator for clarity

        # GitHub link with emojis and animation
        st.write("If you liked my app, please, give me a star! :point_down:")
        github_link = "[![Star on GitHub](https://img.shields.io/github/stars/irinagetman1973/Urban-Vehicle-Detection-via-DL?style=social)](https://github.com/irinagetman1973/Urban-Vehicle-Detection-via-DL)"
        st.markdown(github_link, unsafe_allow_html=True)

        # Simple blinking animation for the star emoji
        placeholder = st.empty()
        for _ in range(5):  # Blink for 5 times
            placeholder.markdown(":star:")
            time.sleep(0.5)
            placeholder.markdown(" ")
            time.sleep(0.5)

    # Logout button in the sidebar
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.page = 'main'
        st.write("Logged out successfully!")
