import streamlit as st
import time
from comparison import compare_models_function
import config
from vizualization import visualize_inferences


# Placeholder for stored results. 
stored_results = [
    {"image_name": "image1.jpg", "model": "YOLOv7", "objects_detected": 5, "image_link": "link_to_image1"},
    {"image_name": "image2.jpg", "model": "YOLOv7", "objects_detected": 3, "image_link": "link_to_image2"},
]

def display_dashboard():   
  
    

    dashboard_sections = ["Compare models", "Statistics", "Query Tool", "Entry Management", "Feedback Section"]

        # By default, the section is set to None to show the instruction page.
    section = st.session_state.get('dashboard_section', "")

        # Render the selectbox and store the choice in 'section'
    section = st.sidebar.selectbox("Choose a section to continue:", [""] + dashboard_sections, key="dashboard_section_selectbox", format_func=lambda x: "Select a section..." if x == "" else x)

    st.session_state.dashboard_section = section

    if not section:
        lottie = """
            <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
            <lottie-player src="https://raw.githubusercontent.com/irinagetman1973/YOLO-Streamlit/main/animation_sphere.json" background="transparent" speed="1" style="width: 400px; height: 400px;" loop autoplay></lottie-player>
            """
        st.markdown("""
            <style>
                iframe {
                    position: fixed;
                    top: 16rem;
                    bottom: 0;
                    left: 105;
                    right: 0;
                    margin: auto;
                    z-index=-1;
                }
            </style>
            """, unsafe_allow_html=True
        )


        st.components.v1.html(lottie, width=410, height=410) # When the selection is empty
        col1, col2 = st.columns([0.6, 0.3])  # Create a 2-column layout with equal width for each column

        with col1:
            st.subheader("🎓 Welcome to the Dashboard!")
            st.divider()
            st.write("""
            Welcome! Please select an option from the sidebar to begin. Each section offers unique functionalities:
            """)

            # Models Performance Comparison
            st.markdown("""
            **:one: **Models Performance Comparison**:**
            Compare the object detection capabilities of different models. 
            View detected images side-by-side for visual comparison, inference times
            and explore detection results including the count of each detected class 
            in a detailed table format.
            You can also save detection results to the database for future analysis.
            """)

            st.divider()

            # Results Access
            st.markdown("""
            **:two: **Results Access**:**
            Access and review past detection results. 
            Visualize them through charts and graphs, 
            and download the results as an Excel worksheet.
            """)

            st.divider()

            # Query Tool
            st.markdown("""
            **:three: **Query Tool**:**
            Use our search tool to find specific entries quickly and efficiently.
            """)

            st.divider()

            # Feedback
            st.markdown("""
            **:four: **Feedback**:**
            We value your input! Share your feedback  to help us improve.
            """)

            st.divider()

            st.write("**Enjoy your experience and happy detecting!** :green_heart:")

            
            

    

    elif section == "Compare models":
      
        compare_models_function()  # Let's define this function next
    
    elif section == "Statistics":
      
      visualize_inferences()

     

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

  
