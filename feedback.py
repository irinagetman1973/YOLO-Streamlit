"""
-------------------------------------------------
   @File Name:     feedback.py
   @Author:        Irina Getman
   @Date:          15/11/2023
   @Description:   receive & store users' feedback
                   to Firebase db 
-------------------------------------------------
"""






import streamlit as st
from firebase_admin import db
import time

# Function to save feedback to Firebase
def save_feedback(feedback_text):

            # Construct the reference to the feedback node in Firebase
      ref = db.reference('feedback')

      # Generate a new child node with a unique key
      new_feedback_ref = ref.push()

      # Create the feedback data with a server timestamp
      feedback_data = {
            "text": feedback_text,
            "timestamp": {".sv": "timestamp"}
      }

      try:
            # Set the data to the new unique node
            new_feedback_ref.set(feedback_data)
            st.success("Feedback submitted successfully!")
      except Exception as e:
            st.error(f"Error saving feedback to Firebase: {e}")
# Your Streamlit UI for feedback
def feedback_ui():
      st.title("Feedback")
      
      feedback = st.text_area("Leave your feedback")



      if st.button("Submit Feedback"):
            
            try:
                  save_feedback(feedback)
            except Exception as e:
                  st.error(f"Error saving data to Firebase: {e}")

      # else:
      #       st.warning("User not logged in or user ID not available.")

      st.write("---")  

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

