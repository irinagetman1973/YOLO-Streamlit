import streamlit as st
from au import authenticate
from app import  display_main_page

st.set_page_config(
    page_title="Vehicle Tracking with YOLOv8",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def display_sidebar():
    if st.sidebar.button("Login/Sign Up", key="login_signup_button"):
        st.session_state.page = 'authentication'
        st.experimental_rerun()

        

def main():
      if 'page' not in st.session_state:
        st.session_state.page = 'main'  # Set default page if not set

      display_sidebar()  # Display the sidebar
      display_main_page()

      if st.session_state.page == 'authentication':
        authenticate()  # This calls the authenticate function from au.py
      # elif st.session_state.page == 'main':
        

if __name__ == "__main__":
    main()
