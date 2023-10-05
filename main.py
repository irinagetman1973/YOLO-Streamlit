import streamlit as st
from au import authenticate
from app import  display_main_page
from avatar_manager import upload_and_store_avatar, get_avatar_url
import time
st.set_page_config(
    page_title="Vehicle Tracking with YOLOv8",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def display_sidebar():
    if not st.session_state.get('authenticated', False):
        if st.sidebar.button("Login/Sign Up", key="login_signup_button"):
            st.session_state.page = 'authentication'
            st.rerun()
    else:
        
        user_id = st.session_state.user["uid"]
        avatar_url = get_avatar_url(user_id)
        st.sidebar.write(f'Welcome, {st.session_state.user["username"]}')
        if avatar_url:
            st.sidebar.image(avatar_url, caption=f"Hello, {st.session_state.username}!", width=150)
            #--------Logout button-------------------
        if st.sidebar.button("Logout", key="logout_button"):
            st.session_state.pop('user', None)  # Remove user info from session_state
            st.session_state.authenticated = False  # Set authenticated state to False
            st.sidebar.success('Logged out successfully.')
            time.sleep(2)
            st.session_state.page = 'main'  # Redirect to main page
            st.rerun()

        # if st.sidebar.button("Back to Main Page"):
        #     st.session_state.page = 'main'
        #     st.rerun()

        # if st.sidebar.button("Dashboard"):
        #     st.session_state.page = 'dashboard'
        #     st.rerun()

        

def main():
      if 'page' not in st.session_state:
        st.session_state.page = 'main'  # Set default page if not set

      display_sidebar()  # Display the sidebar
      # display_main_page()

      if st.session_state.page == 'authentication':
        authenticate()  # This calls the authenticate function from au.py
      elif st.session_state.page == 'main':
        display_main_page()
      # elif st.session_state.page == 'dashboard':
      #   display_dashboard()  
        

if __name__ == "__main__":
    main()
