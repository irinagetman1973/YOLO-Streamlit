import streamlit as st
from au import authenticate
from app import  display_main_page
from avatar_manager import  get_avatar_url, store_avatar
import time
from PIL import Image, ImageDraw, ImageOps
import requests
from io import BytesIO

#-------------Page Configuration-------------------
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
        # st.sidebar.write(f'Welcome, {st.session_state.user["username"]}')
        st.sidebar.markdown(f'### Welcome, **{st.session_state.user["username"]}**!')

        # Check if the user has an avatar
        if avatar_url:
            # Avatar exists, display it
            response = requests.get(avatar_url)
            
            avatar = Image.open(BytesIO(response.content))
        else:
            # Avatar does not exist, offer file uploader
            uploaded_file = st.file_uploader("Choose an avatar:", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                avatar = Image.open(uploaded_file)
                store_avatar(user_id, uploaded_file)  # Assume store_avatar() saves the avatar and updates the URL in the database
                st.session_state['avatar_url'] = get_avatar_url(user_id)  # Update the session_state with the new avatar URL
            else:
                return  # Return early if no file is uploaded, to avoid running the code below with a None avatar

        #-------avatar displaying-------------------
        # avatar = Image.open("avatars/fashion-little-boy.jpg")

        # Create a mask of a filled circle
        mask = Image.new("L", avatar.size)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0) + avatar.size, fill=255)

        # Apply the mask to the avatar
        rounded_avatar = Image.new("RGBA", avatar.size)
        rounded_avatar.paste(avatar, mask=mask)
        avatar_width = 64  
        # Display the rounded avatar in Streamlit
        st.sidebar.image(rounded_avatar,  width=avatar_width)


        # if avatar_url:
        #     st.sidebar.image(avatar_url, width=150)
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
