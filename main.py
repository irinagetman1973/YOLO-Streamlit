"""
-------------------------------------------------
   @File Name:     main.py
   @Author:        Irina Getman
   @Date:          01/10/2023
   @Description:   main script to start app
                   <streamlit run main.py>
-------------------------------------------------
"""

import streamlit as st
from au import authenticate
from app import  display_main_page
from avatar_manager import  get_avatar_url, store_avatar
import time
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from dashboard import display_dashboard


#-------------Page Configuration-------------------
st.set_page_config(
    page_title="YOLO app",
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
        
        st.sidebar.markdown(f'### Welcome, **{st.session_state.user["username"]}**!')

        
         #----Check if the user has an avatar or if an avatar has been chosen during this session-------------------------------
        if avatar_url or st.session_state.get('avatar_chosen', False):
            # Avatar exists, display it
            if not avatar_url:  # If avatar_url is None, get it from session state
                avatar_url = st.session_state.avatar_url
            response = requests.get(avatar_url)
            avatar = Image.open(BytesIO(response.content))
            
                #Create a mask of a filled circle
            mask = Image.new("L", avatar.size)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0) + avatar.size, fill=255)

            # Apply the mask to the avatar
            rounded_avatar = Image.new("RGBA", avatar.size)
            rounded_avatar.paste(avatar, mask=mask)
            avatar_width = 80  
            # Display the rounded avatar in Streamlit
            st.sidebar.image(rounded_avatar,  width=avatar_width)

        elif st.sidebar.checkbox('Upload an avatar?', value=False, key='upload_avatar_checkbox'):  
            # Avatar does not exist, offer file uploader if checkbox is checked
            uploaded_file = st.sidebar.file_uploader("Choose an avatar:", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                avatar = Image.open(uploaded_file)
                store_avatar(user_id, uploaded_file) 
                new_avatar_url = get_avatar_url(user_id)  
                st.session_state['avatar_url'] = new_avatar_url  
                st.session_state['avatar_chosen'] = True 
                st.rerun() 
        else:
            pass

            #--------Logout button-------------------
        if st.sidebar.button("Logout", key="logout_button"):
            st.session_state.pop('user', None)  # Remove user info from session_state
            st.session_state.authenticated = False  # Set authenticated state to False
            st.sidebar.info('Logged out successfully.')
            time.sleep(2)
            st.session_state.page = 'main'  # Redirect to main page
            st.rerun()
        
        if st.session_state.page != 'main':
            if st.sidebar.button("Back to Main Page"):
                st.session_state.page = 'main'
                st.rerun()

        
        if st.session_state.page != 'dashboard':
            if st.sidebar.button("Dashboard"):
                st.session_state.page = 'dashboard'
                st.rerun()

# Function to create a rounded avatar image
def create_rounded_avatar(avatar):
    mask = Image.new("L", avatar.size)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + avatar.size, fill=255)

    rounded_avatar = Image.new("RGBA", avatar.size)
    rounded_avatar.paste(avatar, mask=mask)
    return rounded_avatar



def display_footer():
    st.markdown(
        """
        <style>
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #FFFFFF;
                text-align: center;
                padding: 10px;
            }
        </style>
        <div class="footer">
            <p>&copy; 2023 Atomic Habits | Email: <a href="mailto:atomichabitsforlife@gmail.com">atomichabitsforlife@gmail.com</a> | 
            LinkedIn: <a href="https://www.linkedin.com/in/irina-getman-16871b165/" target="_blank">Irina Getman</a></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

def main():
  
      
    if 'page' not in st.session_state:
        st.session_state.page = 'main'  # Set default page if not set
    display_sidebar()  # Display the sidebar
        

    if st.session_state.page == 'authentication':
        authenticate()  # This calls the authenticate function from au.py
    elif st.session_state.page == 'main':
        display_main_page()
    elif st.session_state.page == 'dashboard':
        display_dashboard()  
        # ------Footer------
    display_footer()

      

if __name__ == "__main__":
    main()
