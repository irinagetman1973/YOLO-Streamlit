"""
-------------------------------------------------
   @File Name:     au.py
   @Author:        Irina Getman
   @Date:          30/09/2023
   @Description:  login/signup using Firebase
-------------------------------------------------
"""



import os
import streamlit as st
import requests
import json
import firebase_admin
from firebase_admin import credentials, db, auth

from firebase_admin.auth import EmailAlreadyExistsError
import dotenv
import time

dotenv.load_dotenv()

#-----------Firebase configuration & Initialization---------
with open('config.json') as config_file:

    config_data = json.load(config_file)

firebaseConfig = config_data['firebaseConfig']

# cred = credentials.Certificate('C:\\Users\\irina\\capstone\\capstone-c23c5-4e7a43be2c53.json')

cred_path = os.environ.get('FIREBASE_CERT_PATH')
# st.write(type(cred_path))
cred_path = cred_path.strip('"') if isinstance(cred_path, str) else cred_path



db_url = os.environ.get('FIREBASE_DB_URL')
if db_url is None:
    raise ValueError("Environment variable FIREBASE_DB_URL is not set")
db_url = db_url.strip('"')


cred = credentials.Certificate(cred_path)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': db_url
    })

# ------------------------------
# User Authentication Interface
# ------------------------------

def authenticate():
    if st.sidebar.button("Back to Main Page"):
            st.session_state.page = 'main'
            st.rerun()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(':green[Login]')
        login_email = st.text_input(':blue[email]', placeholder='enter your email', key='login_email')
        login_password = st.text_input(':blue[password]', placeholder='enter your password', type='password', key='login_password')


        # def handle_login_bt()
        if st.button('Login'):
            login_user(login_email, login_password)
        

    with col2:
        st.subheader(':green[Sign Up]')
        email = st.text_input(':blue[email]', placeholder='enter your email', key='signup_email')
        username = st.text_input(':blue[username]', placeholder='enter your username', key='signup_username')
        password = st.text_input(':blue[password]', placeholder='enter your password', type='password', key='signup_password', help='Password must be at least 6 characters long.')

        if st.button('Create an account'):
            
            try:
                user = auth.create_user(email=email, password=password)
            # If user creation was successful, store the username in Firebase Realtime Database
                if user:
                
                        user_id = user.uid  
                        ref = db.reference(f'/users/{user_id}')
                        ref.set({'username': username})
                        # Store user info in session_state
                        st.session_state.user = {"username": username, "uid": user.uid}
                        st.success('Account created successfully! Please log in now.')
                        st.balloons()
                        time.sleep(3)
                        st.session_state.page = 'main'  # Redirect to main page
                        st.rerun()
                
            except EmailAlreadyExistsError:
                    st.error('The email address is already in use. Please use a different email address.')
            except Exception as e:
                    st.error(f'An error occurred: {e}')
          


# -----------------------
# User Login Function
# -----------------------

def handle_login_bt(user_data):
    # Retrieve the username from Firebase Realtime Database
    ref = db.reference(f'/users/{user_data["localId"]}')
    user_info = ref.get()
    username = user_info.get('username', 'Unknown User')
    # Store user info in session_state
    st.session_state.user = {"username": username, "uid": user_data["localId"]}
    st.session_state['authenticated'] = True
    
    st.success(f"Logged in successfully. Welcome, {username}!")
    time.sleep(3)
    st.session_state.page = 'main'  # Redirect to main page
    st.rerun()

def login_user(email, password):
    # Load firebaseConfig from config.json
    with open('config.json') as config_file:
        config_data = json.load(config_file)
    firebaseConfig = config_data['firebaseConfig']

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={firebaseConfig['apiKey']}"
    data = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    # Ensure user attribute is initialized in session_state
    if "user" not in st.session_state:
        st.session_state["user"] = None

    response = requests.post(url, data=data)
    if response.ok:
        user_data = response.json()
        handle_login_bt(user_data)
        

    else:
        st.warning('Login failed. Please check your credentials and try again.')



if __name__ == "__main__":
    authenticate()


