# avatar_manager.py

import streamlit as st
import firebase_admin
from firebase_admin import db, storage

def upload_and_store_avatar(user_id):
    uploaded_file = st.sidebar.file_uploader("Choose an avatar image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        # You may want to use a library or service like Firebase Storage to store the image
        # Assume store_avatar returns the URL where the avatar is stored
        avatar_url = store_avatar(user_id, uploaded_file)
        
        # Now store the avatar_url in the Firebase Realtime Database
        ref = db.reference(f'/users/{user_id}/avatar')
        ref.set(avatar_url)
        
def store_avatar(user_id, file):
    # Implement your logic to store the avatar and return the URL
    pass

def get_avatar_url(user_id):
    ref = db.reference(f'/users/{user_id}/avatar')
    return ref.get()
