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
    # Assume you have already initialized Firebase Admin SDK as shown in previous examples

    # Get a reference to the Firebase Storage service
    bucket = storage.bucket()

    # Determine the file extension based on the mime type
    mime_type = file.type  # assuming file.type returns the mime type
    extension = ""
    if mime_type == "image/png":
        extension = ".png"
    elif mime_type == "image/jpeg":
        extension = ".jpg"

    if not extension:
        raise ValueError("Unsupported file type")

    # Create a unique file name based on user_id and file extension
    blob = bucket.blob(f'avatars/{user_id}{extension}')

    # Upload the file
    blob.upload_from_string(file.read(), content_type=mime_type)

    # Make the file publicly accessible
    blob.make_public()

    # Return the public URL to the file
    return blob.public_url


def get_avatar_url(user_id):
    ref = db.reference(f'/users/{user_id}/avatar')
    return ref.get()
