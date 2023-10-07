# avatar_manager.py
"""
-------------------------------------------------
   @File Name:     avatar_manager.py
   @Author:        Irina Getman
   @Date:          06/10/2023
   @Description:   store and render users' avatars 
                   from Firebase db 
-------------------------------------------------
"""

import streamlit as st
from firebase_admin import db, storage




        
def store_avatar(user_id, file):
    
      bucket = storage.bucket('capstone-c23c5.appspot.com') 
    
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
      
      # Reset the file object's position to the beginning of the stream
      file.seek(0)

    # Upload the file
      blob.upload_from_file(file, content_type=mime_type)

    
      # Make the file publicly accessible
      blob.make_public()
      avatar_url = blob.public_url

      # Update the user's avatar URL in the Firebase Realtime Database
      ref = db.reference(f'/users/{user_id}/avatar')
      ref.set(avatar_url)

    # Return the public URL to the file
      return avatar_url


def get_avatar_url(user_id, avatar_url=None):
      ref = db.reference(f'/users/{user_id}/avatar')
      if avatar_url is not None:
        # If avatar_url is provided, update it in the database
        try:
            ref.set(avatar_url)
        except Exception as e:
            st.error(f"Failed to update avatar URL in Firebase: {e}")
      else:
        # If avatar_url is not provided, get it from the database
        try:
            return ref.get()
        except Exception as e:
            st.error(f"Failed to get avatar URL from Firebase: {e}")
