

import streamlit as st
import uuid
from dashboard import display_dashboard

def display_authentication_page():
    st.title("YOLO Model Evaluator")

    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = True

    # Sidebar for navigation
    if st.session_state.authenticated:
        menu = ["Home", "Dashboard", "Logout"]
    else:
        menu = ["Home", "Authentication"]
    
    choice = st.sidebar.selectbox("Menu", menu, key="auth_page_selectbox")

    if choice == "Home":
        st.session_state.page = 'main'  # Set session state to main
        return  # Exit the function to allow main() to handle page display

    elif choice == "Authentication":
        authenticate_user()

    elif choice == "Dashboard":
        if st.session_state.authenticated:
            st.session_state.page = 'dashboard'
            return  # Exit the function to allow main() to handle page display
        else:
            st.warning("Please authenticate to access the dashboard.")
            authenticate_user()

    elif choice == "Logout":
        st.session_state.authenticated = False
        st.session_state.page = 'main'  # Redirect to main page after logout
        return  # Exit the function to allow main() to handle page display

def authenticate_user():
    st.subheader("Login to Access Dashboard")

    # Input fields for username and password
    username = st.text_input("Username", key="auth_username")
    password = st.text_input("Password", type="password", key="auth_password")

    # Login button
    if st.button("Login", key="auth_login_button"):
        st.write("Login button pressed!")  # Debug line
        # Hardcoded authentication logic (for demonstration purposes)
        if username == "admin" and password == "password":
            st.session_state.authenticated = True
            st.write("Authentication state:", st.session_state.authenticated)  # Debug line
            st.success("Logged in successfully!")
            st.session_state.page = 'dashboard'  # Redirect to dashboard after successful login
            return  # Exit the function to allow main() to handle page display
        else:
            st.error("Incorrect credentials")

    st.write("Forgot password? [Recover](#)")  # Placeholder link for password recovery

display_authentication_page()































