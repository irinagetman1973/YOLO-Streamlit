import streamlit as st
import firebase_admin

from firebase_admin import credentials
from firebase_admin import auth

cred = credentials.Certificate('capstone-c23c5-4e7a43be2c53.json')
firebase_admin.initialize_app(cred)