import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from firebase_admin import db

# Sample Firebase function to retrieve data (this needs to be connected to your Firebase instance)
def get_inference_data(user_id):
    ref = db.reference(f'/users/{user_id}/Inferences')

    # Fetch data
    data = ref.get()

    # Convert the fetched data into a list of dictionaries 
    return [item for item in data.values()]

    


def get_logged_in_user_id():
    """
    Check if the user's ID is already in the session state.
    """
    
    # If the user's details are in the session state, return the user ID.
    if "user" in st.session_state and "uid" in st.session_state["user"]:
        return st.session_state["user"]["uid"]

    # If not found, return None
    st.warning("User not logged in or user ID not available.")
    return None
  


def visualize_inferences():
      
      user_id = get_logged_in_user_id()
      
      
      if not user_id:
        st.warning("User not logged in or user ID not available.")
        return
      
      data = get_inference_data(user_id)
      
      # Flattening the nested structure
      flattened_data = []
      for entry in data:
            for item in entry:
                  timestamp = item.get('timestamp', None)
                  model = item.get('model', None)
                  
                  # If 'inference_details' exists and it's a list
                  if 'inference_details' in item and isinstance(item['inference_details'], list):
                        for detail in item['inference_details']:
                              detail['timestamp'] = timestamp
                              detail['model'] = model
                              flattened_data.append(detail)
      
      
      df = pd.DataFrame(flattened_data)
      

      st.subheader(':green[Visualizations & Insights]')
      st.divider()

      # Summary Statistics
      st.markdown("### Summary Statistics")
      st.markdown(f"**Total inferences:** {len(df)}")
      st.markdown(f"**Unique objects detected:** {df['class_id'].nunique()}")
      st.markdown(f"**Average count of detections:** {df['count'].mean():.2f}")

      # Display Bar chart for object frequencies and Pie chart for proportion of detected objects side-by-side
      col1, col2 = st.columns(2)

      with col1:
            st.subheader(':blue[Frequency of Detected Objects]')
            fig1, ax1 = plt.subplots()
            df["class_id"].value_counts().plot(kind="bar", ax=ax1)
            st.pyplot(fig1,use_container_width=True)

      with col2:
            st.subheader(':blue[Proportion of Detected Objects]')
            fig2, ax2 = plt.subplots()
            df["class_id"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax2)
            st.pyplot(fig2,use_container_width=True)

      # Line graph for detection trends over time
      # NOTE: This assumes you have a 'timestamp' field in your data
      st.subheader("Detection Trends Over Time")
      col1, col2, col3 = st.columns(3)

# --- First row ---

      with col1:
      # 1. Total Detections Over Time
            df_grouped_by_timestamp = df.groupby("timestamp").size()
            plt.figure(figsize=(10, 5))
            df_grouped_by_timestamp.plot(kind='line')
            plt.title("Total Detections Over Time")
            plt.ylabel("Number of Detections")
            plt.xticks(rotation=45)
            st.pyplot(plt)

      with col2:
      # 2. Unique Objects Detected Over Time
            unique_objects = df.groupby("timestamp")["class_id"].nunique()
            plt.figure(figsize=(10, 5))
            unique_objects.plot(kind='line')
            plt.title("Unique Objects Detected Over Time")
            plt.ylabel("Number of Unique Objects")
            plt.xticks(rotation=45)
            st.pyplot(plt)

      with col3:
      # 3. Average Count of Detections Over Time
            avg_detections = df.groupby("timestamp")["count"].mean()
            plt.figure(figsize=(10, 5))
            avg_detections.plot(kind='line')
            plt.title("Average Count of Detections Over Time")
            plt.ylabel("Average Count")
            plt.xticks(rotation=45)
            st.pyplot(plt)

      # Heatmap for spatial data
      # NOTE: This assumes you have 'x' and 'y' coordinates for each detection
      st.subheader("Detection Heatmap")

      # Assuming the column name is 'coordinates' which contains lists
      if 'coordinates' in df.columns:
      # Extracting x and y
            df['x'] = df['coordinates'].apply(lambda coord: coord[0] if len(coord) > 1 else None)
            df['y'] = df['coordinates'].apply(lambda coord: coord[1] if len(coord) > 1 else None)

            # KDE plot
            fig, ax = plt.subplots()
            sns.kdeplot(data=df, x="x", y="y", cmap="Reds", fill=True, ax=ax)
            st.pyplot(fig)
      else:
            st.warning("Coordinates not found in the data.")




