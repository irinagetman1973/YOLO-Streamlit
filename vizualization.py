import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from firebase_admin import db
from matplotlib.dates import DateFormatter
import io
import base64

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
  
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl', mode='xlsx') as writer:
        df.to_excel(writer, sheet_name='Sheet1')
    return output.getvalue()



def get_table_download_link(df, filename='data.xlsx', link_text='Download data as Excel'):
    excel_data = to_excel(df)
    b64 = base64.b64encode(excel_data).decode()  # Bytes to string
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{link_text}</a>'
    return href



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
      

      st.header(':green[Visualizations & Insights]')
      st.divider()

      st.subheader(':green[Data Table]')
      

      col1,col2 = st.columns(2)
      with col1:
            with st.expander("Click to see data"):
                  
                  st.dataframe(df)
                  if st.button('Download Dataframe as Excel'):
                        st.markdown(get_table_download_link(df), unsafe_allow_html=True)

      with col2:
            with st.expander("Filter Options", expanded=False) as filter_expander:
                  selected_value = st.selectbox('Choose a value', df['class_id'].unique())
                  filtered_df = df[df['class_id'] == selected_value]
                  st.subheader("Filtered Data Table")
                  st.dataframe(filtered_df)
                  if st.button('Download Filtered Dataframe as Excel'):
                        st.markdown(get_table_download_link(filtered_df), unsafe_allow_html=True)
     # Create an expander for Summary Statistics
      with st.expander("**Summary Statistics**", expanded=False):
      
            st.subheader(':green[Summary Statistics]')
            st.markdown(f"**Total inferences:** {len(df)}")
            st.markdown(f"**Unique objects detected:** {df['class_id'].nunique()}")
            st.markdown(f"**Average count of detections:** {df['count'].mean():.2f}")

            # Display Bar chart for object frequencies and Pie chart for proportion of detected objects side-by-side
            col1, col2 = st.columns(2)

            with col1:
                  st.write(':green[**Frequency of Detected Objects**]')
                  fig1, ax1 = plt.subplots()
                  df["class_id"].value_counts().plot(kind="bar", ax=ax1)
                  st.pyplot(fig1,use_container_width=True)

            with col2:
                  st.write(':green[**Proportion of Detected Objects**]')
                  fig2, ax2 = plt.subplots()
                  df["class_id"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax2)
                  st.pyplot(fig2,use_container_width=True)

      with st.expander("**Historical Detection Analysis**", expanded=False):
      
            st.subheader(':green[Detection Trends Over Time]')
            col1, col2, col3 = st.columns(3)
            
            # Convert the timestamp column to datetime format
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Specify colors and font
            colors = ['#E63946', '#1D3557', '#457B9D']
            font = {'family': 'serif',
                  'color':  'green',
                  'weight': 'normal',
                  'size': 12,
                  }

            with col1:
                  st.write(':green[**Total Detections Over Time**]')
                  # 1. Total Detections Over Time
                  fig, ax = plt.subplots(figsize=(10, 5))
                  df_grouped_by_timestamp = df.groupby("timestamp").size()
                  df_grouped_by_timestamp.plot(kind='line', color=colors[0], ax=ax)
                  # Setting the date format
                  date_format = DateFormatter("%Y-%m-%d %H:%M")
                  ax.xaxis.set_major_formatter(date_format)
                  # ax.set_title("Total Detections Over Time", fontdict=font)
                  ax.set_ylabel("Number of Detections", fontdict=font)
                  ax.set_xlabel("Timestamp", fontdict=font)
                  ax.tick_params(axis='x', rotation=45)
                  st.pyplot(fig)

            with col2:
                  # 2. Unique Objects Detected Over Time
                  st.write(':green[**Unique Objects Detected Over Time**]')
                  fig, ax = plt.subplots(figsize=(10, 5))
                  unique_objects = df.groupby("timestamp")["class_id"].nunique()
                  unique_objects.plot(kind='line', color=colors[1], ax=ax)
                  # Setting the date format
                  date_format = DateFormatter("%Y-%m-%d %H:%M")
                  ax.xaxis.set_major_formatter(date_format)
                  # ax.set_title("Unique Objects Detected Over Time", fontdict=font)
                  ax.set_ylabel("Number of Unique Objects", fontdict=font)
                  ax.set_xlabel("Timestamp", fontdict=font)
                  ax.tick_params(axis='x', rotation=45)
                  st.pyplot(fig)

            with col3:
                  # 3. Average Count of Detections Over Time
                  st.write(':green[**Average Count of Detections Over Time**]')
                  fig, ax = plt.subplots(figsize=(10, 5))
                  avg_detections = df.groupby("timestamp")["count"].mean()
                  avg_detections.plot(kind='line', color=colors[2], ax=ax)
                  # Setting the date format
                  date_format = DateFormatter("%Y-%m-%d %H:%M")
                  ax.xaxis.set_major_formatter(date_format)
                  ax.set_title("Average Count of Detections Over Time", fontdict=font)
                  ax.set_ylabel("Average Count", fontdict=font)
                  ax.set_xlabel("Timestamp", fontdict=font)
                  ax.tick_params(axis='x', rotation=45)
                  st.pyplot(fig)


      
      






            






