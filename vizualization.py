import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from firebase_admin import db
from matplotlib.dates import DateFormatter
import io
import base64
from datetime import datetime


# Sample Firebase function to retrieve data (this needs to be connected to your Firebase instance)
def get_inference_data(user_id):
    ref = db.reference(f'/users/{user_id}/Inferences')

    # Fetch data
    data = ref.get()
    # Check if data exists
    if data is None:
        return []

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

      if not data:
            st.write("No data available.")
            return

      # Flattening the nested structure
      flattened_data = []
      for entry in data:
            for item in entry:
                  timestamp = item.get('timestamp', None)
                  if timestamp:
                        readable_date = datetime.utcfromtimestamp(int(timestamp) / 1000).strftime('%Y-%m-%d %H:%M:%S')
                  else:
                        readable_date = None
                  model = item.get('model', None)

                  inference_details = item.get('inference_details', {})
                  if isinstance(inference_details, dict):
                        inference_details['timestamp'] = readable_date
                        inference_details['model'] = model

                        # Robust handling of 'count'
                        count = inference_details.get('count', 1)
                        if isinstance(count, list):
                              if count and isinstance(count[0], (int, float)):  # Check if first element is a number
                                    inference_details['count'] = int(count[0])
                              else:
                                    inference_details['count'] = 1  # Default value if list is empty or non-numeric
                        elif isinstance(count, (int, float)):
                              inference_details['count'] = int(count)
                        else:
                              inference_details['count'] = 1  # Default value if count is not numeric

                        flattened_data.append(inference_details)



      df = pd.DataFrame(flattened_data)
      

      st.header('📊 :green[Visualizations & Insights]')
      st.divider()

      # Create an expander for Summary Statistics
      with st.expander("**Summary Statistics** ", expanded=True):
      
            st.subheader(':blue[Summary Statistics] 📈')
            st.markdown(f"**Total inferences:** {len(df)}")
            st.markdown(f"**Unique objects detected:** {df['class_id'].nunique()}")
            st.markdown(f"**Average count of detections:** {df['count'].mean():.2f}")

            # Display Bar chart for object frequencies and Pie chart for proportion of detected objects side-by-side
            col1, col2 = st.columns(2)

            with col1:
                  st.write(':blue[**Frequency of Detected Objects**]')
                  fig1, ax1 = plt.subplots()
                  df["class_id"].value_counts().plot(kind="bar", ax=ax1)
                  st.pyplot(fig1,use_container_width=True)

            with col2:
                  st.write(':blue[**Proportion of Detected Objects**]')
                  fig2, ax2 = plt.subplots()
                  df["class_id"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax2)
                  st.pyplot(fig2,use_container_width=True)

      
      

      col1,col2 = st.columns(2)
      with col1:
            with st.expander("Data"):
                  st.subheader('📜 :blue[Data Table]')
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
     

      with st.expander("**Historical Detection Analysis**", expanded=False):
      
            st.subheader(':blue[Detection Trends Over Time] 📅')
            col1, col2, col3 = st.columns(3)
            
            # Convert the timestamp column to datetime format
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Specify colors and font
            colors = ['#E63946', '#1D3557', '#457B9D']
            font = {'family': 'serif',
                  'color':  'green',
                  'weight': 'normal',
                  'size': 12,
                  }

            

            with col1:
                  st.write(':green[**Total Detections Over Time by Model**]')
                  fig, ax = plt.subplots(figsize=(10, 5))
                  for model in df['model'].unique():
                        df_model = df[df['model'] == model]
                        df_model_grouped = df_model.groupby("timestamp").size()
                        df_model_grouped.plot(kind='line', ax=ax, label=model)
                  ax.legend()
                  ax.set_ylabel("Number of Detections", fontdict=font)
                  ax.set_xlabel("Date", fontdict=font)
                  ax.tick_params(axis='x', rotation=45)
                  st.pyplot(fig)

           

            if 'class_id' in df.columns:
                  with col2:
                        st.write(':green[**Unique Objects Detected Over Time**]')
                        fig, ax = plt.subplots(figsize=(10, 5))
                        unique_objects = df.groupby("timestamp")["class_id"].nunique()
                        unique_objects.plot(kind='line', color=colors[1], ax=ax)
                        ax.set_ylabel("Number of Unique Objects", fontdict=font)
                        ax.set_xlabel("Date", fontdict=font)
                        ax.tick_params(axis='x', rotation=45)
                        st.pyplot(fig)

            with col3:
                  st.write(':green[**Model Utilization Over Time**]')
                  fig, ax = plt.subplots(figsize=(10, 5))
                  df_model_usage = df.groupby(['timestamp', 'model']).size().unstack().fillna(0)
                  df_model_usage.plot(kind='line', ax=ax)
                  ax.set_ylabel("Usage Frequency", fontdict=font)
                  ax.set_xlabel("Date", fontdict=font)
                  ax.tick_params(axis='x', rotation=45)
                  st.pyplot(fig)

            # with col3:
            #       # 3. Average Count of Detections Over Time
            #       st.write(':green[**Average Count of Detections Over Time**]')
            #       fig, ax = plt.subplots(figsize=(10, 5))
            #       avg_detections = df.groupby("timestamp")["count"].mean()
            #       avg_detections.plot(kind='line', color=colors[2], ax=ax)
            #       # Setting the date format
            #       date_format = DateFormatter("%Y-%m-%d %H:%M")
            #       ax.xaxis.set_major_formatter(date_format)
            #       ax.set_title("Average Count of Detections Over Time", fontdict=font)
            #       ax.set_ylabel("Average Count", fontdict=font)
            #       ax.set_xlabel("Timestamp", fontdict=font)
            #       ax.tick_params(axis='x', rotation=45)
            #       st.pyplot(fig)


      
      






            






