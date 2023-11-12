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
from matplotlib.figure import Figure

@st.cache_data
def get_inference_data(user_id):
    ref = db.reference(f'/users/{user_id}/Inferences')

    # Fetch data
    data = ref.get()
    # Check if data exists
    if data is None:
        return []

    # Convert the fetched data into a list of dictionaries 
    return [item for item in data.values()]

def get_plot_as_image(fig: Figure):
    """Converts a matplotlib Figure to an image bytes (PNG format)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf.getvalue()



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

def apply_filters(df, selected_models, selected_date_range, x1_range, y1_range, x2_range, y2_range, confidence_min, confidence_max):
    # Single place to apply all filters
    filtered_df = df[
        (df['model'].isin(selected_models)) &
        (df['timestamp'].dt.date >= selected_date_range[0]) &
        (df['timestamp'].dt.date <= selected_date_range[1]) &
        (df['x1'].between(*x1_range)) &
        (df['y1'].between(*y1_range)) &
        (df['x2'].between(*x2_range)) &  
        (df['y2'].between(*y2_range)) &
        (df['confidence'].between(confidence_min, confidence_max))
    ]
    return filtered_df

def parse_coordinates(coord_str):
            # Initialize a default value for coordinates
            default_coord = [0, 0, 0, 0] # Example default value
            # Check if 'coord_str' is a string and not NaN or any other type
            if isinstance(coord_str, str):
                  # Split the string by comma and convert to integers if possible
                  try:
                        return [int(part) for part in coord_str.split(',')]
                  except ValueError:
                        # Return the default value if conversion fails
                        return default_coord
            else:
                  # Return the default value if coord_str is not a string
                  return default_coord

def visualize_inferences():

      
      
     
      user_id = get_logged_in_user_id()
      
      if not user_id:
            st.warning("User not logged in or user ID not available.")
            return
            
      data = get_inference_data(user_id)

      if not data:  # Added data validation
        st.markdown("## 📊 Visualizations & Insights")
        st.markdown("### 🙈 Oops!")
        st.write("It seems there's an issue with your data or you don't have any data uploaded yet.")
        st.write("Upload your data and start seeing insights")
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


      

      # Filtering options
      st.sidebar.header('Filter Options')

      # Model filter
      model_options = df['model'].unique().tolist()
      selected_models = st.sidebar.multiselect('Select Model(s)', model_options, default=model_options)

      # Convert the 'timestamp' to a datetime object and then to a date object
      df['timestamp'] = pd.to_datetime(df['timestamp'])

      # Date filter
      date_min = df['timestamp'].min()
      date_max = df['timestamp'].max()
      selected_date_range = st.sidebar.date_input('Select Date Range', [date_min, date_max])

      
           

     
      


      # Apply the function to the 'coordinates' column
      df['parsed_coordinates'] = df['coordinates'].apply(parse_coordinates)

      # Create separate columns for each coordinate component
      df['x1'] = df['parsed_coordinates'].apply(lambda coords: coords[0] if coords else None)
      df['y1'] = df['parsed_coordinates'].apply(lambda coords: coords[1] if coords else None)
      df['x2'] = df['parsed_coordinates'].apply(lambda coords: coords[2] if coords else None)
      df['y2'] = df['parsed_coordinates'].apply(lambda coords: coords[3] if coords else None)

      # Slider for x1 coordinate
      x1_min, x1_max = df['x1'].min(), df['x1'].max()
      x1_range = st.sidebar.slider('Select X1 Coordinate Range', x1_min, x1_max, (x1_min, x1_max))

      # Slider for y1 coordinate
      y1_min, y1_max = df['y1'].min(), df['y1'].max()
      y1_range = st.sidebar.slider('Select Y1 Coordinate Range', y1_min, y1_max, (y1_min, y1_max))

      # Slider for x2 coordinate
      x2_min, x2_max = df['x2'].min(), df['x2'].max()
      x2_range = st.sidebar.slider('Select X2 Coordinate Range', x2_min, x2_max, (x2_min, x2_max))

      # Slider for y2 coordinate
      y2_min, y2_max = df['y2'].min(), df['y2'].max()
      y2_range = st.sidebar.slider('Select Y2 Coordinate Range', y2_min, y2_max, (y2_min, y2_max))


      # Confidence filter
      confidence_min, confidence_max = st.sidebar.slider('Select Confidence Score Range', 0.0, 1.0, (0.0, 1.0))

      
      if len(selected_date_range) == 2:
            # Apply filters to the dataframe
            filtered_df = df[
            (df['model'].isin(selected_models)) &
            (df['timestamp'].dt.date >= selected_date_range[0]) &
            (df['timestamp'].dt.date <= selected_date_range[1]) &
            (df['x1'].between(*x1_range)) &
            (df['y1'].between(*y1_range)) &
            (df['x2'].between(*x2_range)) &  
            (df['y2'].between(*y2_range)) &
            (df['confidence'].between(confidence_min, confidence_max))
            ]

      # Create an expander for Summary Statistics
      with st.expander("**Summary Statistics** ", expanded=True):
      
            st.subheader(':blue[Summary Statistics] 📈')
            st.markdown(f"**Total inferences:** {len(df)}")
            st.markdown(f"**Unique objects detected:** {df['class_id'].nunique()}")
            st.markdown(f"**Average count of detections:** {df['count'].mean():.2f}")

            # Display Bar chart for object frequencies and Pie chart for proportion of detected objects side-by-side
            col1, col2 = st.columns(2)

            with col1:
                  # Count the occurrences of each class_id and sort them in descending order
                  class_id_counts = df["class_id"].value_counts()

                  # Select the top 12 most frequent class_ids
                  top_12_class_ids = class_id_counts.head(12)
                  st.write(':blue[**📊 Frequency of Detected Objects**]')
                  fig1, ax1 = plt.subplots()
                  top_12_class_ids.plot(kind="bar", ax=ax1)
                  st.pyplot(fig1,use_container_width=True)

            with col2:
                  
                  # Count the occurrences of each class_id and sort them in descending order
                  class_id_counts = df["class_id"].value_counts()

                  # Select the top 12 most frequent class_ids
                  top_12_class_ids = class_id_counts.head(12)

                  # Plotting the pie chart for the top 12 class_ids
                  st.write('🍩 :blue[**Proportion of Detected Objects**]')
                  fig2, ax2 = plt.subplots()
                  top_12_class_ids.plot.pie(autopct="%1.1f%%", ax=ax2)
                  st.pyplot(fig2, use_container_width=True)
      
      

      
     
      # Layout for data table and filter options
      col1, col2 = st.columns(2)

      with col1:
            with st.expander("Data", expanded=True):
                  st.subheader('Data Table')
                  st.dataframe(df)
                  st.download_button(
                        label="Download Data as Excel",
                        data=get_table_download_link(df),
                        file_name='full_data.xlsx',
                        mime='application/vnd.ms-excel'
                  )

      with col2:
            with st.expander("Filter Options", expanded=False):
                  selected_value = st.selectbox('Choose a value', df['class_id'].unique())
                  filtered_df = df[df['class_id'] == selected_value]
                  st.subheader("Filtered Data Table")
                  st.dataframe(filtered_df)
                  st.download_button(
                        label="Download Filtered Data as Excel",
                        data=get_table_download_link(filtered_df),
                        file_name='filtered_data.xlsx',
                        mime='application/vnd.ms-excel'
                  )

      
      # Visualization toggle
      show_filtered_viz = st.checkbox('Show Visualizations for Filtered Data')
      if show_filtered_viz and 'class_id' in df.columns:
            with st.container():
                  st.subheader(f'Visualizations for {selected_value}')
                  
                  # Organizing plots in columns
                  col1, col2, col3, col4 = st.columns(4)

                  with col1:
                        # Plot for Total Detections Over Time for the selected class_id
                        fig, ax = plt.subplots(figsize=(10, 5))
                        filtered_df.set_index('timestamp')['count'].plot(ax=ax)
                        ax.set_title(f"Detections of {selected_value} Over Time")
                        ax.set_ylabel('Count')
                        ax.set_xlabel('Timestamp')
                        ax.tick_params(axis='x', rotation=45)
                        st.pyplot(fig)
                        # Download button for the Total Detections Over Time plot
                        st.download_button(
                        label="Download Total Detections Plot",
                        data=get_plot_as_image(fig),
                        file_name='total_detections.png',
                        mime='image/png'
                        )

                  with col2:
                        # Histogram for Confidence Score Distribution
                        fig, ax = plt.subplots(figsize=(10, 5))
                        filtered_df['confidence'].plot(kind='hist', ax=ax, bins=20)
                        ax.set_title(f"Confidence Score Distribution for {selected_value}")
                        ax.set_ylabel('Frequency')
                        ax.set_xlabel('Confidence Score')
                        st.pyplot(fig)
                        # Download button for the Confidence Score Distribution plot
                        st.download_button(
                        label="Download Confidence Distribution Plot",
                        data=get_plot_as_image(fig),
                        file_name='confidence_distribution.png',
                        mime='image/png'
                        )
                  
                  col3, col4 = st.columns(2)
                  
                  with col3:
                        # Boxplot for Detection Count Spread
                        fig, ax = plt.subplots(figsize=(10, 5))
                        filtered_df.boxplot(column=['count'], ax=ax)
                        ax.set_title(f"Detection Count Spread for {selected_value}")
                        ax.set_ylabel('Count')
                        st.pyplot(fig)
                        # Download button for the Detection Count Spread plot
                        st.download_button(
                              label="Download Detection Count Spread Plot",
                              data=get_plot_as_image(fig),
                              file_name='detection_count_spread.png',
                              mime='image/png'
                        )
                  
                  with col4:
                        # Scatter Plot for Confidence vs. Count
                        fig, ax = plt.subplots(figsize=(10, 5))
                        filtered_df.plot(kind='scatter', x='confidence', y='count', ax=ax)
                        ax.set_title(f"Confidence vs. Count for {selected_value}")
                        ax.set_ylabel('Count')
                        ax.set_xlabel('Confidence Score')
                        st.pyplot(fig)
                        # Download button for the Confidence vs. Count plot
                        st.download_button(
                        label="Download Confidence vs Count Plot",
                        data=get_plot_as_image(fig),
                        file_name='confidence_vs_count.png',
                        mime='image/png'
                        )
                        
                 



                     

 #---------------Time Series Visualization------------------------------------#                 
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


      
      






            






