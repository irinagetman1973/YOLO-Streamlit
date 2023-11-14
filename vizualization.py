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


# global df
# df = None

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
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        
    output.seek(0)  # Go to the beginning of the stream
    return output.getvalue()


def get_table_download_link(df, filename='data.xlsx', link_text='Download data as Excel'):
    excel_data = to_excel(df)
    b64 = base64.b64encode(excel_data).decode()  # Bytes to string
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{link_text}</a>'
    return href



def parse_coordinates(coord_str):
    # Initialize a default dictionary for coordinates with keys
    default_coord = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
    # Check if 'coord_str' is a string and not NaN or any other type
    if isinstance(coord_str, str):
        # Split the string by comma and convert to integers if possible
        try:
            parts = [int(part) for part in coord_str.split(',')]
            return {'x1': parts[0], 'y1': parts[1], 'x2': parts[2], 'y2': parts[3]}
        except (ValueError, IndexError):
            # Return the default dictionary if conversion fails or not enough parts
            return default_coord
    else:
        # Return the default dictionary if coord_str is not a string
        return default_coord


def preprocess_data(data):
    # Initialize a list to store the flattened data
    flattened_data = []

    # Loop through each entry in the data
    for entry in data:
        for item in entry:
            # Extract the timestamp and convert it to a readable date, if it exists
            timestamp = item.get('timestamp', None)
            readable_date = None
            if timestamp:
                readable_date = datetime.utcfromtimestamp(int(timestamp) / 1000).strftime('%Y-%m-%d %H:%M:%S')
            
            # Extract other details from the item
            model = item.get('model', None)
            inference_details = item.get('inference_details', {})
            
            # Check if inference_details is a dictionary and proceed
            if isinstance(inference_details, dict):
                # Add the timestamp and model to the details
                inference_details['timestamp'] = readable_date
                inference_details['model'] = model
                
                # Extract and clean 'count' data
                count = inference_details.get('count', 1)
                inference_details['count'] = parse_count(count)
                
            #     # Extract and clean 'coordinates' data, if present
            #     if 'coordinates' in inference_details:
            #         # The parse_coordinates now returns a dictionary
            #         coords_dict = parse_coordinates(inference_details['coordinates'])
            #         inference_details.update(coords_dict)
                
                # Append the cleaned details to the flattened_data list
                flattened_data.append(inference_details)
    
    # Convert the flattened data into a pandas DataFrame
    df = pd.DataFrame(flattened_data)
    
    # Convert the 'timestamp' column to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Return the processed DataFrame
    return df


def parse_count(count):
    # Ensure count is an integer, even when it's a list or in an unexpected format
    if isinstance(count, list) and count and isinstance(count[0], (int, float)):
        return int(count[0])
    elif isinstance(count, (int, float)):
        return int(count)
    return 1  # Default to 1 if count is missing or in an unexpected format


def generate_visualizations(df):
    # Check if the DataFrame is empty
    if df.empty:
        st.write("No data to visualize.")
        return

    # Visualization 1: Bar Chart of Object Frequencies
    st.subheader('Object Frequencies')
    fig, ax = plt.subplots()
    df['class_id'].value_counts().head(10).plot(kind='bar', ax=ax)
    ax.set_ylabel('Frequency')
    ax.set_title('Top 10 Detected Objects')
    st.pyplot(fig)

    # Visualization 2: Line Chart of Detections Over Time
    st.subheader('Detections Over Time')
    fig, ax = plt.subplots()
    df.set_index('timestamp')['count'].resample('D').sum().plot(ax=ax)
    ax.set_ylabel('Number of Detections')
    ax.set_title('Daily Detections Trend')
    st.pyplot(fig)

    # Visualization 3: Confidence Score Distribution
    st.subheader('Confidence Score Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df, x='confidence', bins=20, kde=True, ax=ax)
    ax.set_title('Confidence Scores Histogram')
    st.pyplot(fig)

    # Visualization 4: Heatmap of Detections by Model and Class
    if 'model' in df.columns and 'class_id' in df.columns:
        st.subheader('Heatmap of Detections by Model and Class')
        pivot_table = pd.pivot_table(df, values='count', index='model', columns='class_id', aggfunc=np.sum, fill_value=0)
        fig, ax = plt.subplots()
        sns.heatmap(pivot_table, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
        ax.set_title('Detections by Model and Class')
        st.pyplot(fig)

    pass

def get_filter_inputs(df, identifier):
    # Sidebar interface for filter inputs
    try:
        if 'selected_models' not in st.session_state:
            st.session_state['selected_models'] = []

        # Model filter
        model_options = df['model'].unique().tolist()

        # Use the identifier to create a truly unique key for the multiselect widget
        st.sidebar.divider()
       
       

        st.sidebar.markdown("### 📇 Select the parameters below to filter the dataset.")
        selected_models = st.sidebar.multiselect(
            'Select Model(s)',
            model_options,
            default=st.session_state['selected_models']
        )

        st.session_state['selected_models'] = selected_models

        # Date filter
        df['timestamp'] = pd.to_datetime(df['timestamp'])  # Ensure timestamp is in datetime format
        date_min = df['timestamp'].min().date()
        date_max = df['timestamp'].max().date()
        selected_date_range = st.sidebar.date_input('Select Date Range', [date_min, date_max])

        # Return a dictionary of filter options
        filter_options = {
            'selected_models': selected_models,
            'selected_date_range': selected_date_range
        }

        return filter_options

    except Exception as e:
        st.error(f"Error in get_filter_inputs: {e}")
        # If there's an error, you can return an empty dictionary or some default values
        return {}

def apply_filters():
      # Check if 'df' and 'filter_options' are available in the session state
      # st.write("apply_filters started...")
      if 'df' in st.session_state and 'filter_options' in st.session_state:
            df = st.session_state['df']
            filter_options = st.session_state['filter_options']

            # st.write("Debug - apply_filters called")  # Confirm the function is called

            # Unpack filter options
            selected_models = filter_options['selected_models']
            selected_date_range = filter_options['selected_date_range']

            # Apply filters to the DataFrame
            # Only applying model and date range filters
            filtered_df = df[
                  df['model'].isin(selected_models) &
                  df['timestamp'].dt.date.between(*selected_date_range)
            ]
            
            # Update the session state with the filtered dataframe
            st.session_state['filtered_data'] = filtered_df
            st.session_state['filtered'] = True

            # Provide feedback about the operation
            if filtered_df.empty:
                  st.sidebar.warning("No data matches the filters.")
                  # st.write("Debug - No data matches the filters.")  # Debug print
            else:
                  st.sidebar.success(f"Filtered data contains {len(filtered_df)} rows.")
                  # st.write(f"Debug - Filtered data contains {len(filtered_df)} rows.")  # Debug print

            # Refresh the page to reflect changes
            # st.experimental_rerun()

      else:
            st.sidebar.error("Data or filter options are not set in the session state.")
            # st.write("Debug - Data or filter options are not set in the session state.")  # Debug print




def update_filtered_data():
    # Check if the dataframe is available in the session state
    if 'df' in st.session_state and 'class_id' in st.session_state['df']:
        # Get the current selection from session state
        selected_value = st.session_state.class_id_select
        # Update the filtered dataframe in the session state
        st.session_state.filtered_data = st.session_state['df'][st.session_state['df']['class_id'] == selected_value]



    

def visualize_inferences():

      st.session_state['filtered'] = False
      # Ensure that 'filtered_data' is initialized in the session state
      if 'filtered_data' not in st.session_state:
            st.session_state.filtered_data = pd.DataFrame()
     
     
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
 
      

      st.header('📊 :green[Visualizations & Insights]')
      st.divider()
      
      df = preprocess_data(data)
      # Store the processed DataFrame in the session state
      st.session_state.df = df
      if df.empty:
        st.error("Processed data is empty.")
        return

        
      

    

      # Create an expander for Summary Statistics
      with st.expander("**Summary Statistics** ", expanded=False):
      
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
      
      
      filter_options = get_filter_inputs(df, 'inferences')
      st.session_state['filter_options'] = filter_options
      # Assuming df is already in session_state and has been preprocessed
      if 'filtered_data' not in st.session_state:
            st.session_state['filtered_data'] = st.session_state.get('df', pd.DataFrame())
     

      
      # When the user clicks the 'Apply Filters' button, apply the filters.
      if st.sidebar.button('Apply Filters'):
             apply_filters()

            
    


      

      # Layout for data table and filter options
      col1, col2 = st.columns(2)

      with col1:
            with st.expander("Data", expanded=True):
                  st.subheader("📜 :blue[**Data table**]")
                  st.dataframe(df)
                  st.download_button(
                        label="Download Data as Excel",
                        data=to_excel(df),
                        file_name='full_data.xlsx',
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                  )
      
      with col2:
         
            with st.expander("Filter Options", expanded=True):
                  # Ensure 'df' and 'class_id' are in session state and have data before creating the selectbox
                  if 'df' in st.session_state and 'class_id' in st.session_state.df.columns and not st.session_state.df.empty:
                        # class_ids = st.session_state.df['class_id'].unique()
                        # selected_value = st.selectbox(
                        # 'Choose a value',
                        # options=class_ids,
                        # index=0,
                        # key='class_id_select',
                        # on_change=update_filtered_data  # This line ensures update_filtered_data is called on value change
                        # )

                        st.subheader("🔍:blue[**Filtered Data Table**]")
                        st.dataframe(st.session_state.filtered_data)  # Display the filtered data from session state

                        # Download button for filtered data
                        st.download_button(
                        label="Download Excel file",
                        data=to_excel(df),
                        file_name="data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                  else:
                        st.write("Please select a class ID to display the data.")
                        
      

      # # Use the filtered DataFrame for further processing or visualization
      # st.write(filtered_df.head())
      # if 'filtered_data' in st.session_state and not st.session_state['filtered_data'].empty:
      #       st.dataframe(st.session_state['filtered_data'].describe())
      # else:
      #       st.write("No data available to display.")
      

      
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


      
      






            






