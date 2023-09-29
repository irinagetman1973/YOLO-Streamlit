import streamlit as st
import pandas as pd
import sqlite3

#--------database connection------------
conn = sqlite3.connect('products.db')

# Authentication (this is a simple example, in a real-world scenario, you'd use a more secure method)
def is_authenticated(username, password):
    return username == "data_analyst" and password == "secure_password"

# def run_custom_query(query, selected_version):
#     # Connect to your database
#     # with sqlite3.connect('products.db') as conn:
#     #     # Execute the query and fetch results
#     conn = sqlite3.connect('products.db')
#     try:
#         # Ensure model_version is always included in the results

#         results = pd.read_sql_query(query,conn)

#         #ensure model_version is always included in the results
#         if 'model_version' not in results.columns:
#             results['model_version'] = selected_version
                
#         return results
#     except Exception as e:
#             st.error(f"An error occurred: {e}")
#             return None

# Main app
# def main():
#     st.title("Custom Query Interface for Data Analysts")

#     # Authentication
#     username = st.sidebar.text_input("Username")
#     password = st.sidebar.text_input("Password", type="password")
    
#     if is_authenticated(username, password):
#         # Input for SQL Query
#         query = st.text_area("Input your SQL query here:")
#      # Model version selection
#     model_versions = ["yolov6", "yolov7", "yolov8", "yolo-nas"]
#     selected_version = st.selectbox("Choose a YOLO model version:", model_versions)

    
#     # Input for SQL Query
#     query = st.text_area("Input your SQL query here:", key="sql_query_input")

#     if st.button("Run Query"):
#             results = run_custom_query(query,selected_version)
#             if results is not None:
#                 st.dataframe(results, width=1000, height=600)
#     else:
#         st.error("Please provide valid credentials to access the custom query interface.")

# if __name__ == "__main__":
#     main()

def run_custom_query(query):
    # Connect to your database
    conn = sqlite3.connect('products.db')
    try:
        results = pd.read_sql_query(query, conn)
        return results
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def main():
    st.title("Custom Query Interface for Data Analysts")

    # Authentication
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if is_authenticated(username, password):
        # Model version selection
        model_versions = ["yolov6", "yolov7", "yolov8", "yolo-nas"]

        ##-----------------single selectbox----------------------
        # selected_version = st.selectbox("Choose a YOLO model version:", model_versions)

        #--------------multiselect box--------------------------
        selected_versions = st.multiselect("Choose YOLO model versions to compare:", model_versions)


        if st.button("Get Metrics"):

            if selected_versions:
                models_str = ', '.join([f"'{model}'" for model in selected_versions])
                query = f"SELECT model_version, precision, recall FROM yolo_metrics WHERE model_version IN ({models_str})"
                results = run_custom_query(query)
                
                if results is not None:
                        st.dataframe(results, width=1000)
                else:
                    st.warning("No data found for the selected models.")
            else:
                st.error("Please provide valid credentials to access the custom query interface.")

if __name__ == "__main__":
    main()