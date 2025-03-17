import streamlit as st
import pandas as pd
import numpy as np
import pickle  # For loading the trained model
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

# Load the trained model
with open('training_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load Data
data = pd.read_csv('cleaned_data.csv')

# App Mode Selection
mode = st.radio("Select App Mode:", ["Light Mode", "GitHub Mode"])
# Apply color theme based on mode
if mode == "GitHub Mode":
    plt.style.use("dark_background")  # Dark mode for GitHub theme
    sns.set_palette("Greens")  # Green color theme
else:
    plt.style.use("default")  # Light mode

# Clean the 'State' column
data['State'] = data['State'].str.strip().str.title().fillna('Unknown')

# Title and description
st.title("College Recommendation Predictor")
st.write("Enter the following details to predict the recommended college category.")

# **User Input for Model Prediction**
average_rating = st.number_input("Average Rating", min_value=0.0, max_value=10.0, step=0.1)
placement_fee_ratio = st.number_input("Placement vs Fee Ratio", min_value=0.0, step=0.00001)
ug_fee_scaled = st.number_input("UG Fee (Scaled)", min_value=0.0, max_value=1.0, step=0.001)
pg_fee_scaled = st.number_input("PG Fee (Scaled)", min_value=0.0, max_value=1.0, step=0.001)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[average_rating, placement_fee_ratio, ug_fee_scaled, pg_fee_scaled]])
    prediction = model.predict(input_data)[0]
    
    st.success(f"The predicted college category is: **{prediction}**")

# **Area Selection for Graphs**
st.write("### Explore Colleges by Area")
area_options = sorted(data['State'].unique().tolist())  # Sorted list for better UX
area = st.selectbox("Select Area", area_options)

# Filter Data
filtered_data = data[data['State'] == area]

# **Check if filtered_data is empty**
if filtered_data.empty:
    st.warning(f"No data available for {area}. Try selecting a different area.")
else:
    # **Graphs for Selected Area**
    st.write(f"### Colleges in {area}")

    # **Average Rating Comparison (Sorted)**
    st.write("### Average Rating Comparison")
    plt.figure(figsize=(12, 6))  # Bigger figure for better visibility
    sorted_data = filtered_data.sort_values(by='Average Rating', ascending=False)
    sns.barplot(x='College Name', y='Average Rating', data=sorted_data)
    plt.xticks(rotation=45, ha='right')  # Better visibility
    plt.xlabel("College Name")  # Add label
    plt.ylabel("Average Rating")  # Add label
    st.pyplot(plt)

    # **Placement vs Fee Ratio (Sorted)**
    st.write("### Placement vs Fee Ratio")
    plt.figure(figsize=(12, 6))
    sorted_data = filtered_data.sort_values(by='Placement vs Fee Ratio', ascending=False)
    sns.barplot(x='College Name', y='Placement vs Fee Ratio', data=sorted_data)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("College Name")
    plt.ylabel("Placement vs Fee Ratio")
    st.pyplot(plt)

    # **UG Fee Distribution (Histogram)**
    st.write("### UG Fee Distribution")
    plt.figure(figsize=(12, 6))
    sns.histplot(filtered_data['UG fee (tuition fee)'], kde=True, bins=15)  # More bins for better visualization
    plt.xlabel("UG Fee (Tuition Fee)")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    # **UG Fee (Scaled) (Sorted)**
    st.write("### UG Fee (Scaled)")
    plt.figure(figsize=(12, 6))
    sorted_data = filtered_data.sort_values(by='UG fee (scaled)', ascending=False)
    sns.barplot(x='College Name', y='UG fee (scaled)', data=sorted_data)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("College Name")
    plt.ylabel("UG Fee (Scaled)")
    st.pyplot(plt)

    # **PG Fee (Scaled) (Sorted)**
    st.write("### PG Fee (Scaled)")
    plt.figure(figsize=(12, 6))
    sorted_data = filtered_data.sort_values(by='PG fee (scaled)', ascending=False)
    sns.barplot(x='College Name', y='PG fee (scaled)', data=sorted_data)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("College Name")
    plt.ylabel("PG Fee (Scaled)")
    st.pyplot(plt)

# **Footer**
footer = """
    <style>
    .footer{
        position:fixed;
        left:0;
        bottom:0;
        width:100%;
        background-color:#90EE90;
        color: #6c757d;
        font-size:14px;
        padding:15px;
        text-align:center;
    }
    .footer a{
        color:black;
        text-decoration:none;
        font-weight:bold;
    }
    </style>
    <div class="footer">
        Developed with ❤️ by Arshia Singh using Streamlit
    </div>
"""

st.markdown(footer,unsafe_allow_html=True)
