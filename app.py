import streamlit as st
import pandas as pd
import numpy as np
import pickle  # For loading the trained model
import seaborn as sns  # Fixed alias
import matplotlib.pyplot as plt

# Load the trained model
with open('training_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load Data
data = pd.read_csv('cleaned_data.csv')

# Clean the 'State' column to avoid matching issues
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

    # **Average Rating Comparison**
    st.write("### Average Rating Comparison")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='College Name', y='Average Rating', data=filtered_data)
    plt.xticks(rotation=90)
    st.pyplot(plt)

    # **Placement vs Fee Ratio**
    st.write("### Placement vs Fee Ratio")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='College Name', y='Placement vs Fee Ratio', data=filtered_data)
    plt.xticks(rotation=90)
    st.pyplot(plt)

    # **UG Fee Distribution**
    st.write("### UG Fee Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_data['UG fee (tuition fee)'], kde=True)
    st.pyplot(plt)

# **Footer**
st.markdown("---")
st.markdown("Developed with ❤️ by Arshia Singh using Streamlit")
