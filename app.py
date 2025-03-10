import streamlit as st
import pandas as pd
import numpy as np
import pickle  # For loading the trained model
import seaborn as sb
import matplotlib.pyplot as plt

# Load the trained model
with open('training_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load Data
data = pd.read_csv('cleaned_data.csv')

# Filter Data
filtered_data = data[data['State'] == area]

# Title and description
st.title("College Recommendation Predictor")
st.write("Enter the following details to predict the recommended college category.")

# Input fields
average_rating = st.number_input("Average Rating", min_value=0.0, max_value=10.0, step=0.1)
placement_fee_ratio = st.number_input("Placement vs Fee Ratio", min_value=0.0, step=0.00001)
ug_fee_scaled = st.number_input("UG Fee (Scaled)", min_value=0.0, max_value=1.0, step=0.001)
pg_fee_scaled = st.number_input("PG Fee (Scaled)", min_value=0.0, max_value=1.0, step=0.001)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[average_rating, placement_fee_ratio, ug_fee_scaled, pg_fee_scaled]])
    prediction = model.predict(input_data)[0]
    
    st.success(f"The predicted college category is: **{prediction}**")
    
# Average Rating Comparison
st.write("### Average Rating Comparison")
plt.figure(figsize=(10, 6))
sns.barplot(x='College Name', y='Average Rating', data=filtered_data)
plt.xticks(rotation=90)
st.pyplot(plt)

# Placement vs Fee Ratio
st.write("### Placement vs Fee Ratio")
plt.figure(figsize=(10, 6))
sns.barplot(x='College Name', y='Placement vs Fee Ratio', data=filtered_data)
plt.xticks(rotation=90)
st.pyplot(plt)

# UG Fee Distribution
st.write("### UG Fee Distribution")
plt.figure(figsize=(10, 6))
sns.histplot(filtered_data['UG fee (tuition fee)'], kde=True)
st.pyplot(plt)

# Add a footer
st.markdown("---")
st.markdown("Developed with ❤️ by Arshia Singh using Streamlit")

