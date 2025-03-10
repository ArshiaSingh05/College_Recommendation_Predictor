import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

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

# Sidebar for user inputs
with st.sidebar:
    st.title("Input Values")
    average_rating = st.slider("Average Rating", 0.0, 10.0, 8.0, 0.1)
    placement_fee_ratio = st.slider("Placement vs Fee Ratio", 0.0, 0.00002, 0.00001, 0.000001)
    ug_fee_scaled = st.slider("UG Fee (Scaled)", 0.0, 1.0, 0.5, 0.01)
    pg_fee_scaled = st.slider("PG Fee (Scaled)", 0.0, 1.0, 0.5, 0.01)

# Prediction button
if st.button("Predict"):
    input_data = pd.DataFrame([[average_rating, placement_fee_ratio, ug_fee_scaled, pg_fee_scaled]],
                              columns=['Average Rating', 'Placement vs Fee Ratio', 'UG fee (scaled)', 'PG fee (scaled)'])
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"The predicted college category is: **{prediction}**")


# Graph Resizing Logic
sidebar_closed = st.sidebar.empty()

if sidebar_closed:
    graph_size = (12, 8)
else:
    graph_size = (10, 6)

# Average Rating Comparison
st.write("### Average Rating Comparison")
plt.figure(figsize=graph_size)
sns.barplot(x='College Name', y='Average Rating', data=data)
plt.xticks(rotation=90)
st.pyplot(plt)

# Placement vs Fee Ratio
st.write("### Placement vs Fee Ratio")
plt.figure(figsize=graph_size)
sns.barplot(x='College Name', y='Placement vs Fee Ratio', data=data)
plt.xticks(rotation=90)
st.pyplot(plt)

# UG Fee Distribution
st.write("### UG Fee Distribution")
plt.figure(figsize=graph_size)
sns.histplot(data['UG fee (tuition fee)'], kde=True)
st.pyplot(plt)

# Add a footer
st.markdown("---")
st.markdown("Developed with ❤️ by Arshia Singh using Streamlit")
