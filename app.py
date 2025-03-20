import streamlit as st
import pandas as pd
import numpy as np
import pickle  # For loading the trained model
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

# Set page config first
st.set_page_config(page_title="Your App", layout="wide", initial_sidebar_state="collapsed")

st.title("🎓 College Recommendation Project")
st.markdown("<div style='text-align: center", unsafe_allow_html=True)

# Load the trained model
with open('training_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load Data
data = pd.read_csv('cleaned_data.csv')

# App Mode Selection
mode = st.sidebar.radio("Select App Mode:", ["Light Mode", "GitHub Mode"])

# Apply color theme based on mode
if mode == "GitHub Mode":
    st.markdown(
        """
        <style>
            .stApp { background-color: black; color: white; }
            section[data-testid="stSidebar"] { background-color: #1c1c1c; }
            h1, h2, h3, h4, h5, h6, p, label { color: white; }
            input, textarea { background-color: #333; color: white; }
            select { background-color: #222; color: white; }
            button { background-color: #333; color: white; border: 1px solid white;h1, h2, h3, h4, h5, h6, p, label { color: black; } }
        </style>
        """,
        unsafe_allow_html=True
    )
    plt.style.use("dark_background")
    sns.set_palette("Greens")
else:
    st.markdown(
        """
        <style>
        .stApp { background-color: white; color: black; }
        </style>
        """,
        unsafe_allow_html=True
    )
    plt.style.use("default")

# Collapsible Sidebar for User Input
with st.sidebar:
    # Sidebar UI with Sliders instead of Buttons
    st.sidebar.header("Adjust Parameters")

    average_rating = st.sidebar.slider("Average Rating", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
    placement_vs_fee_ratio = st.sidebar.slider("Placement vs Fee Ratio", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    ug_fee_scaled = st.sidebar.slider("UG Fee (Scaled)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
    pg_fee_scaled = st.sidebar.slider("PG Fee (Scaled)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)


    # **Graph & Area Selection**
    selected_area = st.selectbox("Select Area", sorted(data['State'].str.strip().str.title().fillna('Unknown').unique().tolist()))

    # **Prediction Button**
    if st.button("Predict"):
        feature_names = ['Average Rating', 'Placement vs Fee Ratio', 'UG fee (scaled)', 'PG fee (scaled)']
        # Define input_data properly
        input_data = [[average_rating, placement_vs_fee_ratio, ug_fee_scaled, pg_fee_scaled]]  
        input_df = pd.DataFrame(input_data, columns=feature_names)  # No more NameError
        # Model Prediction
        prediction = model.predict(input_df)[0]  
        st.success(f"The predicted college category is: **{prediction}**")

# **Filtered Data**
filtered_data = data if selected_area == "All" else data[data['State'] == selected_area]

if filtered_data.empty:
    st.warning(f"No data available for {selected_area}. Try selecting a different area.")
else:
    st.subheader(f"📍 Colleges in {selected_area}")

# Display Selected Values
st.subheader("Explore Colleges by Area")
st.write(f"Selected Area: {selected_area}")

# **Filter Data by Selected Area**
filtered_data = data[data['State'] == selected_area]

# **Check if filtered_data is empty**
if filtered_data.empty:
    st.warning(f"No data available for {selected_area}. Try selecting a different area.")
else:
    # **Graphs for Selected Area**
    st.write(f"### Colleges in {selected_area}")

    # **Average Rating Comparison**
    st.write("### 📊 Average Rating Comparison")
    # Filter colleges based on slider input
    filtered_data = filtered_data[filtered_data['Average Rating'] >= average_rating]
    # Sort the updated dataset
    sorted_data = filtered_data.sort_values(by='Average Rating', ascending=False)
    plt.figure(figsize=(14, 6))  # Increase figure size
    sns.barplot(x='College Name', y='Average Rating', data=sorted_data)
    # Adjust x-axis labels dynamically
    num_colleges = len(sorted_data)
    if num_colleges <= 10:  
        plt.xticks(rotation=30, ha='right', fontsize=12)  # Less rotation for small lists
        plt.gca().set_xticks(range(num_colleges))  # Show all labels
        #plt.gca().set_xticklabels([college_name for college_name in sorted_data.index])  # Ensure all names appear
    else:
        plt.xticks(rotation=60, ha='right', fontsize=9)  # More rotation for large lists
        ticks = range(0, num_colleges, 5)  # Every 3rd label
        plt.gca().set_xticks(ticks)
        #plt.gca().set_xticklabels([sorted_data.index[i] for i in ticks])  # Assign correct labels
    plt.xlabel("College Name", fontsize=14, labelpad=15)
    plt.ylabel("Average Rating", fontsize=14, labelpad=15)
    plt.subplots_adjust(bottom=0.3)  # Adds space for labels
    st.pyplot(plt)


    # **Placement vs Fee Ratio**
    st.write("### 📈 Placement vs Fee Ratio")
    plt.figure(figsize=(14, 6))
    # Sort data
    sorted_data = filtered_data.sort_values(by='Placement vs Fee Ratio', ascending=False)
    # Plot
    sns.barplot(x=sorted_data['College Name'], y=sorted_data['Placement vs Fee Ratio'])
    # Adjust x-axis labels
    plt.xticks(rotation=60, ha='right', fontsize=9)
    # Limit displayed labels dynamically (every 5th label for large datasets)
    if len(sorted_data) > 15:
        plt.gca().set_xticklabels(sorted_data['College Name'][::5])
    # Improve spacing
    plt.xlabel("College Name", fontsize=14, labelpad=15)
    plt.ylabel("Placement vs Fee Ratio", fontsize=14, labelpad=15)
    plt.subplots_adjust(bottom=0.3)  # Adds space for labels
    st.pyplot(plt)


    # **UG Fee Distribution**
    st.write("### 💰 UG Fee Distribution")
    plt.figure(figsize=(12, 6))
    sns.histplot(filtered_data['UG fee (tuition fee)'], kde=True, bins=15)
    plt.xlabel("UG Fee (Tuition Fee)",fontsize=14,labelpad=15)
    plt.ylabel("Frequency",fontsize=14,labelpad=15)
    st.pyplot(plt)

    # **UG Fee (Scaled)**
    st.write("### 📏 UG Fee (Scaled)")
    plt.figure(figsize=(14, 6))
    sorted_data = filtered_data.sort_values(by='UG fee (scaled)', ascending=False)
    sns.barplot(x='College Name', y='UG fee (scaled)', data=sorted_data)

    plt.xticks(rotation=60, ha='right', fontsize=9)
    ticks = plt.gca().get_xticks()
    plt.gca().set_xticks(ticks[::5])
    plt.xlabel("College Name",fontsize=14,labelpad=15)
    plt.ylabel("UG Fee (Scaled)",fontsize=14,labelpad=15)
    plt.subplots_adjust(bottom=0.3)  # Adds space for labels
    st.pyplot(plt)

    # **PG Fee (Scaled)**
    st.write("### 🏛️ PG Fee (Scaled)")
    plt.figure(figsize=(14, 6))
    sorted_data = filtered_data.sort_values(by='PG fee (scaled)', ascending=False)
    sns.barplot(x='College Name', y='PG fee (scaled)', data=sorted_data)

    plt.xticks(rotation=60, ha='right', fontsize=9)
    ticks = plt.gca().get_xticks()
    plt.gca().set_xticks(ticks[::5])
    plt.xlabel("College Name",fontsize=14,labelpad=15)
    plt.ylabel("PG Fee (Scaled)",fontsize=14,labelpad=15)
    plt.subplots_adjust(bottom=0.3)  # Adds space for labels
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
        text-align:right;
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

st.markdown(footer, unsafe_allow_html=True)
