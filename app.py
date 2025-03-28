import streamlit as st
import pandas as pd
import numpy as np
import pickle  # For loading the trained model
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import pathlib
import textwrap

# Set page config first
st.set_page_config(page_title="College Recommendation", layout="wide", initial_sidebar_state="collapsed")

st.title("🎓 College Recommendation Project")
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

# Load the trained model
with open('training_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load Data
data = pd.read_csv('cleaned_data.csv')
filtered_data = data.copy() 

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
    st.header("Adjust Parameters")

    average_rating = st.slider("Average Rating", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
    placement_vs_fee_ratio = st.slider("Placement vs Fee Ratio", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    min_ug_fee, max_ug_fee = data['UG fee (tuition fee)'].min(), data['UG fee (tuition fee)'].max()
    min_pg_fee, max_pg_fee = data['PG fee'].min(), data['PG fee'].max()
    ug_fee_range = st.slider("UG Fee Range", min_ug_fee, max_ug_fee, (min_ug_fee, max_ug_fee))
    pg_fee_range = st.slider("PG Fee Range", min_pg_fee, max_pg_fee, (min_pg_fee, max_pg_fee))


    # **Graph & Area Selection**
    selected_area = st.selectbox("Select Area", ['All']+sorted(data['State'].str.strip().str.title().fillna('Unknown').unique().tolist()))
    selected_stream = st.selectbox("Select Stream", ['All']+sorted(data['Stream'].str.strip().str.title().fillna('Unknown').unique().tolist()))
    # **Prediction Button**
    if st.button("Predict"):
        if all(val >= 0 for val in [average_rating, placement_vs_fee_ratio, ug_fee_range, pg_fee_range]):
            input_data = [[average_rating, placement_vs_fee_ratio, ug_fee_range, pg_fee_range]]
            prediction = model.predict(pd.DataFrame(input_data, columns=['Average Rating', 'Placement vs Fee Ratio', 'UG fee (tuition fee)', 'PG fee']))[0]
            st.success(f"The predicted college category is: **{prediction}**")
        else:
            st.warning("Please adjust the sliders to provide valid input values.")


if filtered_data.empty:
    st.warning(f"No data available for {selected_area}. Try selecting a different area from the side bar.")
else:
    st.subheader(f"📍 Colleges in {selected_area}")

# Display Selected Values
st.subheader("Explore Colleges by Area and Desired Stream")
st.write(f"Selected Area: {selected_area}")
st.write(f"Selected Stream: {selected_stream}")

# **Filter Data by Selected Area and Stream**
if selected_stream == "All":
    filtered_data = data[data['State'] == selected_area] 
else:
    filtered_data = data[(data['State'] == selected_area) & (data['Stream'] == selected_stream)] 

# **Check if filtered_data is empty**
if filtered_data.empty:
    st.warning(f"No data available for {selected_area}. Try selecting a different area.")
else:
    # **Graphs for Selected Area**
    st.write(f"### Colleges in {selected_area}")

    # Create columns for Average Rating Comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Average Rating - Bar Chart")
        filtered_rating_data = filtered_data[filtered_data["Average Rating"] >= average_rating].copy()
        filtered_rating_data = filtered_rating_data.dropna(subset=["Average Rating", "College Name"])
        st.write(f"🔍 Number of colleges after filtering: {len(filtered_rating_data)}")
        st.write("Filtered Data Preview:", filtered_data)
        if filtered_rating_data.empty:
            st.warning("⚠️ No colleges found with this Average Rating.")
        else:
            num_colleges = len(filtered_rating_data)
            fig_width = max(12, min(25, num_colleges * 0.4))
            fig_height = 6 if num_colleges <= 20 else 8
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.barplot(x="College Name", y="Average Rating", data=filtered_rating_data, ax=ax)
            ax.set_xticklabels(filtered_rating_data["College Name"], rotation=60, ha='right', fontsize=9)
            ax.set_xlabel("College Name")
            ax.set_ylabel("Average Rating")
            ax.set_title("Average Rating - Bar Chart")
            st.pyplot(fig)

        with col2:
            st.markdown("### 🥧 Top 10 Colleges - Pie Chart")
            # Group by College Name and take the highest Average Rating per college
            unique_colleges = (
                filtered_rating_data.groupby("College Name")["Average Rating"]
                .max()  # Take the highest rating for each college
                .reset_index()
            )
            # Sort and select top 10 unique colleges
            top_10_colleges = unique_colleges.sort_values(by="Average Rating", ascending=False).head(10)
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(
                top_10_colleges["Average Rating"],
                labels=top_10_colleges["College Name"],
                autopct="%1.1f%%",
                startangle=140,
                wedgeprops={"linewidth": 1, "edgecolor": "black"},
                textprops={"fontsize": 10},
            )
            plt.title("Top 10 Colleges - Pie Chart", fontsize=14)
            st.pyplot(fig)


    # Create columns for Placement vs Fee Ratio
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 📈 Placement vs Fee Ratio - Bar Chart")
        # Filter data based on the selected Placement vs Fee Ratio
        filtered_ratio_data = filtered_data[filtered_data['Placement vs Fee Ratio'] >= placement_vs_fee_ratio]
        if filtered_ratio_data.empty:
            st.warning("No colleges match the selected Placement vs Fee Ratio.")
        else:
            # Dynamically adjust figure size based on the number of colleges
            num_colleges = len(filtered_ratio_data)
            fig_width = max(12, min(25, num_colleges * 0.4))  # Dynamic width adjustment
            fig_height = 6 if num_colleges <= 20 else 8  # Adjust height if too many labels
            # Create figure
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            # Plot bar chart
            sns.barplot(x="College Name", y="Placement vs Fee Ratio", data=filtered_ratio_data, ax=ax, color="steelblue")
            # Rotate x-axis labels for readability
            if num_colleges > 10:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=9)
            # Set labels and title
            plt.xlabel("College Name")
            plt.ylabel("Placement vs Fee Ratio")
            plt.title("Placement vs Fee Ratio - Bar Chart")
            # Display in Streamlit
            st.pyplot(fig)



    with col4:
        st.markdown("### 🥧 Top 10 Colleges - Doughnut Chart")
        # Group by College Name and take the highest Average Rating per college
        unique_colleges = (
            filtered_ratio_data.groupby("College Name", as_index=False)  # Keep College Name
            .agg({"Placement vs Fee Ratio": "max", "Average Rating": "max"})  # Get max rating per college
        )
        # Sort and select top 10 unique colleges
        top10_colleges = unique_colleges.sort_values(by="Placement vs Fee Ratio", ascending=False).head(10)
        # Create doughnut chart
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            top10_colleges["Average Rating"],
            labels=top10_colleges["College Name"],
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops={"linewidth": 1, "edgecolor": "black"},
            textprops={"fontsize": 10},
        )
        # Add a white circle at the center to create a doughnut effect
        centre_circle = plt.Circle((0, 0), 0.70, fc="white")
        fig.gca().add_artist(centre_circle)
        plt.title("Top 10 Colleges - Doughnut Chart", fontsize=14)
        st.pyplot(fig)

    # Create columns for UG Fee
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("### 💰 UG Fee - Bar Chart")
        filtered_fee_data = filtered_data[
            (filtered_data["UG fee (tuition fee)"] >= ug_fee_range[0]) & 
            (filtered_data["UG fee (tuition fee)"] <= ug_fee_range[1])
        ]
        if filtered_fee_data.empty:
            st.warning("⚠ No colleges found with this UG Fee.")
        else:
            num_colleges = len(filtered_fee_data)
            fig_width = min(20, max(12, num_colleges * 0.5))  # Dynamic width
            fig_height = 8  # Fixed height
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            # Sort colleges by UG Fee for better visualization
            filtered_fee_data = filtered_fee_data.sort_values("UG fee (tuition fee)", ascending=False)
            # Plot bar chart with vertical bars
            sns.barplot(x="College Name", y="UG fee (tuition fee)", data=filtered_fee_data, ax=ax, color="steelblue")
            # Adjust x-labels dynamically for better readability
            ax.set_xticklabels(
                [textwrap.fill(label.get_text(), width=15) for label in ax.get_xticklabels()],
                rotation=30 if num_colleges > 10 else 0, 
                ha="right" if num_colleges > 10 else "center", 
                fontsize=9
            )
            ax.set_xlabel("College Name", fontsize=12)
            ax.set_ylabel("UG Fee", fontsize=12)
            ax.set_title("UG Fee - Bar Chart", fontsize=16)

            st.pyplot(fig)

    with col6:
        st.markdown("### 💰 UG Fee - Histogram")
        # Check if UG Fee column exists and is not empty
        if "UG fee (tuition fee)" not in filtered_data.columns or filtered_data["UG fee (tuition fee)"].isna().all():
            st.warning("⚠ No valid UG Fee data available.")
        else:
            # Use fig, ax to ensure Streamlit renders it properly
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(filtered_data["UG fee (tuition fee)"].dropna(), kde=True, bins=15, ax=ax)
            ax.set_xlabel("UG Fee", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title("UG Fee - Histogram", fontsize=16)
        st.pyplot(fig)  # Use fig here

    # Create columns for PG Fee
    col7, col8 = st.columns(2)

    with col7:
        st.write("### 🏛️ PG Fee - Bar Chart")
        filtered_pg_data = filtered_data[
            (filtered_data["PG fee"] >= pg_fee_range[0]) & 
            (filtered_data["PG fee"] <= pg_fee_range[1])
        ]
        filtered_pg_data = filtered_pg_data.nlargest(40, "PG fee")
        if filtered_ratio_data.empty:
            st.warning("⚠ No colleges found for this Placement vs Fee Ratio.")
        else:
            num_colleges = len(filtered_ratio_data)
            fig_width = min(20, max(12, num_colleges * 0.5))  # Dynamic width
            fig_height = 8  # Fixed height
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.barplot(x="College Name", y="Placement vs Fee Ratio", data=filtered_ratio_data, ax=ax, color="steelblue")
            # Dynamically adjust x-labels for better readability
            ax.set_xticklabels(
                [textwrap.fill(label.get_text(), width=15) for label in ax.get_xticklabels()],
                rotation=30 if num_colleges > 10 else 0, 
                ha="right" if num_colleges > 10 else "center", 
                fontsize=9
            )
            ax.set_xlabel("College Name", fontsize=12)
            ax.set_ylabel("Placement vs Fee Ratio", fontsize=12)
            ax.set_title("Placement vs Fee Ratio - Bar Chart", fontsize=16)
            st.pyplot(fig)

    st.markdown("### 📊 Fee Data Table")
    st.write(filtered_data[["College Name", "UG fee (tuition fee)", "PG fee", "Average Rating", "Placement vs Fee Ratio"]].head(20))
    col9, col10=st.columns(2)
    with col9:
        st.write("Columns in filtered_data:", filtered_data.columns)
    with col10:
        st.write(filtered_data["Average Rating"].describe())
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
