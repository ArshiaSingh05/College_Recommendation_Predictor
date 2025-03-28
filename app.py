import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import pathlib
import textwrap

st.set_page_config(page_title="College Recommendation", layout="wide", initial_sidebar_state="collapsed")

st.title("🎓 College Recommendation Project")
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

# Loading trained model
with open('random_forest.pkl', 'rb') as file:
    rf_model = joblib.load(file)
with open('gradient_boosting.pkl', 'rb') as file:
    gb_model = joblib.load(file)
with open('meta_model.pkl', 'rb') as file:
    meta_model = joblib.load(file)

# Loading Data
data = pd.read_csv('cleaned_data.csv')
filtered_data = data.copy() 

mode = st.sidebar.radio("Select App Mode:", ["Light Mode", "GitHub Mode"])
if mode == "GitHub Mode":
    st.markdown(
        """
        <style>
            .stApp { background-color: black; color: white; }
            section[data-testid="stSidebar"] { background-color: #1c1c1c; }
            h1, h2, h3, h4, h5, h6, p, label { color: white; }
            input, textarea { background-color: #333; color: white; }
            select { background-color: #222; color: white; }
            button {
                background: linear-gradient(135deg, black, #006400);
                color: white;
                border: 1px solid #008000;
                padding: 8px 15px;
                font-weight: bold;
                border-radius: 8px;
                cursor: pointer;
                transition: background 0.3s ease-in-out;
            }
            button:hover {
                background: linear-gradient(135deg, #003300, #008000);
            }
                                                   /* Dark mode custom boxes */
            .custom-box {
                background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
                color: white;
                padding: 15px;
                border-radius: 12px;
                box-shadow: 4px 4px 8px rgba(0, 0, 0, 0.15), inset 2px 2px 5px rgba(255, 255, 255, 0.2);
                margin-bottom: 15px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                backdrop-filter: blur(10px);
            }
            .about-box {
                background: linear-gradient(135deg, #0f5726, #1e7e34);
                color: white;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
            }
            .handles-box {
                background: linear-gradient(135deg, #0f5726, #1e7e34);
                color: white;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
            }
            .projects-box {
                background: linear-gradient(135deg, #0f5726, #1e7e34);
                color: white;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
            }
                                                 /* FOOTER */
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background: linear-gradient(to right, black, darkgreen);
                color: white;
                font-size: 14px;
                padding: 15px;
                text-align: right;
                box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.2);
                z-index: 1000;
            }
            .footer a {
                color: white;
                text-decoration: none;
                font-weight: bold;
            }
        </style>
        <div class="footer">
            Developed with ❤️ by Arshia Singh using Streamlit
        </div>
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
                                            /* ABOUT ME CUSTOMIZATION */
        .custom-box {
            padding: 15px;
            border-radius: 12px;
            box-shadow: 4px 4px 8px rgba(0, 0, 0, 0.15), inset 2px 2px 5px rgba(255, 255, 255, 0.2);
            margin-bottom: 15px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(10px);
        }
        .about-box {
            background: linear-gradient(135deg, rgba(109, 213, 237, 0.7), rgba(33, 147, 176, 0.7));
            color: black;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        }
        .handles-box {
            background: linear-gradient(135deg, rgba(109, 213, 237, 0.7), rgba(33, 147, 176, 0.7));
            color: black;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        }
        .projects-box {
            background: linear-gradient(135deg, rgba(109, 213, 237, 0.7), rgba(33, 147, 176, 0.7));
            color: black;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        }
        h3 {
            margin-bottom: 10px;
            font-size: 18px;
        }
        a {
            text-decoration: none;
            font-weight: bold;
            color: white;
        }
        a:hover {
            text-decoration: underline;
            color: #ffd700;
        }
                                           /* FOOTER */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: linear-gradient(to right, white, #7EC8E3);
            color: #6c757d;
            font-size: 14px;
            padding: 15px;
               text-align: right;
                box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.2);
                z-index: 1000;
        }
        .footer a {
            color: black;
            text-decoration: none;
            font-weight: bold;
        }
        </style>
        <div class="footer">
            Developed with ❤️ by Arshia Singh using Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )
    plt.style.use("default")

with st.sidebar:
    st.header("Adjust Parameters")
    average_rating = st.slider("Average Rating", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
    placement_vs_fee_ratio = st.slider("Placement vs Fee Ratio", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    min_ug_fee, max_ug_fee = data['UG fee (tuition fee)'].min(), data['UG fee (tuition fee)'].max()
    min_pg_fee, max_pg_fee = data['PG fee'].min(), data['PG fee'].max()
    ug_fee_range = st.slider("UG Fee Range", min_ug_fee, max_ug_fee, (min_ug_fee, max_ug_fee))
    pg_fee_range = st.slider("PG Fee Range", min_pg_fee, max_pg_fee, (min_pg_fee, max_pg_fee))
    selected_area = st.selectbox("Select Area", ['All'] + sorted(data['State'].str.strip().str.title().fillna('Unknown').unique().tolist()))
    selected_stream = st.selectbox("Select Stream", ['All'] + sorted(data['Stream'].str.strip().str.title().fillna('Unknown').unique().tolist()))
    category_mapping = {0: "Poor", 1: "Average", 2: "Good", 3: "Excellent"}
    # Predict button
    if st.button("Predict"):
        if all(isinstance(val, (int, float)) and val >= 0 for val in [
            average_rating, placement_vs_fee_ratio, 
            ug_fee_range[0], ug_fee_range[1], 
            pg_fee_range[0], pg_fee_range[1]
        ]):
            input_data = [[
                average_rating, 
                placement_vs_fee_ratio, 
                (ug_fee_range[0] + ug_fee_range[1]) / 2,  
                (pg_fee_range[0] + pg_fee_range[1]) / 2
            ]]
            input_df = pd.DataFrame(input_data, columns=['Average Rating', 'Placement vs Fee Ratio', 'UG fee (tuition fee)', 'PG fee'])
            rf_pred = rf_model.predict(input_df)
            gb_pred = gb_model.predict(input_df)
            stacked_input = pd.DataFrame({"RF_Pred": rf_pred, "GB_Pred": gb_pred})
            final_prediction = meta_model.predict(stacked_input)
            predicted_category = category_mapping.get(int(final_prediction[0]), "Unknown")
            st.success(f"📢 The predicted college category is: **{predicted_category}**")
            # Get the exact selected average rating
            selected_rating = input_data[0][0]
            if selected_area != 'All':
                filtered_by_area = filtered_data[filtered_data['State'].str.strip().str.lower() == selected_area.strip().lower()]
            else:
                filtered_by_area = filtered_data
            if selected_stream != 'All':
                filtered_by_area = filtered_by_area[filtered_by_area['Stream'].str.strip().str.lower() == selected_stream.strip().lower()]
            else:
                filtered_by_area=filtered_data
            matching_colleges = filtered_by_area[filtered_by_area['Average Rating'] == selected_rating]
            if not matching_colleges.empty:
                # Find the best college in selected area based on Placement vs Fee Ratio
                best_college = matching_colleges.loc[matching_colleges['Placement vs Fee Ratio'].idxmax()]
                best_college_name = best_college['College Name']
                st.info(f"🏆 **Best College with {selected_rating} Rating in {selected_area}:** {best_college_name}")
            else:
                st.warning(f"No colleges found with an Average Rating of {selected_rating} in {selected_area}.")
        else:
            st.warning("⚠ Please adjust the sliders to provide valid input values.")

st.sidebar.title("👤 My Profile")
# About Me
st.sidebar.markdown(
    """
    <div class="custom-box about-box">
        <h3>ABOUT ME</h3>
        <p><strong>Name:</strong> Arshia Singh</p>
        <p><strong>University:</strong> Lovely Professional University</p>
        <p><strong>Field:</strong> Machine Learning & Data Science</p>
        <p>Passionate about solving real-world problems with data and AI! 🚀</p>
    </div>
    """, unsafe_allow_html=True
)
# My Handles
st.sidebar.markdown(
    """
    <div class="custom-box handles-box">
        <h3>MY HANDLES</h3>
        <p>🔗 <a href="https://github.com/ArshiaSingh05" target="_blank">GitHub</a></p>
        <p>💼 <a href="https://www.linkedin.com/in/arshia05/" target="_blank">LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True
)
# My Projects
st.sidebar.markdown(
    """
    <div class="custom-box projects-box">
        <h3>MY PROJECTS</h3>
        <p>📌 <a href="https://github.com/ArshiaSingh05/College_Recommendation_Predictor" target="_blank">College Prediction</a></p>
    </div>
    """, unsafe_allow_html=True
)

if filtered_data.empty:
    st.warning(f"No data available for {selected_area}. Try selecting a different area from the side bar.")
else:
    st.subheader(f"📍 Colleges in {selected_area}")

st.subheader("Explore Colleges by Area and Desired Stream")
st.write(f"Selected Area: {selected_area}")
st.write(f"Selected Stream: {selected_stream}")

if selected_stream == "All":
    filtered_data = data[data['State'] == selected_area] 
else:
    filtered_data = data[(data['State'] == selected_area) & (data['Stream'] == selected_stream)] 
if filtered_data.empty:
    st.warning(f"No data available for {selected_area}. Try selecting a different area.")
else:
    st.write(f"### Colleges in {selected_area}")
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
        unique_colleges = (
            filtered_rating_data.groupby("College Name")["Average Rating"]
            .max()
            .reset_index()
        )
        top_10_colleges = unique_colleges.sort_values(by="Average Rating", ascending=False).head(10)
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

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 📈 Placement vs Fee Ratio - Bar Chart")
        filtered_ratio_data = filtered_data[filtered_data['Placement vs Fee Ratio'] >= placement_vs_fee_ratio]
        if filtered_ratio_data.empty:
            st.warning("No colleges match the selected Placement vs Fee Ratio.")
        else:
            num_colleges = len(filtered_ratio_data)
            fig_width = max(12, min(30, num_colleges * 0.5))
            fig_height = 6 if num_colleges <= 20 else 8 
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.barplot(x="College Name", y="Placement vs Fee Ratio", data=filtered_ratio_data, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=9)
            plt.xlabel("College Name")
            plt.ylabel("Placement vs Fee Ratio")
            plt.title("Placement vs Fee Ratio - Bar Chart")
            st.pyplot(fig)

    with col4:
        st.markdown("### 🥧 Top 10 Colleges - Doughnut Chart")
        unique_colleges = (
            filtered_ratio_data.groupby("College Name", as_index=False)
            .agg({"Placement vs Fee Ratio": "max", "Average Rating": "max"})
        )
        top10_colleges = unique_colleges.sort_values(by="Placement vs Fee Ratio", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            top10_colleges["Average Rating"],
            labels=top10_colleges["College Name"],
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops={"linewidth": 1, "edgecolor": "black"},
            textprops={"fontsize": 10},
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc="white")
        fig.gca().add_artist(centre_circle)
        plt.title("Top 10 Colleges - Doughnut Chart", fontsize=14)
        st.pyplot(fig)

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
            fig_width = max(12, min(30, num_colleges * 0.5))
            fig_height = 6 if num_colleges <= 20 else 8  
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            filtered_fee_data = filtered_fee_data.sort_values("UG fee (tuition fee)", ascending=False)
            sns.barplot(x="College Name", y="UG fee (tuition fee)", data=filtered_fee_data, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=9)
            plt.xlabel("College Name")
            plt.ylabel("UG Fee")
            plt.title("UG Fee - Bar Chart")
            st.pyplot(fig)

    with col6:
        st.markdown("### 💰 UG Fee - Histogram")
        if "UG fee (tuition fee)" not in filtered_data.columns or filtered_data["UG fee (tuition fee)"].isna().all():
            st.warning("⚠ No valid UG Fee data available.")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(filtered_data["UG fee (tuition fee)"].dropna(), kde=True, bins=15, ax=ax)
            ax.set_xlabel("UG Fee", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title("UG Fee - Histogram", fontsize=16)
        st.pyplot(fig) 

    col7, col8 = st.columns(2)
    with col7:
        st.markdown("### 🏛 PG Fee - Bar Chart")
        filtered_pg_fee_data = filtered_data[
            (filtered_data["PG fee"] >= pg_fee_range[0]) & 
            (filtered_data["PG fee"] <= pg_fee_range[1])
        ]
        if filtered_pg_fee_data.empty:
            st.warning("⚠ No colleges found with this PG Fee.")
        else:
            num_colleges = len(filtered_pg_fee_data)
            fig_width = max(12, min(25, num_colleges * 0.4))  
            fig_height = 6 if num_colleges <= 20 else 8  
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            filtered_pg_fee_data = filtered_pg_fee_data.sort_values("PG fee", ascending=False)
            sns.barplot(x="College Name", y="PG fee", data=filtered_pg_fee_data, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=9)
            ax.set_xlabel("College Name")
            ax.set_ylabel("PG Fee")
            ax.set_title("PG Fee - Bar Chart")
            st.pyplot(fig)


    st.markdown("### 📊 Fee Data Table")
    st.write(filtered_data[["College Name", "UG fee (tuition fee)", "PG fee", "Average Rating", "Placement vs Fee Ratio"]].head(20))
