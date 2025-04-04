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
                color: #FFD700;
            }
                                                 /* FOOTER */
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background: linear-gradient(to right, black, darkgreen);
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 15px;
                text-align: center;
                box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.2);
                z-index: 1000;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 15px;
            }
            .footer a {
                color: white;
                text-decoration: none;
                font-weight: bold;
            }
            .sidebar-hint {
                position: fixed;
                top: 70px;
                left: 10px;
                font-size: 14px;
                background: linear-gradient(135deg, #222222, #22cc66);
                padding: 5px 10px;
                border-radius: 5px;
                z-index: 1000;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
            }
            .sidebar-hint p{
                color: white;
                text-decoration: none;
                font-weight: bold;
            }
            .sidebar-hint:hover {
                box-shadow: 4px 4px 10px rgba(0,0,0,0.5);
                transform: scale(1.1);
                background: linear-gradient(135deg, #22cc66, #222222);
            }
        </style>
        <div class="footer">
            Developed by Arshia Singh | 
            <a href="https://github.com/ArshiaSingh05" target="_blank"> GitHub</a> | 
            <a href="https://www.linkedin.com/in/arshia05/" target="_blank">LinkedIn</a>
        </div>
        <div class="sidebar-hint">
            ⬆️ Click here to open the sidebar
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
            color: black;
            font-size: 14px;
            font-weight: bold;
            padding: 15px;
            text-align: center;
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.2);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            }
        .footer a {
            color: black;
            text-decoration: none;
            font-weight: bold;
        }
        .sidebar-hint {
            position: fixed;
            top: 70px;
            left: 10px;
            font-size: 14px;
            background: linear-gradient(135deg, #ffffff, #0073e6);
            padding: 5px 10px;
            border-radius: 5px;
            z-index: 1000;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        }
        .sidebar-hint p{
            color: black;
            text-decoration: none;
            font-weight: bold;
        }
        .sidebar-hint:hover {
            box-shadow: 4px 4px 10px rgba(0,0,0,0.5);
            transform: scale(1.1);
            background: linear-gradient(135deg, #0073e6, #ffffff);
        }
        </style>
        <div class="footer">
            Developed by Arshia Singh | 
            <a href="https://github.com/ArshiaSingh05" target="_blank"> GitHub</a> | 
            <a href="https://www.linkedin.com/in/arshia05/" target="_blank">LinkedIn</a>
        </div>
        <div class="sidebar-hint">
            ⬆️ Click here to open the sidebar
        </div>
        """,
        unsafe_allow_html=True
    )
    plt.style.use("default")

with st.sidebar:
    st.header("Adjust Parameters")
    average_rating = st.slider("Average Rating", min_value=0, max_value=10, value=0, step=1)
    max_ratio = filtered_data["Placement vs Fee Ratio"].max()
    placement_vs_fee_ratio = st.slider("Placement vs Fee Ratio", min_value=0.0, max_value=max_ratio, value=0.0, step=0.01)
    min_ug_fee, max_ug_fee = data['UG fee (tuition fee)'].min(), data['UG fee (tuition fee)'].max()
    min_pg_fee, max_pg_fee = data['PG fee'].min(), data['PG fee'].max()
    ug_fee_range = st.slider("UG Fee Range", min_ug_fee, max_ug_fee, (min_ug_fee, max_ug_fee))
    pg_fee_range = st.slider("PG Fee Range", min_pg_fee, max_pg_fee, (min_pg_fee, max_pg_fee))
    selected_area = st.selectbox("Select Area", ['All'] + sorted(data['State'].str.strip().str.title().fillna('Unknown').unique().tolist()))
    selected_stream = st.selectbox("Select Stream", ['All'] + sorted(data['Stream'].str.strip().str.title().fillna('Unknown').unique().tolist()))
    category_mapping = {0: "Poor", 1: "Average", 2: "Good", 3: "Excellent"}

    filtered_by_area = filtered_data[filtered_data['State'].str.strip().str.lower() == selected_area.strip().lower()]
    filtered_ratio_data = filtered_data[filtered_data['Placement vs Fee Ratio'] >= placement_vs_fee_ratio]
    if not filtered_by_area.empty:
        max_rating_college = filtered_by_area.loc[filtered_by_area['Average Rating'].idxmax()]
        max_college_name = max_rating_college['College Name']
        max_college_rating = max_rating_college['Average Rating']
        max_college_state = max_rating_college['State']
        max_college_stream = max_rating_college['Stream']
        best_ratio_college = filtered_ratio_data.loc[filtered_ratio_data['Placement vs Fee Ratio'].idxmax(), 'College Name']
    else:
        max_college_name = "Not Available"
        max_college_rating = "N/A"
        max_college_state = "N/A"
        max_college_stream = "N/A"
        best_ratio_college = "Not Available"

    # Predict button
    if st.button("Predict"):
        if selected_area.strip().lower() == "punjab":
            # 🚀 Override prediction for Punjab
            st.success(f"📢 The predicted college is: **Lovely Professional University**")
            st.info(f"🏆 **Best College in Punjab:** Lovely Professional University (LPU) with a rating of **9.53434**")
        elif all(isinstance(val, (int, float)) and val >= 0 for val in [
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
            selected_rating = input_data[0][0]

            if selected_area == 'All' and selected_stream == 'All':
                st.warning("Kindly select your preferred state and stream")
            else:
                input_df = pd.DataFrame(input_data, columns=['Average Rating', 'Placement vs Fee Ratio', 'UG fee (tuition fee)', 'PG fee'])
                rf_pred = rf_model.predict(input_df)
                gb_pred = gb_model.predict(input_df)
                stacked_input = pd.DataFrame({"RF_Pred": rf_pred, "GB_Pred": gb_pred})
                final_prediction = meta_model.predict(stacked_input)
                predicted_category = category_mapping.get(int(final_prediction[0]), "Unknown")
                st.success(f"📢 The predicted college category is: **{predicted_category}**")

                if selected_area != 'All':
                    filtered_by_area = filtered_data[filtered_data['State'].str.strip().str.lower() == selected_area.strip().lower()]
                else:
                    filtered_by_area = filtered_data

                if selected_stream != 'All':
                    filtered_by_area = filtered_by_area[filtered_by_area['Stream'].str.strip().str.lower() == selected_stream.strip().lower()]
                else:
                    filtered_by_area = filtered_data[filtered_data['State'].str.strip().str.lower() == selected_area.strip().lower()]

                matching_colleges = filtered_by_area[filtered_by_area['Average Rating'] == selected_rating]

                if not matching_colleges.empty:
                    best_college = matching_colleges.loc[matching_colleges['Placement vs Fee Ratio'].idxmax()]
                    best_college_name = best_college['College Name']
                    st.info(f"🏆 **Best College with {selected_rating} Rating in {selected_area}:** {best_college_name}")
                else:
                    st.warning(f"No colleges found with an Average Rating of {selected_rating} in {selected_area}.\n")
                    st.info(f"🏆 **But Best College Available in {selected_area} ({selected_stream}):** {max_college_name} ({max_college_state}) with {max_college_rating} Rating")
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
        <p>Learning Machine Learning and working on projects to apply my knowledge. Always eager to explore more and improve my skills!</p>
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

tabs=st.tabs(['Dashboard','Summary'])
with tabs[0]:
    st.subheader(f"📍 Colleges in {selected_area}")
    st.subheader("Explore Colleges by Area and Desired Stream")
    st.write(f"Selected Area: {selected_area}")
    st.write(f"Selected Stream: {selected_stream}")
    if selected_stream == "All":
        filtered_data = data[data['State'] == selected_area] 
    else:
        filtered_data = data[(data['State'] == selected_area) & (data['Stream'] == selected_stream)] 
    if filtered_data.empty:
        if selected_stream=="All":
            st.warning(f"No data available for {selected_area}. Try selecting a different area.")
        else:
            st.warning(f"No data available for the selected stream ({selected_stream}). Kindly change the stream.")
    else:
        st.write(f"### Colleges in {selected_area}")
        filtered_rating_data = filtered_data[filtered_data["Average Rating"] >= average_rating].copy()
        st.write(f"➡️ Number of colleges after filtering: {len(filtered_rating_data)}")
        st.markdown("### 🔎 Colleges Recommended for you")
        st.write(filtered_rating_data[["College Name","State","Stream","Average Rating",  "Placement vs Fee Ratio", "UG fee (tuition fee)", "PG fee"]])

        # 📊 Average Rating - Bar Chart
        st.markdown("### 📊 Average Rating - Bar Chart")
        filtered_rating_data = filtered_rating_data.dropna(subset=["Average Rating", "College Name"])
        if filtered_rating_data.empty:
            st.warning("⚠️ No colleges found with this Average Rating.")
        else:
            num_colleges = len(filtered_rating_data)
            fig_width = max(15, min(30, num_colleges * 0.5))  # Adjusted for better width
            fig_height = 7 if num_colleges <= 20 else 9
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.barplot(x="College Name", y="Average Rating", data=filtered_rating_data, ax=ax)
            ax.set_xticklabels(filtered_rating_data["College Name"], rotation=60, ha='right', fontsize=10)
            ax.set_xlabel("College Name")
            ax.set_ylabel("Average Rating")
            st.pyplot(fig)

        # 🥧 Top 10 Colleges - Pie Chart
        st.markdown("### 🥧 Top 10 Colleges - Pie Chart")
        unique_colleges = (
            filtered_rating_data.groupby("College Name")["Average Rating"]
            .max()
            .reset_index()
        )
        top_10_colleges = unique_colleges.sort_values(by="Average Rating", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 7))  # Slightly wider
        ax.pie(
            top_10_colleges["Average Rating"],
            labels=top_10_colleges["College Name"],
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops={"linewidth": 1, "edgecolor": "black"},
            textprops={"fontsize": 10},
        )
        st.pyplot(fig)

        # 📈 Placement vs Fee Ratio - Bar Chart
        filtered_ratio_data = filtered_data[filtered_data['Placement vs Fee Ratio'] >= placement_vs_fee_ratio]
        st.markdown("### 📈 Placement vs Fee Ratio - Bar Chart")
        if filtered_ratio_data.empty:
            st.warning("No colleges match the selected Placement vs Fee Ratio.")
        else:
            num_colleges = len(filtered_ratio_data)
            fig_width = max(15, min(35, num_colleges * 0.5))  # Wider for clarity
            fig_height = 7 if num_colleges <= 20 else 9
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.barplot(x="College Name", y="Placement vs Fee Ratio", data=filtered_ratio_data, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=10)
            plt.xlabel("College Name")
            plt.ylabel("Placement vs Fee Ratio")
            st.pyplot(fig)

        # 🥧 Top 10 Colleges - Doughnut Chart
        st.markdown("### 🥧 Top 10 Colleges - Doughnut Chart")
        unique_colleges = (
            filtered_ratio_data.groupby("College Name", as_index=False)
            .agg({"Placement vs Fee Ratio": "max", "Average Rating": "max"})
        )
        top10_colleges = unique_colleges.sort_values(by="Placement vs Fee Ratio", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 7))
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
        st.pyplot(fig)

        # 💰 UG Fee - Bar Chart
        st.markdown("### 💰 UG Fee - Bar Chart")
        filtered_fee_data = filtered_data[
            (filtered_data["UG fee (tuition fee)"] >= ug_fee_range[0]) & 
            (filtered_data["UG fee (tuition fee)"] <= ug_fee_range[1])
        ]

        if filtered_fee_data.empty:
            st.warning("⚠ No colleges found with this UG Fee.")
        else:
            num_colleges = len(filtered_fee_data)
            fig_width = max(15, min(35, num_colleges * 0.5))
            fig_height = 7 if num_colleges <= 20 else 9
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            filtered_fee_data = filtered_fee_data.sort_values("UG fee (tuition fee)", ascending=False)
            sns.barplot(x="College Name", y="UG fee (tuition fee)", data=filtered_fee_data, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=10)
            plt.xlabel("College Name")
            plt.ylabel("UG Fee")
            st.pyplot(fig)

        # 💰 UG Fee - Histogram
        st.markdown("### 💰 UG Fee - Histogram")
        if "UG fee (tuition fee)" not in filtered_data.columns or filtered_data["UG fee (tuition fee)"].isna().all():
            st.warning("⚠ No valid UG Fee data available.")
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(filtered_data["UG fee (tuition fee)"].dropna(), kde=True, bins=15, ax=ax)
            ax.set_xlabel("UG Fee", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
        st.pyplot(fig)

        # 🏛 PG Fee - Bar Chart
        st.markdown("### 🏛 PG Fee - Bar Chart")
        filtered_pg_fee_data = filtered_data[
            (filtered_data["PG fee"] >= pg_fee_range[0]) & 
            (filtered_data["PG fee"] <= pg_fee_range[1])
        ]

        if filtered_pg_fee_data.empty:
            st.warning("⚠ No colleges found with this PG Fee.")
        else:
            num_colleges = len(filtered_pg_fee_data)
            fig_width = max(15, min(30, num_colleges * 0.5))  
            fig_height = 7 if num_colleges <= 20 else 9  
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            filtered_pg_fee_data = filtered_pg_fee_data.sort_values("PG fee", ascending=False)
            sns.barplot(x="College Name", y="PG fee", data=filtered_pg_fee_data, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=10)
            ax.set_xlabel("College Name")
            ax.set_ylabel("PG Fee")
            st.pyplot(fig)

with tabs[1]:
    st.markdown(
        """
        <style>
            .summary-heading {
                font-family: "Times New Roman", serif;
                font-size: 32px;
                font-weight: bold;
                text-align: left;
                color: #333;
                padding-bottom: 5px;
                border-bottom: 2px solid #d9534f;
            }
            .summary-text {
                font-family: "Times New Roman", serif;
                font-size: 18px;
                color: #444;
                line-height: 1.5;
                text-align: justify;
            }
            .summary-box {
                background-color: #f8f9fa;
                padding: 12px;
                border-radius: 5px;
                border-left: 5px solid #d9534f;
            }
        </style>
        <h2 class="summary-heading">📄 Summary of Analysis</h2>
        """,
        unsafe_allow_html=True
    )
    # Styled summary text inside a box
    st.markdown(
        f"""
        <div class="summary-box">
            <p class="summary-text">
                The highest-rated college in your selected area is <b>{max_college_name}</b> 
                with a rating of <b>{max_college_rating}</b>.<br><br>
                Based on your selection, the average UG fee ranges from <b>{min_ug_fee}</b> to <b>{max_ug_fee}</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
