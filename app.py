import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(
    page_title="🩺 Diabetes Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------
# SESSION STATE FOR DASHBOARD
# ------------------------
if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

# ------------------------
# WELCOME SCREEN
# ------------------------
if not st.session_state.show_dashboard:
    st.markdown("""
    <div style='
        display:flex;
        justify-content:center;
        align-items:center;
        height:70vh;
        flex-direction:column;
        background: linear-gradient(-45deg, #c9d6ff, #a1c4fd, #e2e9fb, #c2f0fc);
        background-size: 400% 400%;
        animation: gradient 10s ease infinite;
        border-radius:15px;
        text-align:center;
    '>
        <h1 style='font-size:50px; animation:fadeIn 2s ease-in-out;'>🩺 Welcome to Diabetes Prediction Dashboard</h1>
        <p style='font-size:20px; animation:fadeIn 3s ease-in-out;'>Predict diabetes risk with interactive charts & ML</p>
    </div>

    <style>
    @keyframes gradient {
        0% {background-position:0% 50%;}
        50% {background-position:100% 50%;}
        100% {background-position:0% 50%;}
    }
    @keyframes fadeIn {
        0% {opacity:0;}
        100% {opacity:1;}
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div style='text-align:center; margin-top:20px;'>", unsafe_allow_html=True)
    if st.button("🚀 Get Started"):
        st.session_state.show_dashboard = True
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------
# MAIN DASHBOARD
# ------------------------
if st.session_state.show_dashboard:

    # ------------------------
    # CUSTOM CSS FOR DASHBOARD
    # ------------------------
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #c9d6ff, #e2e2e2, #a1c4fd, #c2e9fb);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    .fade-in {animation: fadeIn 1.5s ease-in;}
    @keyframes fadeIn {0% {opacity:0;} 100% {opacity:1;}}
    .stButton>button {
        background: linear-gradient(90deg, #0077b6, #00b4d8);
        color: white; border-radius: 12px; font-size: 18px; padding: 10px 30px;
        border: none; transition: 0.3s ease; box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
    }
    .stButton>button:hover {background: linear-gradient(90deg, #00b4d8, #0077b6);
        transform: scale(1.05); box-shadow: 0px 8px 20px rgba(0,0,0,0.4);}
    .metric-card {transition: transform 0.3s, box-shadow 0.3s; border-radius: 12px; padding: 15px; color:white; text-align:center; font-weight:bold;}
    .metric-card:hover {transform: scale(1.05); box-shadow: 0px 8px 25px rgba(0,0,0,0.4);}
    .metric-green {background-color: #2ecc71;}
    .metric-yellow {background-color: #f1c40f;}
    .metric-red {background-color: #e74c3c;}
    [data-testid="stPlotlyChart"], [data-testid="stAltairChart"] {transition: transform 0.3s, box-shadow 0.3s;}
    [data-testid="stPlotlyChart"]:hover, [data-testid="stAltairChart"]:hover {transform: scale(1.02); box-shadow: 0px 8px 25px rgba(0,0,0,0.3);}
    </style>
    """, unsafe_allow_html=True)

    # ------------------------
    # HEADER
    # ------------------------
    st.markdown("<h1 class='fade-in'>🩺 Diabetes Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align:center; color:gray;' class='fade-in'>Predict Diabetes using Logistic Regression</h5>", unsafe_allow_html=True)
    st.markdown("---")

    # ------------------------
    # LOAD DATA
    # ------------------------
    url = "https://github.com/YBIFoundation/Dataset/raw/main/Diabetes.csv"
    df = pd.read_csv(url)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ------------------------
    # MODEL TRAINING
    # ------------------------
    X = df[['pregnancies','glucose','diastolic','triceps','insulin','bmi','dpf','age']]
    y = df['diabetes']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_scaled,y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test,y_pred)

    # ------------------------
    # METRIC CARDS
    # ------------------------
    st.subheader("🧠 Model Performance")
    col1, col2, col3 = st.columns(3)
    def metric_class(name, val):
        if name=="Accuracy":
            if val>=0.9: return "metric-card metric-green"
            elif val>=0.8: return "metric-card metric-yellow"
            else: return "metric-card metric-red"
        else: return "metric-card metric-green"

    col1.markdown(f"<div class='{metric_class('Accuracy', accuracy)}'><h3>Accuracy</h3><p>{accuracy*100:.2f}%</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='{metric_class('Training Size', len(X_train))}'><h3>Training Size</h3><p>{len(X_train)}</p></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='{metric_class('Testing Size', len(X_test))}'><h3>Testing Size</h3><p>{len(X_test)}</p></div>", unsafe_allow_html=True)

    # ------------------------
    # PREDICTION FORM
    # ------------------------
    st.markdown("---")
    st.subheader("🔮 Predict Diabetes for New Patient")
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 200, 85)
        diastolic = st.number_input("Diastolic (BP)", 0, 122, 66)
    with col2:
        triceps = st.number_input("Triceps Thickness",0,99,29)
        insulin = st.number_input("Insulin",0,846,0)
        bmi = st.number_input("BMI",0.0,70.0,26.6)
    with col3:
        dpf = st.number_input("Diabetes Pedigree Function",0.0,2.5,0.351)
        age = st.number_input("Age",1,120,31)

    prob_placeholder = st.empty()
    dist_placeholder = st.empty()

    if st.button("🩸 Predict Diabetes"):
        input_data = np.array([[pregnancies,glucose,diastolic,triceps,insulin,bmi,dpf,age]])
        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]

        # Probability chart
        fig_prob, ax_prob = plt.subplots()
        ax_prob.bar(['Non-Diabetic','Diabetic'], [1-prob, prob], color=['lightgreen','salmon'])
        ax_prob.set_ylim(0,1)
        ax_prob.set_ylabel("Probability")
        ax_prob.set_title("Predicted Probability")
        prob_placeholder.pyplot(fig_prob)

        # Glucose comparison
        fig_dist, ax_dist = plt.subplots()
        sns.kdeplot(df[df['diabetes']==0]['glucose'], label='Non-Diabetic', shade=True, ax=ax_dist)
        sns.kdeplot(df[df['diabetes']==1]['glucose'], label='Diabetic', shade=True, ax=ax_dist)
        ax_dist.axvline(glucose, color='blue', linestyle='--', label='Patient Glucose')
        ax_dist.set_title("Glucose Comparison")
        ax_dist.set_xlabel("Glucose")
        ax_dist.set_ylabel("Density")
        ax_dist.legend()
        dist_placeholder.pyplot(fig_dist)

        # Result
        if pred==1:
            st.markdown("<p style='animation:fadeIn 1s; color:red; font-size:20px;'>⚠️ The person is likely to have <b>Diabetes</b>.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='animation:fadeIn 1s; color:green; font-size:20px;'>✅ The person is <b>not likely</b> to have Diabetes.</p>", unsafe_allow_html=True)
