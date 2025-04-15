# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np 

# Cargar modelo, scaler y columnas esperadas
model = joblib.load("model/log_reg_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# Columnas numéricas que deben ser escaladas
num_cols = ['Age', 'Credit_Amount', 'Duration_Months']

# Función para preprocesar el input
def preprocess_input(user_input):
    df = pd.DataFrame([user_input])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    df[num_cols] = scaler.transform(df[num_cols])
    return df

# Función para determinar la categoría del score
def get_credit_category(score):
    if score < 560:
        return 'Poor'
    elif score < 660:
        return 'Fair'
    elif score < 725:
        return 'Good'
    elif score < 760:
        return 'Very Good'
    else:
        return 'Excellent'

# Función para calcular el credit score
def calculate_credit_score(prob):
    return int(300 + prob * (900 - 300))


import plotly.graph_objects as go

def plot_credit_score_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "<b>Simulated Canadian Credit Score</b>",
            'font': {'size': 20, 'color': 'black'}
        },
        number={'font': {'color': 'black'}},  # Score en negro
        gauge={
            'axis': {'range': [300, 900], 'tickwidth': 1, 'tickcolor': "black", 'tickfont': {'color': 'black'}},
            'bar': {'color': "rgba(0,0,0,0)"},  # Ocultar barra
            'steps': [
                {'range': [300, 560], 'color': '#d73027', 'name': 'Poor'},
                {'range': [560, 660], 'color': '#fc8d59', 'name': 'Fair'},
                {'range': [660, 725], 'color': '#fee08b', 'name': 'Good'},
                {'range': [725, 760], 'color': '#d9ef8b', 'name': 'Very Good'},
                {'range': [760, 900], 'color': '#1a9850', 'name': 'Excellent'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.8,
                'value': score
            }
        }
    ))

    # Agregar etiquetas manuales como anotaciones
    annotations = [
        {'x': 0.18, 'y': 0.4, 'text': 'Poor'},
        {'x': 0.5, 'y': 0.9, 'text': 'Fair'},
        {'x': 0.65, 'y': 0.8, 'text': 'Good'},
        {'x': 0.78, 'y': 0.65, 'text': 'Very Good'},
        {'x': 0.83, 'y': 0.4, 'text': 'Excellent'}
    ]

    for ann in annotations:
        fig.add_annotation(
            x=ann['x'], y=ann['y'],
            text=f"<b>{ann['text']}</b>",
            showarrow=False,
            font=dict(size=12, color="black")
        )

    st.plotly_chart(fig, use_container_width=True)

# Título principal
st.title("Canadian Credit Risk Simulator 🇨🇦")
st.subheader("Let's estimate your Canadian credit score 📊")

# Placeholder para mostrar el resultado ANTES del formulario
result_container = st.container()

# Inicio del formulario
with st.form("credit_form"):
    st.markdown("### 💳 Credit Information")

    col1, col2, col3 = st.columns(3)
    with col1:
        Duration_Months = st.slider('Credit Duration (Months)', 4, 72, 24)
    with col2:
        Credit_Amount = st.number_input('Credit Amount', min_value=250, max_value=20000, value=5000)
    with col3:
        Age = st.slider('Age', 18, 75, 35)

    st.markdown("### 🧾 Financial & Personal Info")

    col4, col5 = st.columns(2)
    with col4:
        Status_Checking_Account = st.selectbox('Checking Account Status', ['< 0 DM', '0 <= ... < 200 DM', '>= 200 DM / salary assignment', 'no checking account'])
        Credit_History = st.selectbox('Credit History', ['no credits taken/ all paid', 'existing credits paid back duly till now', 'delay in paying off in the past', 'critical account/ other credits existing'])
        Purpose = st.selectbox('Purpose', ['radio/TV', 'car (new)', 'car (used)', 'furniture/equipment', 'domestic appliances', 'education', 'repairs', 'retraining', 'others'])
        Savings_Account_Bonds = st.selectbox('Savings Account', ['< 100 DM', '100 <= ... < 500 DM', '500 <= ... < 1000 DM', '>= 1000 DM', 'unknown/ no savings account'])
        Employment_Since = st.selectbox('Employment Duration', ['unemployed', '< 1 year', '1 <= ... < 4 years', '4 <= ... < 7 years', '>= 7 years'])
        Personal_Status_Sex = st.selectbox('Personal Status & Sex', ['male : single', 'male : married/widowed', 'male : divorced/separated', 'female : divorced/separated/married'])

    with col5:
        Other_Debtors_Guarantors = st.selectbox('Other Debtors/Guarantors', ['none', 'guarantor', 'co-applicant'])
        Property = st.selectbox('Property', ['real estate', 'car or other', 'building society savings / life insurance', 'unknown / no property'])
        Other_Installment_Plans = st.selectbox('Other Installment Plans', ['none', 'bank', 'stores'])
        Housing = st.selectbox('Housing', ['own', 'rent', 'for free'])
        Job = st.selectbox('Job Type', ['unemployed/unskilled - non-resident', 'unskilled - resident', 'skilled employee/official', 'highly qualified/self-employed'])
        Telephone = st.selectbox('Telephone', ["yes, registered under the customer's name", 'none'])
        Foreign_Worker = st.selectbox('Are you a foreign worker?', ['yes', 'no'])

    st.markdown("### 🧮 Additional Details")
    col6, col7, col8 = st.columns(3)
    with col6:
        Installment_Rate = st.selectbox('Installment Rate', [1, 2, 3, 4])
    with col7:
        Present_Residence_Since = st.selectbox('Years in Current Residence', [1, 2, 3, 4])
    with col8:
        Number_Credits = st.selectbox('Number of Existing Credits', [1, 2, 3, 4])

    People_Liable = st.selectbox('Number of People Liable', [1, 2])

    submitted = st.form_submit_button("Simulate Credit Score")

# Si el usuario envió el formulario
if submitted:
    user_input = {
        'Duration_Months': Duration_Months,
        'Credit_Amount': Credit_Amount,
        'Age': Age,
        'Status_Checking_Account': Status_Checking_Account,
        'Credit_History': Credit_History,
        'Purpose': Purpose,
        'Savings_Account_Bonds': Savings_Account_Bonds,
        'Employment_Since': Employment_Since,
        'Personal_Status_Sex': Personal_Status_Sex,
        'Other_Debtors_Guarantors': Other_Debtors_Guarantors,
        'Property': Property,
        'Other_Installment_Plans': Other_Installment_Plans,
        'Housing': Housing,
        'Job': Job,
        'Telephone': Telephone,
        'Foreign_Worker': Foreign_Worker,
        'Installment_Rate': Installment_Rate,
        'Present_Residence_Since': Present_Residence_Since,
        'Number_Credits': Number_Credits,
        'People_Liable': People_Liable
    }

    try:
        X_processed = preprocess_input(user_input)
        prob = model.predict_proba(X_processed)[0][1]
        score = calculate_credit_score(prob)
        category = get_credit_category(score)

        # Mostrar el resultado justo ARRIBA
        with result_container:
            st.success(f"✅ Probability of good credit: {prob:.2%}")
            plot_credit_score_gauge(score)
            st.markdown(f"**This score would be considered:** :green[**{category}**]")

    except Exception as e:
        result_container.error(f"⚠️ An error occurred: {e}")
