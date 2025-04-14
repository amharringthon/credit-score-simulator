# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Cargar modelo, scaler y columnas
model = joblib.load("model/log_reg_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# Función para procesar el input
def preprocess_input(user_input):
    df = pd.DataFrame([user_input])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    df_scaled = scaler.transform(df)
    return df_scaled

# Función para calcular el credit score
def calculate_credit_score(prob):
    return int(300 + prob * (900 - 300))

# Gráfico del score
def plot_credit_score(score):
    ranges = [300, 560, 660, 725, 760, 900]
    labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#1a9850']

    category = ''
    for i in range(len(ranges) - 1):
        if ranges[i] <= score < ranges[i + 1]:
            category = labels[i]
            break
    else:
        category = labels[-1]

    fig, ax = plt.subplots(figsize=(10, 2.5))
    for i in range(len(ranges) - 1):
        ax.barh(0, ranges[i + 1] - ranges[i], left=ranges[i], color=colors[i], edgecolor='black', height=0.5)

    ax.plot(score, 0, 'k^', markersize=15)
    ax.text(score, 0.3, f'{score}', ha='center', fontsize=12, fontweight='bold')

    for i in range(len(labels)):
        midpoint = (ranges[i] + ranges[i + 1]) / 2
        ax.text(midpoint, -0.55, labels[i], ha='center', fontsize=10)

    ax.set_xlim(300, 900)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    ax.set_title('📊 Simulated Canadian Credit Score', fontsize=14, fontweight='bold')

    st.pyplot(fig)
    st.markdown(f"**This score would be considered: `{category}`**")

# Interfaz
st.title("Canadian Credit Risk Simulator 🇨🇦")
st.write("Complete the form below to estimate a simulated credit score.")

# Formulario de entrada (puedes agregar más si quieres)
with st.form("credit_form"):
    Duration_Months = st.slider('Credit Duration (Months)', 4, 72, 24)
    Credit_Amount = st.number_input('Credit Amount', min_value=250, max_value=20000, value=5000)
    Status_Checking_Account = st.selectbox('Checking Account Status', ['no checking account', '< 0 DM', '0 <= ... < 200 DM', '>= 200 DM / salary assignment'])
    Purpose = st.selectbox('Credit Purpose', ['radio/TV', 'car (new)', 'car (used)', 'furniture/equipment', 'business', 'education', 'repairs', 'domestic appliances', 'others', 'retraining'])
    Employment_Since = st.selectbox('Employment Duration', ['unemployed', '< 1 year', '1 <= ... < 4 years', '4 <= ... < 7 years', '>= 7 years'])
    Foreign_Worker = st.radio('Are you a foreign worker?', ['yes', 'no'])
    
    submitted = st.form_submit_button("Simulate")

if submitted:
    user_input = {
        'Duration_Months': Duration_Months,
        'Credit_Amount': Credit_Amount,
        'Status_Checking_Account': Status_Checking_Account,
        'Purpose': Purpose,
        'Employment_Since': Employment_Since,
        'Foreign_Worker': Foreign_Worker
    }

    X_processed = preprocess_input(user_input)
    prob = model.predict_proba(X_processed)[0][1]
    score = calculate_credit_score(prob)

    st.success(f"✅ Probability of good credit: {prob:.2%}")
    st.markdown("---")
    plot_credit_score(score)
