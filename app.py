import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64

# --- Load model, scaler, and feature order ---
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_order = joblib.load("model/features.pkl")

# --- Helper functions ---
def probability_to_score(p, min_score=300, max_score=900):
    return round(max_score - (p * (max_score - min_score)))

def score_category(score):
    if score <= 692:
        return "Poor"
    elif score <= 742:
        return "Fair"
    elif score <= 789:
        return "Good"
    elif score <= 832:
        return "Very Good"
    else:
        return "Excellent"

def category_tip(category):
    tip_texts = {
        "Poor": "Pay down overdue balances and avoid taking on new credit.",
        "Fair": "Reduce your credit utilization and make all payments on time.",
        "Good": "Maintain low balances and monitor your credit regularly.",
        "Very Good": "Great job! Keep up your consistent credit behavior.",
        "Excellent": "You're in excellent standing—keep doing what you're doing!"
    }
    tip = tip_texts.get(category, "")
    return f"📌 <strong>Tip:</strong> {tip}"

def plot_speedometer(score):
    # Define ranges, labels, and colors (reversed order)
    ranges = [(833, 900), (790, 832), (743, 789), (693, 742), (300, 692)]
    labels = ["Excellent", "Very Good", "Good", "Fair", "Poor"]
    colors = ["#27ae60", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Draw arcs
    for i, (start, end) in enumerate(ranges):
        theta1 = 180 - (180 * (end - 300) / 600)
        theta2 = 180 - (180 * (start - 300) / 600)
        wedge = patches.Wedge(center=(0, 0), r=1, theta1=theta1, theta2=theta2,
                              width=0.3, facecolor=colors[i], edgecolor='white')
        ax.add_patch(wedge)

    # Add labels
    for i, (start, end) in enumerate(ranges):
        theta1 = 180 - (180 * (end - 300) / 600)
        theta2 = 180 - (180 * (start - 300) / 600)
        angle_rad = np.deg2rad((theta1 + theta2) / 2)

        # Prevent overlap on the right
        if labels[i] in ["Very Good", "Excellent"]:
            distance = 1.18
            fontsize = 9
        else:
            distance = 1.1
            fontsize = 10

        x = np.cos(angle_rad) * distance
        y = np.sin(angle_rad) * distance

        ax.text(x, y, labels[i], ha='center', va='center',
                fontsize=fontsize)

    # Draw needle
    needle_theta = np.deg2rad(180 - 180 * (score - 300) / 600)
    ax.arrow(0, 0, np.cos(needle_theta) * 0.7, np.sin(needle_theta) * 0.7,
             width=0.015, head_width=0.05, head_length=0.05, fc='black', ec='black')

    # Clean plot
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(0, 1.2)
    ax.axis('off')
    ax.set_aspect('equal')

    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    return buf

# --- App title ---
st.markdown("<h1 style='text-align: center;'>📊 Credit Score Simulator</h1>", unsafe_allow_html=True)

# --- Output container ---
result_container = st.container()

# --- Input form ---
with st.form("credit_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("👤 Age", 18, 100, 35)
    with col2:
        monthly_income = st.number_input("💰 Monthly Income ($)", min_value=0, value=5000)

    col3, col4 = st.columns(2)
    with col3:
        debt_ratio = st.slider("📉 Debt-to-Income Ratio", 0.0, 2.0, 0.3)
    with col4:
        revolving_utilization = st.slider("💳 Revolving Utilization", 0.0, 2.0, 0.3)

    col5, col6, col7 = st.columns(3)
    with col5:
        open_credit_lines = st.number_input("📂 Open Credit Lines", 0, 20, 3)
    with col6:
        real_estate_loans = st.number_input("🏠 Real Estate Loans", 0, 10, 1)
    with col7:
        number_of_dependents = st.slider("👶 Dependents", 0, 10, 1)

    col8, col9, col10 = st.columns(3)
    with col8:
        times_30_59_days_late = st.slider("30–59 Days Late", 0, 10, 0)
    with col9:
        times_60_89_days_late = st.slider("60–89 Days Late", 0, 10, 0)
    with col10:
        times_90_days_late = st.slider("90+ Days Late", 0, 10, 0)

    submit_button = st.form_submit_button("Submit")

# --- Prediction and output ---
if submit_button:
    input_data = pd.DataFrame([{
        'age': age,
        'MonthlyIncome': monthly_income,
        'DebtToIncomeRatio': debt_ratio,
        'RevolvingUtilization': revolving_utilization,
        'RealEstateLoans': real_estate_loans,
        'OpenCreditLines': open_credit_lines,
        'Times30_59DaysLate': times_30_59_days_late,
        'Times60_89DaysLate': times_60_89_days_late,
        'Times90DaysLate': times_90_days_late,
        'NumberOfDependents': number_of_dependents
    }])

    # Derived features
    input_data["TotalPastDue"] = (
        input_data["Times30_59DaysLate"] +
        input_data["Times60_89DaysLate"] +
        input_data["Times90DaysLate"]
    )
    input_data["FinancialStressScore"] = debt_ratio * revolving_utilization
    input_data["CreditBurdenPerLine"] = monthly_income / (open_credit_lines + 1)
    input_data["AgeUtilizationRatio"] = age / (revolving_utilization + 0.01)
    input_data["IncomeAgeRatio"] = monthly_income / (age + 1)
    input_data["LinesPerYear"] = open_credit_lines / (age + 1)

    # Prediction
    input_data = input_data[feature_order]
    input_scaled = scaler.transform(input_data)
    prob_default = model.predict_proba(input_scaled)[:, 1][0]
    score = probability_to_score(prob_default)
    category = score_category(score)

    # Display results
    with result_container:
        st.markdown("<h4 style='text-align: center;'>Your Credit Score Result is: </h4>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='text-align: center; font-size: 72px; font-weight: bold; color: black;'>{score}</div>",
            unsafe_allow_html=True
        )
        st.markdown(f"<h4 style='text-align: center;'>Category: {category}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>🔍 Estimated Probability of Default: <strong>{prob_default:.2%}</strong></p>", unsafe_allow_html=True)

        # Speedometer chart
        img_buf = plot_speedometer(score)
        st.markdown(
            f"<div style='text-align: center'><img src='data:image/png;base64,{base64.b64encode(img_buf.getvalue()).decode()}'></div>",
            unsafe_allow_html=True
        )

        # Tip
        st.markdown(f"<br><p style='text-align: center;'>{category_tip(category)}</p>", unsafe_allow_html=True)
