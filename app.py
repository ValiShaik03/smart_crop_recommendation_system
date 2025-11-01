import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 🌾 PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Smart Crop Recommendation", page_icon="🌱", layout="centered")

st.title("🌾 Smart Crop Recommendation System for Farmers")
st.markdown("""
This app helps farmers find the **best crop to grow** 🌱  
based on their **soil nutrients**, **weather conditions**, and **rainfall** data.

👉 Just upload a **Kaggle Crop Recommendation Dataset (CSV)** and enter your soil values.
""")

# ----------------------------
# 📘 LEARN MORE SECTION
# ----------------------------
with st.expander("🌿 What do these terms mean? (Tap to Learn More)"):
    st.markdown("""
    - **Nitrogen (N):** 🧪 Helps plants grow green and leafy.  
      Too little → yellow leaves; too much → weak stems.
    - **Phosphorus (P):** 🌿 Builds strong roots and more flowers.
    - **Potassium (K):** 🍎 Helps plants produce healthy fruits and resist disease.
    - **Temperature (°C):** 🌤️ Affects plant growth and yield.
    - **Humidity (%):** 💧 Moisture in the air — important for seed growth.
    - **pH:** ⚖️ Soil acidity level (ideal: 6–7).
    - **Rainfall (mm):** ☔ Water availability for crops.
    """)

# ----------------------------
# 📤 UPLOAD DATASET
# ----------------------------
uploaded_file = st.file_uploader("📂 Upload your Crop Dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head())

    # Detect feature columns
    possible_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    feature_cols = [col for col in possible_features if col in df.columns]
    target_col = 'label' if 'label' in df.columns else None

    if not feature_cols or not target_col:
        st.error("⚠️ Dataset must contain columns: N, P, K, temperature, humidity, ph, rainfall, and label.")
    else:
        st.info(f"✅ Features: {', '.join(feature_cols)} | Target: {target_col}")

        # ----------------------------
        # 🧠 TRAIN MODEL
        # ----------------------------
        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_train_scaled, y_train)
        acc = model.score(X_test_scaled, y_test)

        st.success(f"🌱 Model trained successfully with accuracy: **{acc*100:.2f}%**")

        # ----------------------------
        # 📊 MODEL EVALUATION
        # ----------------------------
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.subheader("📈 Model Evaluation Metrics")
        fig_cm, ax_cm = plt.subplots(figsize=(8, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig_cm)
        st.dataframe(report_df.style.background_gradient(cmap="Greens"))

        # ----------------------------
        # 🌿 FARMER-FRIENDLY INPUTS
        # ----------------------------
        st.markdown("### 🌾 Enter Your Soil and Weather Values")

        feature_descriptions = {
            "N": ("Nitrogen (N)", "🧪 Helps leaves grow green and healthy. (Range: 0–140)"),
            "P": ("Phosphorus (P)", "🌿 Helps strong roots and flowering. (Range: 5–145)"),
            "K": ("Potassium (K)", "🍎 Helps fruits and plant strength. (Range: 5–205)"),
            "temperature": ("Temperature (°C)", "🌤️ Average temperature of your area."),
            "humidity": ("Humidity (%)", "💧 Moisture in air — affects growth."),
            "ph": ("Soil pH", "⚖️ Ideal between 6.0 and 7.5 for most crops."),
            "rainfall": ("Rainfall (mm)", "☔ Average seasonal rainfall.")
        }

        user_inputs = {}
        for col in feature_cols:
            label, help_text = feature_descriptions.get(col, (col, ""))
            user_inputs[col] = st.number_input(
                f"{label}",
                min_value=float(df[col].min()),
                max_value=float(df[col].max()),
                value=float(df[col].mean()),
                help=help_text
            )

        # ----------------------------
        # 🌾 PREDICTION
        # ----------------------------
        if st.button("🔍 Suggest the Best Crop"):
            input_data = np.array([[user_inputs[col] for col in feature_cols]])
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)[0]

            st.success(f"🌾 Based on your soil and weather conditions, you should grow: **{prediction.upper()}**")

            st.markdown("""
            ### 🌱 Recommendation Summary:
            - ✅ Soil nutrients are suitable for this crop.
            - 💧 Ensure proper watering based on rainfall.
            - 🌤️ Temperature and pH are favorable for good yield.
            - 📈 Maintain these levels for better productivity.
            """)

            # Visualization
            st.subheader("📊 Your Soil Condition Overview")
            fig, ax = plt.subplots()
            ax.bar(feature_cols, [user_inputs[col] for col in feature_cols], color="green", alpha=0.7)
            ax.set_ylabel("Value")
            plt.xticks(rotation=45)
            st.pyplot(fig)

else:
    st.info("⬆️ Upload a dataset to start using the system.")

st.markdown("---")
st.caption("👨‍🌾 Developed by Shaik Mahaboob Vali | AI/ML Crop Suggestion App | Streamlit")
