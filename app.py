import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------
# App Configuration
# --------------------------------------
st.set_page_config(page_title="Soil Classification & Crop Suggestion", page_icon="🌾", layout="wide")

st.title("🌱 Soil Classification & Crop Suggestion App")
st.markdown("""
Welcome to the **AI-Powered Soil & Crop Recommendation System**!  
Use this app to:
- 🧠 Analyze soil nutrients (NPK, pH, moisture, etc.)
- 🌾 Predict the most suitable crop for your soil
- 📈 Help farmers boost yield using AI-driven insights
""")

# --------------------------------------
# Step 1: Dataset Selection
# --------------------------------------
st.header("📂 Choose Dataset")

dataset_option = st.radio(
    "Select dataset source:",
    ("Default Dataset", "Upload Your Own Dataset")
)

if dataset_option == "Default Dataset":
    df = pd.read_csv("data/Crop_recommendation.csv")
    st.success("✅ Default dataset loaded successfully!")
else:
    uploaded_file = st.file_uploader("📥 Upload your dataset (CSV format)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Uploaded dataset loaded successfully!")
    else:
        st.warning("⚠️ Please upload a dataset to continue.")
        st.stop()

# Clean column names
df.columns = [c.strip().lower() for c in df.columns]

# Rename common variations automatically
rename_map = {
    'nitrogen': 'n',
    'phosphorus': 'p',
    'potassium': 'k',
    'temp': 'temperature',
    'moisture': 'humidity',
    'crop': 'label',
    'crop_name': 'label'
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------
# Step 2: Model Training
# --------------------------------------
st.header("⚙️ Model Training ")

# Detect label column
possible_labels = ['label', 'crop', 'target']
label_col = None
for col in possible_labels:
    if col in df.columns:
        label_col = col
        break

if label_col is None:
    st.error("❌ Could not find the target column (label/crop). Please check your dataset.")
    st.stop()

X = df.drop(columns=[label_col])
y = df[label_col]

# Encode categorical columns automatically
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    st.warning(f"⚠️ Encoding categorical columns: {', '.join(categorical_cols)}")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Encode target labels if needed
label_encoder = None
if y.dtypes == 'object':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y.astype(str))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
accuracy = model.score(X_test_scaled, y_test)

st.success(f"✅ Model trained successfully with accuracy: **{accuracy * 100:.2f}%**")

# Save model & scaler
os.makedirs("models", exist_ok=True)
with open("models/soil_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

if label_encoder:
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

# --------------------------------------
# Step 3: User Input for Prediction
# --------------------------------------
# --------------------------------------
# Step 3: User Input for Prediction
# --------------------------------------
st.header("🌾 Enter Soil Values for Crop Suggestion")

st.markdown("Enter your soil parameters below (hover over ❓ to understand each term):")

# Helpful descriptions for each column
param_help = {
    "n": "Nitrogen (N): Supports leaf and stem growth. 🧪 Typical range: 0 – 140 mg/kg.",
    "p": "Phosphorus (P): Promotes roots and flowers. 🧪 Typical range: 5 – 145 mg/kg.",
    "k": "Potassium (K): Strengthens stems, improves disease resistance. 🧪 Typical range: 5 – 205 mg/kg.",
    "temperature": "Temperature (°C): Average soil temperature. 🌡️ Ideal range: 15 – 35 °C.",
    "humidity": "Humidity (%): Moisture content in soil or air. 💧 Ideal range: 40 – 90 %.",
    "ph": "pH Level: Measures soil acidity/alkalinity. ⚖️ Ideal range: 5.5 – 7.5.",
    "rainfall": "Rainfall (mm): Average rainfall during crop growth. 🌧️ Typical range: 20 – 300 mm."
}

numeric_cols = X.columns.tolist()
user_data = {}

col1, col2, col3 = st.columns(3)
for i, col in enumerate(numeric_cols):
    help_text = param_help.get(col.lower(), "Enter value for this feature.")
    if i % 3 == 0:
        user_data[col] = col1.number_input(f"{col.upper()}", value=float(df[col].mean()), help=help_text)
    elif i % 3 == 1:
        user_data[col] = col2.number_input(f"{col.upper()}", value=float(df[col].mean()), help=help_text)
    else:
        user_data[col] = col3.number_input(f"{col.upper()}", value=float(df[col].mean()), help=help_text)

if st.button("🌿 Suggest Best Crop"):
    input_data = np.array([list(user_data.values())]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Decode label if label encoder exists
    if label_encoder:
        crop_name = label_encoder.inverse_transform([prediction])[0]
    else:
        crop_name = str(prediction)

    st.success(f"🌾 Based on the given soil data, the recommended crop is: **{crop_name.upper()}**")
