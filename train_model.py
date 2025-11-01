# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1️⃣ Load dataset
df = pd.read_csv('data/soil_data.csv')

# Example columns: pH, Nitrogen, Phosphorus, Potassium, Moisture, Soil_Type
X = df[['pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture']]
y = df['Soil_Type']

# 2️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4️⃣ Train model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train_scaled, y_train)

# 5️⃣ Save model & scaler
pickle.dump(model, open('models/soil_model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

print("✅ Model and Scaler saved successfully!")
