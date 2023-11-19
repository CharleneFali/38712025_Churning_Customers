# app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import load_model
import joblib
from keras.layers import Input, Dense
from keras.models import Model
# Load the pre-trained model
model_path = 'Churn_model.joblib'
churn_model = joblib.load(model_path)

def create_mlp_model(input_shape, optimizer='adam'):
    inputs = Input(shape=(input_shape,))
    hidden_layer1 = Dense(64, activation='relu')(inputs)
    hidden_layer2 = Dense(32, activation='relu')(hidden_layer1)
    hidden_layer3 = Dense(16, activation='relu')(hidden_layer2)
    output_layer = Dense(1, activation='sigmoid')(hidden_layer3)

    model = Model(inputs=inputs, outputs=output_layer)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Function to preprocess input data
def preprocess_input(data):
    # Assuming data is a dictionary with keys corresponding to feature names
    df = pd.DataFrame(data, index=[0])
    
    # Encode categorical variables
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.transform(df[column])
    
    # Select numeric features
    numeric_features = ['TotalCharges', 'MonthlyCharges', 'tenure', 'SeniorCitizen']
    X_numeric = df[numeric_features]
    
    # Scale numeric features
    X_scaled = scaler.transform(X_numeric)
    
    # Convert the scaled array to a DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_features)
    
    # Select features
    selected_features = ['TotalCharges', 'MonthlyCharges', 'tenure', 'SeniorCitizen', 'Partner', 'Dependents', 'Contract']
    X_selected = pd.concat([X_scaled_df, df[selected_features]], axis=1)
    
    return X_selected

# LabelEncoder and StandardScaler for preprocessing
label_encoder = LabelEncoder()
scaler = StandardScaler()

# Streamlit app
def main():
    st.title("Customer Churn Prediction App")

    # Collect input features from the user
    total_charges = st.number_input("Total Charges")
    monthly_charges = st.number_input("Monthly Charges")
    tenure = st.number_input("Tenure")
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ['No', 'Yes'])
    dependents = st.selectbox("Dependents", ['No', 'Yes'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])

    # Create a dictionary with input data
    input_data = {
        'TotalCharges': total_charges,
        'MonthlyCharges': monthly_charges,
        'tenure': tenure,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'Contract': contract
    }

    # Preprocess input data
    X_input = preprocess_input(input_data)

    # Make predictions
    prediction = churn_model.predict(X_input)
    confidence = churn_model.predict_proba(X_input)[:, 1]

    st.subheader("Prediction:")
    if prediction[0] == 0:
        st.success("No Churn (Customer will not churn)")
    else:
        st.warning("Churn (Customer will churn)")

    st.subheader("Confidence:")
    st.write(f"The model is {confidence[0] * 100:.2f}% confident in this prediction.")

if __name__ == "__main__":
    main()
