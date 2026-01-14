import streamlit as st
import pandas as pd
from reprocessing sklearn.pimport StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import base64

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA LOADING AND MODEL TRAINING ---
@st.cache_data
def load_and_train(data_path):
    """Loads data, preprocesses it, and trains a RandomForest model."""
    try:
        column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        df = pd.read_csv(data_path, header=None, names=column_names, na_values='?')
    except FileNotFoundError:
        st.error(f"Error: The data file '{data_path}' was not found. Please make sure it's in the same folder as the app.")
        return None, None

    # Handle missing values by filling with the median of the column
    for col in ['ca', 'thal']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Binarize the target variable
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # Prepare data for modeling
    X = df.drop('target', axis=1)
    y = df['target']

    # Scale the features for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train a RandomForest model
    model = RandomForestClassifier(
        n_estimators=100,      # The number of decision trees in the forest
        max_depth=10,          # The maximum depth of each tree
        min_samples_leaf=2,    # The minimum number of samples required to be at a leaf node
        random_state=42        # Ensures reproducible results
    )
    model.fit(X_scaled, y)

    return scaler, model

# Load the scaler and model using your data file
scaler, model = load_and_train('heart_data.txt')


# --- USER INTERFACE (UI) ---
if scaler and model:
    st.title("❤️ Advanced Heart Disease Predictor")
    st.markdown("Enter patient details in the sidebar to get a prediction. The model is based on the Cleveland Clinic Foundation dataset.")

    # Glossary Expander
    with st.expander("Click here for a glossary of the medical terms used"):
        st.markdown("""
        * **Age:** The patient's age in years.
        * **Sex:** The patient's biological sex (Male or Female).
        * **Chest Pain Type (cp):** Describes the type of chest pain experienced (Type 1: typical angina, Type 2: atypical angina, Type 3: non-anginal pain, Type 4: asymptomatic).
        * **Resting Blood Pressure (trestbps):** The patient's blood pressure (in mm Hg) while at rest.
        * **Serum Cholesterol (chol):** The total amount of cholesterol in the blood (in mg/dl).
        * **Fasting Blood Sugar > 120 mg/dl (fbs):** Indicates if the patient's blood sugar is high after fasting, which can be a sign of diabetes.
        * **Resting ECG Results (restecg):** Results from an electrocardiogram taken at rest (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy).
        * **Maximum Heart Rate Achieved (thalach):** The highest heart rate the patient reached during a stress test.
        * **Exercise Induced Angina (exang):** Whether the patient experienced chest pain during exercise.
        * **ST Depression by Exercise (oldpeak):** A measurement from a stress test ECG indicating potential heart muscle strain.
        * **Slope of Peak Exercise ST Segment (slope):** Describes the slope of the ST segment on the ECG during peak exercise.
        * **Major Vessels Colored by Fluoroscopy (ca):** The number of major arteries (0-3) found to be blocked during an angiogram.
        * **Thalassemia (thal):** A result from a thallium stress test indicating blood flow to the heart (3: Normal, 6: Fixed defect, 7: Reversible defect).
        """)

    # Sidebar for user inputs
    st.sidebar.header("Patient Medical Details")
    st.sidebar.markdown("Use the sliders and selectors to input data.")

    def user_input_features():
        age = st.sidebar.slider('Age', 29, 77, 54, help="Patient's age in years.")
        sex = st.sidebar.radio('Sex', ('Male', 'Female'), help="Patient's gender.")
        cp = st.sidebar.selectbox('Chest Pain Type', (1, 2, 3, 4), format_func=lambda x: f'Type {x}', help="1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic.")
        trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 94, 200, 132, help="While at rest.")
        chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 126, 564, 246, help="Total cholesterol level.")
        fbs = st.sidebar.radio('Fasting Blood Sugar > 120 mg/dl', ('False', 'True'), help="Is the patient's fasting blood sugar higher than 120 mg/dl?")
        restecg = st.sidebar.selectbox('Resting ECG Results', (0, 1, 2), format_func=lambda x: {0:'Normal', 1:'ST-T wave abnormality', 2:'Probable or definite left ventricular hypertrophy'}[x])
        thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 150, help="During a stress test.")
        exang = st.sidebar.radio('Exercise Induced Angina', ('No', 'Yes'), help="Did the patient experience angina during exercise?")
        oldpeak = st.sidebar.slider('ST Depression by Exercise', 0.0, 6.2, 1.0, step=0.1, help="ST depression induced by exercise relative to rest.")
        slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', (1, 2, 3), format_func=lambda x: {1:'Upsloping', 2:'Flat', 3:'Downsloping'}[x])
        ca = st.sidebar.selectbox('Major Vessels Colored by Fluoroscopy', (0, 1, 2, 3), help="Number of major vessels (0-3) colored by fluoroscopy.")
        thal = st.sidebar.selectbox('Thalassemia', (3, 6, 7), format_func=lambda x: {3:'Normal', 6:'Fixed defect', 7:'Reversible defect'}[x], help="A blood disorder called thalassemia.")

        data = {
            'age': age, 'sex': 1 if sex == 'Male' else 0, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': 1 if fbs == 'True' else 0, 'restecg': restecg,
            'thalach': thalach, 'exang': 1 if exang == 'Yes' else 0, 'oldpeak': oldpeak,
            'slope': slope, 'ca': ca, 'thal': thal
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Main panel for displaying inputs and results
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Patient's Input:")
        st.dataframe(input_df.T.rename(columns={0: 'Values'}))

    with col2:
        st.subheader("Prediction:")
        if st.button('Get Prediction', type="primary"):
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)

            if prediction[0] == 1:
                st.error('**Result: High Risk of Heart Disease**', icon="💔")
            else:
                st.success('**Result: Low Risk of Heart Disease**', icon="❤️")

            st.subheader("Prediction Confidence:")
            prob_df = pd.DataFrame({
                'Risk Level': ['Low Risk', 'High Risk'],
                'Probability': [f"{p*100:.2f}%" for p in prediction_proba[0]]
            })
            st.table(prob_df)

    st.markdown("---")
    st.write("*Disclaimer: This prediction is based on a machine learning model and is not a substitute for professional medical advice.*")
else:
    st.error("Model could not be loaded. Please check the data file and code.")