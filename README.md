# heart-disease-prediction-ml
This project implements an AI-based heart disease prediction system using machine learning models, including Random Forest, Linear Regression, and Sigmoid Function (Logistic Regression). The model is trained and evaluated using the University of Cleveland (UCI) Heart Disease dataset to accurately predict the likelihood of heart disease in patients.
# Heart Disease Risk Predictor ❤️

## Project Overview
This project is a comprehensive Machine Learning solution designed to assess cardiac health risks based on clinical parameters. It utilizes a **Random Forest Classifier** trained on the Cleveland Clinic Heart Disease dataset to provide high-accuracy binary classifications (High Risk vs. Low Risk).

The repository includes a production-ready **Streamlit web application**, an automated **unit testing framework**, and detailed **test case documentation**.

## Key Features
* **Machine Learning Model:** Implements a Random Forest ensemble model with feature scaling via `StandardScaler`.
* **Interactive UI:** A user-friendly Streamlit dashboard for real-time patient data input and risk assessment.
* **Robust Testing:** Includes a suite of 15+ automated test cases covering data integrity, model performance, and boundary conditions.
* **Automated Reporting:** A custom Python test runner that generates execution reports.

## Technical Stack
* **Language:** Python 3.x
* **ML Libraries:** Scikit-learn, Pandas, NumPy
* **Deployment/UI:** Streamlit
* **Testing:** Unittest, Docx-parser

## Dataset Information
[cite_start]The model is trained on the **UCI Cleveland Clinic Heart Disease dataset**, which contains 14 clinical attributes including age, sex, chest pain type (cp), cholesterol levels (chol), and maximum heart rate (thalach)[cite: 272, 274].

## How to Run
1. **Clone the repository:**
   `git clone https://github.com/YOUR_USERNAME/heart-disease-risk-predictor.git`
2. **Install dependencies:**
   `pip install -r requirements.txt`
3. **Launch the Web App:**
   `streamlit run app.py`
4. **Run Automated Tests:**
   `python test_heart_predictor.py`

## Test Results & Quality Assurance
The project underwent rigorous testing to ensure prediction reliability. 
* **Unit Tests:** Validated data loading, missing value handling (median imputation), and target binarization.
* **Accuracy:** The model achieves a performance rate significantly above random baseline on clinical test data.
* **Automated Reports:** Test execution results are captured in `test_report.txt`.

## Contact
**Abhishek Prajapati**
*Member, GDG AI Team*
