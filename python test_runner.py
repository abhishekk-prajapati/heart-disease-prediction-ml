import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import docx
import re
from datetime import datetime

# --- PART 1: MODEL AND SCALER LOADING ---
# This section is adapted from your app.py to load the trained model and scaler.

def load_and_train(data_path):
    """Loads data, preprocesses it, and trains a RandomForest model."""
    try:
        column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        df = pd.read_csv(data_path, header=None, names=column_names, na_values='?')
    except FileNotFoundError:
        print(f"Error: The data file '{data_path}' was not found.")
        return None, None

    for col in ['ca', 'thal']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=2, random_state=42
    )
    model.fit(X_scaled, y)
    return scaler, model

# --- PART 2: TEST CASE PARSER ---
# This function reads the .docx file and extracts test cases.

def parse_test_cases(docx_path):
    """Parses test cases from a .docx file."""
    try:
        doc = docx.Document(docx_path)
    except Exception as e:
        print(f"Error opening '{docx_path}': {e}")
        return []
        
    test_cases = []
    current_case = None
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Check for the start of a new test case
        if text.startswith('TC-'):
            if current_case:
                test_cases.append(current_case)
            current_case = {'id': text.split('[')[0].strip(), 'inputs': {}}

        elif 'Expected Output:' in text and current_case:
            current_case['expected_output'] = text.split(':')[1].strip()

        # Parse input parameters
        elif ':' in text and current_case and 'inputs' in current_case:
            # Simple regex to split key-value pairs
            match = re.match(r'([^:]+):\s*(.*)', text)
            if match:
                key, value = match.groups()
                # Clean up key from any source tags
                key = key.split(']')[-1].strip()
                current_case['inputs'][key] = value.strip()

    if current_case:
        test_cases.append(current_case)
        
    return test_cases

# --- PART 3: TEST EXECUTION ENGINE ---
# This function runs a single test case against the model.

def run_single_test(scaler, model, test_case):
    """Prepares input data and runs prediction for one test case."""
    # Convert text inputs to numerical format, as done in the Streamlit app
    inputs = test_case['inputs']
    data = {
        'age': int(inputs['age']),
        'sex': 1 if inputs['sex'] == 'Male' else 0,
        'cp': int(inputs['cp']),
        'trestbps': int(inputs['trestbps']),
        'chol': int(inputs['chol']),
        'fbs': 1 if inputs['fbs'] == 'True' else 0,
        'restecg': int(inputs['restecg']),
        'thalach': int(inputs['thalach']),
        'exang': 1 if inputs['exang'] == 'Yes' else 0,
        'oldpeak': float(inputs['oldpeak']),
        'slope': int(inputs['slope']),
        'ca': int(inputs['ca']),
        'thal': int(inputs['thal'])
    }
    
    # Create a DataFrame with the correct column order
    feature_columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    input_df = pd.DataFrame(data, index=[0])[feature_columns]
    
    # Scale the input and make a prediction
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    
    # Convert numerical prediction back to text for comparison
    actual_output = 'High Risk' if prediction[0] == 1 else 'Low Risk'
    
    # Determine result
    status = 'Pass' if actual_output == test_case['expected_output'] else 'Fail'
    
    return actual_output, status

# --- PART 4: MAIN EXECUTION AND REPORTING ---

def main():
    """Main function to load model, parse tests, run them, and generate a report."""
    print("🚀 Starting Heart Disease Predictor Test Automation...")
    
    # Load model
    scaler, model = load_and_train('heart_data.txt')
    if not scaler or not model:
        print("❌ Model loading failed. Aborting tests.")
        return

    print("✅ Model and Scaler loaded successfully.")

    # Parse test cases
    test_cases = parse_test_cases('heart_test_cases_15.docx')
    if not test_cases:
        print("❌ No test cases found or file could not be read. Aborting.")
        return
        
    print(f"✅ Found {len(test_cases)} test cases to execute.")
    print("-" * 50)
    
    results = []
    passed_count = 0
    
    # Execute each test case
    for case in test_cases:
        actual, status = run_single_test(scaler, model, case)
        results.append({
            'id': case['id'],
            'expected': case['expected_output'],
            'actual': actual,
            'status': status
        })
        if status == 'Pass':
            passed_count += 1
            
    # Generate the report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("          HEART DISEASE PREDICTION - TEST REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n--- Individual Test Case Results ---\n")

    for res in results:
        report_lines.append(
            f"ID: {res['id']:<8} | Status: {res['status']:<4} | "
            f"Expected: {res['expected']:<10} | Actual: {res['actual']:<10}"
        )

    total_tests = len(results)
    failed_count = total_tests - passed_count
    pass_rate = (passed_count / total_tests) * 100 if total_tests > 0 else 0

    summary_header = "\n--- Summary ---"
    summary_body = (
        f"Total Tests Run: {total_tests}\n"
        f"✅ Passed:         {passed_count}\n"
        f"❌ Failed:         {failed_count}\n"
        f"📊 Pass Rate:      {pass_rate:.2f}%"
    )
    
    report_lines.append(summary_header)
    report_lines.append(summary_body)
    report_lines.append("=" * 60)

    final_report = "\n".join(report_lines)

    # Print report to console
    print(final_report)

    # Save report to a file
    with open("test_report.txt", "w") as f:
        f.write(final_report)
        
    print(f"\n📄 Report saved to test_report.txt")

if __name__ == "__main__":
    main()