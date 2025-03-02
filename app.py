# MODEL DEPLOYMENT WITH FLASK
# --------------------------------

# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('churn_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    form_data = request.form.to_dict()
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'gender': [1 if form_data.get('gender') == 'Male' else 0],
        'SeniorCitizen': [int(form_data.get('seniorcitizen', 0))],
        'Partner': [1 if form_data.get('partner') == 'Yes' else 0],
        'Dependents': [1 if form_data.get('dependents') == 'Yes' else 0],
        'tenure': [int(form_data.get('tenure', 0))],
        'PhoneService': [1 if form_data.get('phoneservice') == 'Yes' else 0],
        'MultipleLines': [form_data.get('multiplelines', 'No')],
        'InternetService': [form_data.get('internetservice', 'No')],
        'OnlineSecurity': [form_data.get('onlinesecurity', 'No')],
        'OnlineBackup': [form_data.get('onlinebackup', 'No')],
        'DeviceProtection': [form_data.get('deviceprotection', 'No')],
        'TechSupport': [form_data.get('techsupport', 'No')],
        'StreamingTV': [form_data.get('streamingtv', 'No')],
        'StreamingMovies': [form_data.get('streamingmovies', 'No')],
        'Contract': [form_data.get('contract', 'Month-to-month')],
        'PaperlessBilling': [1 if form_data.get('paperlessbilling') == 'Yes' else 0],
        'PaymentMethod': [form_data.get('paymentmethod', 'Electronic check')],
        'MonthlyCharges': [float(form_data.get('monthlycharges', 0))],
        'TotalCharges': [float(form_data.get('totalcharges', 0))]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]
    
    # Format output
    result = "Customer will churn" if prediction[0] == 1 else "Customer will stay"
    confidence = f"{prediction_proba[0]*100:.2f}%"
    
    return render_template('index.html', 
                          prediction_text=f'Prediction: {result}', 
                          prediction_probability=f'Confidence: {confidence}',
                          form_data=form_data)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    # For API calls (e.g., from other applications)
    data = request.get_json(force=True)
    
    # Convert to DataFrame
    input_df = pd.DataFrame(data, index=[0])
    
    # Make prediction
    prediction = model.predict(input_df)
    output = int(prediction[0])
    probability = float(model.predict_proba(input_df)[:, 1][0])
    
    return jsonify({
        'prediction': output,
        'probability': probability,
        'status': 'Customer will churn' if output == 1 else 'Customer will stay'
    })

# Create templates folder and HTML file
os.makedirs('templates', exist_ok=True)

# Write the HTML template
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Customer Churn Prediction</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            padding: 30px;
            background-color: #f8f9fa;
        }
        .prediction-result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
        }
        .prediction-positive {
            background-color: #f8d7da;
            color: #721c24;
        }
        .prediction-negative {
            background-color: #d4edda;
            color: #155724;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4 text-center">Customer Churn Prediction</h1>
        
        {% if prediction_text %}
        <div class="prediction-result {% if 'will churn' in prediction_text %}prediction-positive{% else %}prediction-negative{% endif %}">
            <h3>{{ prediction_text }}</h3>
            <p>{{ prediction_probability }}</p>
        </div>
        {% endif %}
        
        <div class="card mt-4">
            <div class="card-header">
                <h4>Enter Customer Details</h4>
            </div>
            <div class="card-body">
                <form action="{{ url_for('predict') }}" method="post">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="form-group">
                                <label>Gender:</label>
                                <select name="gender" class="form-control">
                                    <option value="Male" {% if form_data and form_data.gender == 'Male' %}selected{% endif %}>Male</option>
                                    <option value="Female" {% if form_data and form_data.gender == 'Female' %}selected{% endif %}>Female</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>Senior Citizen:</label>
                                <select name="seniorcitizen" class="form-control">
                                    <option value="0" {% if form_data and form_data.seniorcitizen == '0' %}selected{% endif %}>No</option>
                                    <option value="1" {% if form_data and form_data.seniorcitizen == '1' %}selected{% endif %}>Yes</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>Partner:</label>
                                <select name="partner" class="form-control">
                                    <option value="Yes" {% if form_data and form_data.partner == 'Yes' %}selected{% endif %}>Yes</option>
                                    <option value="No" {% if form_data and form_data.partner == 'No' %}selected{% endif %}>No</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>Dependents:</label>
                                <select name="dependents" class="form-control">
                                    <option value="Yes" {% if form_data and form_data.dependents == 'Yes' %}selected{% endif %}>Yes</option>
                                    <option value="No" {% if form_data and form_data.dependents == 'No' %}selected{% endif %}>No</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>Tenure (months):</label>
                                <input type="number" name="tenure" class="form-control" min="0" max="72" value="{{ form_data.tenure if form_data else 0 }}">
                            </div>
                        </div>
                        
                        <div class="col-md-6">
                            <div class="form-group">
                                <label>Contract:</label>
                                <select name="contract" class="form-control">
                                    <option value="Month-to-month" {% if form_data and form_data.contract == 'Month-to-month' %}selected{% endif %}>Month-to-month</option>
                                    <option value="One year" {% if form_data and form_data.contract == 'One year' %}selected{% endif %}>One year</option>
                                    <option value="Two year" {% if form_data and form_data.contract == 'Two year' %}selected{% endif %}>Two year</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>Payment Method:</label>
                                <select name="paymentmethod" class="form-control">
                                    <option value="Electronic check" {% if form_data and form_data.paymentmethod == 'Electronic check' %}selected{% endif %}>Electronic check</option>
                                    <option value="Mailed check" {% if form_data and form_data.paymentmethod == 'Mailed check' %}selected{% endif %}>Mailed check</option>
                                    <option value="Bank transfer (automatic)" {% if form_data and form_data.paymentmethod == 'Bank transfer (automatic)' %}selected{% endif %}>Bank transfer (automatic)</option>
                                    <option value="Credit card (automatic)" {% if form_data and form_data.paymentmethod == 'Credit card (automatic)' %}selected{% endif %}>Credit card (automatic)</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>Internet Service:</label>
                                <select name="internetservice" class="form-control">
                                    <option value="DSL" {% if form_data and form_data.internetservice == 'DSL' %}selected{% endif %}>DSL</option>
                                    <option value="Fiber optic" {% if form_data and form_data.internetservice == 'Fiber optic' %}selected{% endif %}>Fiber optic</option>
                                    <option value="No" {% if form_data and form_data.internetservice == 'No' %}selected{% endif %}>No</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>Monthly Charges ($):</label>
                                <input type="number" name="monthlycharges" class="form-control" min="0" step="0.01" value="{{ form_data.monthlycharges if form_data else 0 }}">
                            </div>
                            
                            <div class="form-group">
                                <label>Total Charges ($):</label>
                                <input type="number" name="totalcharges" class="form-control" min="0" step="0.01" value="{{ form_data.totalcharges if form_data else 0 }}">
                            </div>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn btn-primary btn-block">Predict Churn</button>
                </form>
            </div>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
    ''')

if __name__ == '__main__':
    app.run(debug=True)