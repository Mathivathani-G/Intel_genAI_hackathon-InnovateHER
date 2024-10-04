from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved model, scaler, and label encoder
model = joblib.load('gb_investment_recommendation_model.pkl')
scaler = joblib.load('scaler.pkl')
label_enc_product = joblib.load('label_enc_product.pkl')

# Define the prediction function
def predict_investment(input_data):
    # Convert input_data into a DataFrame
    user_df = pd.DataFrame([input_data])
    
    # One-Hot Encode the input categorical variables
    user_df = pd.get_dummies(user_df, columns=['gender', 'individual_goal', 'risk_tolerance'], drop_first=True)

    # Align the user input with the training data columns
    for col in model.feature_names_in_:
        if col not in user_df.columns:
            user_df[col] = 0
    user_df = user_df[model.feature_names_in_]  # Ensure columns match the model's expectations

    # Scale numeric features (age and financial literacy)
    user_df[['Age', 'Financial Literacy']] = scaler.transform(user_df[['Age', 'Financial Literacy']])
    
    # Predict the recommended investment product
    prediction = model.predict(user_df)
    
    # Convert the prediction back to its original label using the target label encoder
    investment_product = label_enc_product.inverse_transform(prediction)
    
    return investment_product[0]

@app.route('/')
def index():
    return render_template('suggestionmodelindex.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = request.form['age']
    gender = request.form['gender']
    individual_goal = request.form['individual_goal']
    risk_tolerance = request.form['risk_tolerance']
    financial_literacy = request.form['financial_literacy']

    # Prepare the input data for prediction
    user_input = {
        'Age': float(age),
        'gender': gender,
        'individual_goal': individual_goal,
        'risk_tolerance': risk_tolerance,
        'Financial Literacy': float(financial_literacy)
    }

    # Make prediction
    predicted_investment = predict_investment(user_input)
    
    # Render result back to the webpage
    return render_template('suggestionmodelindex.html', prediction_text=f'Recommended Investment Product: {predicted_investment}')

if __name__ == '__main__':
    app.run(debug=True)
