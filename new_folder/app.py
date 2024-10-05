from flask import Flask, request, render_template
import pandas as pd
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

# Load the saved model, scaler, and label encoder
import pickle

# Use a raw string or double backslashes for the file path
file_path = r'InnovateHER\new_folder\gb_investment_recommendation_model copy.pkl'
# or 
# file_path = 'InnovateHER\\new_folder\\gb_investment_recommendation_model copy.pkl'

# Open the file in read-binary mode
with open(file_path, 'rb') as file:
    model = joblib.load(file)

#model = pickle.load('InnovateHER\new_folder\gb_investment_recommendation_model copy.pkl')
#scaler = joblib.load('scaler.pkl')
import pickle

# Use a raw string or double backslashes for the file path
file_path = r'InnovateHER\new_folder\scaler.pkl'
# or 
# file_path = 'InnovateHER\\new_folder\\gb_investment_recommendation_model copy.pkl'

# Open the file in read-binary mode
with open(file_path, 'rb') as file:
    scaler = joblib.load(file)

#label_enc_product = joblib.load('label_enc_product.pkl')
import pickle

# Use a raw string or double backslashes for the file path
file_path = r'InnovateHER\new_folder\label_enc_product.pkl'
# or 
# file_path = 'InnovateHER\\new_folder\\gb_investment_recommendation_model copy.pkl'

# Open the file in read-binary mode
with open(file_path, 'rb') as file:
    label_enc_product= joblib.load(file)

#new_model = joblib.load('linear_regression.pkl') 
import pickle

# Use a raw string or double backslashes for the file path
file_path = r'InnovateHER\new_folder\linear_regression.pkl'
# or 
# file_path = 'InnovateHER\\new_folder\\gb_investment_recommendation_model copy.pkl'

# Open the file in read-binary mode
with open(file_path, 'rb') as file:
    new_model = joblib.load(file)

# Load the expense data for prediction
#expense_data = pd.read_excel('new_data (1).xlsx')
import pandas as pd

# Absolute path to the Excel file
file_path = r'InnovateHER\new_folder\new_data (1).xlsx'

try:
    expense_data = pd.read_excel(file_path)
    #print(expense_data)  # Display the DataFrame
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
# Encode the categorical columns using previously defined encoders
month_encoder = LabelEncoder()
type_encoder = LabelEncoder()
gender_encoder = LabelEncoder()
expense_data['month_name_encoded'] = month_encoder.fit_transform(expense_data['month_name'])
expense_data['type_encoded'] = type_encoder.fit_transform(expense_data['type'])
expense_data['gender_encoded'] = gender_encoder.fit_transform(expense_data['gender'])

# Predict expenses for the latest month and year for each category
def get_predicted_expenses_new(expense_data, model):
    latest_year = expense_data['year'].max()
    latest_month = expense_data[expense_data['year'] == latest_year]['month_name_encoded'].max()
    categories = expense_data['type'].unique()
    predicted_expenses = {}

    for category in categories:
        category_encoded = type_encoder.transform([category])[0]
        input_data = pd.DataFrame([[latest_month, category_encoded]], columns=['month', 'type_encoded'])
        predicted_expense = model.predict(input_data)[0]
        predicted_expenses[category] = predicted_expense

    return predicted_expenses

# Allocate the budget based on predicted expenses
def allocate_budget(predicted_expenses, total_budget):
    total_expenses = sum(predicted_expenses.values())
    allocation_ratios = {category: expense / total_expenses for category, expense in predicted_expenses.items()}
    budget_allocation = {}
    allocated_budget = 0

    for category, ratio in allocation_ratios.items():
        category_budget = ratio * total_budget
        if allocated_budget + category_budget <= total_budget:
            budget_allocation[category] = category_budget
            allocated_budget += category_budget
        else:
            budget_allocation[category] = total_budget - allocated_budget
            break

    return budget_allocation
temp=10
count=0
def budget_split(d,budget,temp):
  global count
  
  temp+=count*5
  count+=1
  m_budget=budget-budget*temp/100
  tot_exp=sum(d.values())
  perc={}
  val={}
  for i in d.keys():
    perc[i]=d[i]/tot_exp
    val[i]=m_budget*perc[i]
  return val

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
def dashboard():
    return render_template('indexdashboard.html')

@app.route('/suggestion')
def suggestion():
    return render_template('suggestionmodelindex.html')

@app.route('/budget_expense')
def budget_expense():
    return render_template('budget_input.html')

@app.route('/allocation_result', methods=['POST'])
def allocate():
    total_budget = float(request.form['total_budget'])
    predicted_expenses = get_predicted_expenses_new(expense_data, new_model)
    allocation = budget_split(predicted_expenses, total_budget,temp)
    allocated=sum(allocation.values())
    return render_template('allocation_result.html', allocation=allocation, total_budget=total_budget,allocated=allocated)


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
