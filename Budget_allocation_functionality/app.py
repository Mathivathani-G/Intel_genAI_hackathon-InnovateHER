from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

# Load the pre-trained model
model_path = 'linear_regression.pkl'
from joblib import load
new_model = load('linear_regression.pkl') 


# Load the expense data for prediction
expense_data = pd.read_excel('new_data (1).xlsx')

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

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

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

if __name__ == '__main__':
    app.run(debug=True)
