from flask import Flask, render_template, request
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# File path for storing the Excel file
EXCEL_FILE_PATH = 'data.xlsx'

# Initialize an empty dataframe if the Excel file does not exist
if not os.path.isfile(EXCEL_FILE_PATH):
    df = pd.DataFrame(columns=["date", "month", "year", "type", "amount"])
    df.to_excel(EXCEL_FILE_PATH, index=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    bot_response = ""
    
    if request.method == 'POST':
        if 'user_input' in request.form:
            # Handle user input (similar to previous code)
            user_input = request.form['user_input']
            # Process input and store to dataframe as needed (implement your logic here)
            # For example: Extract date, month, year, type, and amount and store in df
            # (Implement your logic for NLP classification and data extraction)
        
        if 'upload_file' in request.files:
            uploaded_file = request.files['upload_file']
            if uploaded_file.filename != '':
                # Load the uploaded Excel file into a dataframe
                new_data = pd.read_excel(uploaded_file)
                # Append the new data to the existing dataframe
                existing_data = pd.read_excel(EXCEL_FILE_PATH)
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                combined_data.to_excel(EXCEL_FILE_PATH, index=False)
                bot_response = "Data has been successfully appended to the Excel file."

    return render_template('index.html', bot_response=bot_response)

if __name__ == '__main__':
    app.run(debug=True)
