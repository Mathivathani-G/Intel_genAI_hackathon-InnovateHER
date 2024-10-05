from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline
import datetime
import re

app = Flask(__name__)

# Load the pre-trained model
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name)
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Store the last five messages and responses
history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    bot_response = ""
    if request.method == 'POST':
        user_input = request.form['user_input']
        
        # Check if the 'show' button was pressed
        if request.form.get('show'):
            # Show the last five entries
            bot_response = "\n".join(history[-5:]) if history else "No history available."
        else:
            # Process the input normally
            date_info = extract_date(user_input)
            amount = extract_amount(user_input)
            bot_response = f"Extracted Amount: {amount}, Date: {date_info}"
            # Store the user input and bot response in history
            history.append(f"User: {user_input} | Bot: {bot_response}")

    return render_template('index.html', bot_response=bot_response)

def extract_amount(user_input):
    # Extract the amount from the user input
    numbers = re.findall(r'\d+', user_input)
    return numbers[-1] if numbers else 'No amount found'

def extract_date(user_input):
    today = datetime.datetime.now()
    if "today" in user_input.lower():
        return today.strftime("%Y-%m-%d")
    elif "yesterday" in user_input.lower():
        yesterday = today - datetime.timedelta(days=1)
        return yesterday.strftime("%Y-%m-%d")
    else:
        # Check for other date formats in the input
        match = re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})', user_input, re.IGNORECASE)
        if match:
            day, month, year = match.groups()
            month_number = datetime.datetime.strptime(month, "%B").month
            return f"{year}-{month_number:02d}-{day.zfill(2)}"

        match = re.search(r'(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})', user_input)
        if match:
            day, month, year = match.groups()
            if len(year) == 2:  # Handle two-digit year
                year = '20' + year
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    return 'No valid date found'

if __name__ == '__main__':
    app.run(debug=True)
