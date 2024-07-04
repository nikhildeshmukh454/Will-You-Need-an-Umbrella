from flask import Flask, render_template, request
from main import RainPrediction  # Assuming RainPrediction class is in main.py
import pickle

# Load the RainPrediction object from the pickle file
with open('rain_prediction_model.pkl', 'rb') as file:
    Rain_pred = pickle.load(file)

app = Flask(__name__)

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and display prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    test_data = [
        request.form['MinTemp'],
        request.form['MaxTemp'],
        request.form['Rainfall'],
        request.form['WindGustSpeed'],
        request.form['WindSpeed9am'],
        request.form['WindSpeed3pm'],
        request.form['Humidity9am'],
        request.form['Humidity3pm'],
        request.form['Pressure9am'],
        request.form['Pressure3pm'],
        request.form['Temp9am'],
        request.form['Temp3pm'],
        request.form['RainToday']
    ]
    
    # Convert the data to the format needed by the model
    test_data = [float(i) for i in test_data]

    # Make prediction
    prediction = Rain_pred.predict(test_data)
    # prediction=1
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
